# Copyright (c) 2023 Thomas Palmeira Ferraz, Helene Maxcici, Teysir Baoueb
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
import gdown
import logging

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from functools import partial
import copy

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict, Audio, Dataset
import evaluate
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, 
                          WhisperProcessor, WhisperForConditionalGeneration)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
from transformers.models.whisper.english_normalizer import (BasicTextNormalizer,  
                                                            EnglishTextNormalizer)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import is_datasets_available

import audio_degrader as ad

from data_utils import (prepare_audio, apply_degradation,
                                prepare_dataset,
                                DataCollator,
                                DataCollatorwithDegradation,
                                evaluate_robustness)

# Utils
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

lang_to_whisper = {"gl":"Galician", 
                   "fr":"French", 
                   "fa":"Persian", 
                   "en": "English",
                  "es": "Spanish"}
lang_to_id = {"gl":"gl_es", 
              "fr":"fr_fr", 
              "fa":"fa_ir", 
              "en": "en_us",
              "es": "es_419",
              "libri_en":"clean"}

def load_finetuned(size="tiny", language="French"):
  """Download finetuned weights from drive"""
  # Create folder to save results
  if not os.path.isdir("./finetuned"):
    os.mkdir("./finetuned")
  # Get current folder
  dirname = os.path.dirname(__file__)
  # Load dictionary of ids
  with open(dirname+"/finetuned/finetuned_models_ids.json") as file:
    dict_finetuned = json.load(file)  
  # Download 
  for f, file_id in dict_finetuned[size][language].items():
    gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t",
                 output=f"./finetuned/{f}",
                 use_cookies=False)


def text_normalizer(normalize = "none", spelling_normalizer = None):
  """
  Create text normalizer
  """
  # Get text normalizer
  if normalize == "english_normalizer":
    assert spelling_normalizer is not None
    normalizer = EnglishTextNormalizer(spelling_normalizer)
  elif normalize == "basic_normalizer":
    normalizer = BasicTextNormalizer(remove_diacritics=False,
                                          split_letters=False)
  elif normalize == "lower":
    normalizer = lambda x : x.lower()
  else:
    # No normalization
    normalizer = None
  
  return normalizer

class Metrics:
  """
  Computes metrics with text normalization option.
  """
  def __init__(self, tokenizer, metric_wer, normalize = "none"):
    """
    normalize:
     if english_normalizer, applies full english normalizer
     if basic_normalizer, applies subset of rules for all languages
     if lowe, only lower case the text
     else no normalization is applied
    """
    self.tokenizer = tokenizer
    self.metric_wer = metric_wer
    self.normalizer = text_normalizer(normalize, 
                                tokenizer.english_spelling_normalizer)

  def __call__(self, pred, per_sample = False):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id to be removed
    label_ids[label_ids == -100] = self.tokenizer.pad_token_id  
    # remove special tokens
    pred_list = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_list = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # normalize
    if self.normalizer is not None:
      pred_list = [self.normalizer(text) for text in pred_list]
      label_list = [self.normalizer(text) for text in label_list]

    if not per_sample:
      wer = 100 * self.metric_wer.compute(predictions=pred_list, 
                                          references=label_list)
      return {"wer": wer}
    
    else:
      wer_list = []
      for pred, label in zip(pred_list, label_list):
        wer = self.metric_wer.compute(predictions=[pred], 
                                      references=[label],
                                      concatenate_texts=True)
        wer_list.append(round(100 * wer,2))
      return {"wer": wer_list,
              "predictions": pred_list,
              "references": label_list}

class Seq2SeqTrainerwithAugmentation(Seq2SeqTrainer):
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = copy.deepcopy(self.data_collator)
        data_collator.list_degradations = None

        if is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def main():

    args = arg_parse()
    
    ### Load datasets ###
    if args.dataset == "librispeech_asr":
      args.lang = "libri_en"
    dataset = DatasetDict()
    # Load train and validation splits for finetuning
    if args.train:
      dataset["train"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                      split="train")
      dataset["val"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                      split="validation")
    # Load test set
    dataset["test"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                   split="test")
    ### Debug settings ###
    if args.debug:
      print("\nDebug Mode.")
      for s, d in dataset.items():
        dataset[s] = dataset[s].select(list(range(0, 10)))
    
    print(dataset)

    ### Load pretrained/finetuned ###
    # Load finetuned weights and configurations from drive
    if args.finetuned:
      assert args.size in ["tiny", "base"], (
            "Supported finetuned model sizes are tiny and base.")
      try:
        load_finetuned(args.size, lang_to_whisper[args.lang])
      except Exception as e:
        print("\nFailed to download finetuned weights from Drive." 
              "Please refresh or try later.\n")
        print(e)
        exit()
      # Set path of loaded model
      model_name_or_path = "./finetuned"
      # Read configurations
      with open("./finetuned/config.json") as file:
        config = json.load(file)  
      architecture = config['_name_or_path']
      print(f"\nLoaded model: finetuned/{args.size}/{args.lang}\n")
    else:
      # Load pretrained weights from Hugging Face
      if args.model_name_or_path is None:
        # Set path of loaded model
        model_name_or_path = "openai/whisper-"+args.size
      else:
        model_name_or_path = args.model_name_or_path
      architecture = model_name_or_path
      print(f"\nLoaded model: pretrained/{args.size}/{args.lang}\n")

    ### CPU Mode ####
    if args.cpu_mode:
        args.max_steps=10
        args.fp16=0
        args.warmup_steps=1
        args.eval_steps=2
        args.logging_steps=1

    ### Load degradations ###
    if (args.degradations_path is not None) and (not args.multiple_eval):
      with open(args.degradations_path) as json_file:
        list_degradations = json.load(json_file)
    else:
      list_degradations = None
    
    ### Instanciate Whisper Pipeline classes ###
    # Load feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    # Load tokenizer related to model
    tokenizer = WhisperTokenizer.from_pretrained(architecture, 
                                                language=lang_to_whisper[args.lang], 
                                                task=args.task)
    # Create processor, wrapper of feature_extractor and tokenizer
    processor = WhisperProcessor(feature_extractor, tokenizer)
    # Build Whisper architecture + load weights
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path) 
    model.config.use_cache = False # Turn off warning
    # Force task and language
    if args.fix_forced_decoder_ids:
      forced_decoder_ids = processor.get_decoder_prompt_ids(
                                      language=lang_to_whisper[args.lang], 
                                      task="transcribe")
      model.generation_config.forced_decoder_ids = forced_decoder_ids
    else: 
      model.config.forced_decoder_ids = None # To be predicted
    # Remove supressed tokens
    if not args.suppress_tokens:
        model.config.suppress_tokens = []

    ### Prepare metric ###
    metric = evaluate.load("wer")
    compute_metrics = Metrics(tokenizer=tokenizer, 
                              metric_wer=metric,
                              normalize=args.normalize)

    ### Use PEFT ###
    if args.use_peft:
      #from peft import prepare_model_for_int8_training
      #model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
      from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
      config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], 
                          lora_dropout=0.05, bias="none")
      model = get_peft_model(model, config)
      model.print_trainable_parameters()


    ### Original preprocessing pipeline (No data augmentation) ###
    """
    if (list_degradations is None) and (not args.multiple_eval):
      # Preprocess audio: resamples
      dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
      # Extract features: compute log-magnitude Mel Spectrogram
      prepare_dataset_func = partial(prepare_dataset, 
                                     tokenizer=tokenizer, 
                                     feature_extractor=feature_extractor,
                                     dataset=args.dataset)
      dataset = dataset.map(prepare_dataset_func,
                            num_proc=2)
      # Prep data_collator
      data_collator = DataCollator(processor=processor)
    """
    ### Preprocessing pipeline with data augmentation ###
    #else:
    # All audio processing done in the DataCollator
    data_collator = DataCollatorwithDegradation(processor,
                                                  args.dataset,
                                                  list_degradations)
      
    ### Finetune ###
    if args.train:
      # Perform training and evaluation
      # Refere to https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
      # Set Hyperparameters
      training_args = Seq2SeqTrainingArguments(
          output_dir=args.output_dir,
          # Fix seed 
          seed=42,
          data_seed=42,
          # General
          generation_max_length=225,
          # Train hyperparam
          num_train_epochs=args.num_train_epochs,
          per_device_train_batch_size=args.per_device_train_batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps, 
          learning_rate=args.learning_rate, # Initial learning rate
          warmup_steps=args.warmup_steps,
          weight_decay=args.weight_decay,
          logging_steps=args.logging_steps,
          # Computing details         
          gradient_checkpointing=args.gradient_checkpointing,
          fp16=bool(args.fp16),
          # Evaluation
          per_device_eval_batch_size=args.per_device_eval_batch_size,
          evaluation_strategy="epoch",
          predict_with_generate=True,
          # More
          remove_unused_columns= False, 
          label_names=["labels"], # Needed for PEFT
          # Save model
          save_strategy= "epoch",
          save_total_limit = 1,
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
      )

      # Perform Early stopping
      if args.patience != -1:
        early_stop = EarlyStoppingCallback(early_stopping_patience=args.patience,
                                       early_stopping_threshold=args.early_stopping_threshold)
        callbacks = [early_stop]
      else:
        callbacks = None
      # Set Trainer
      trainer = Seq2SeqTrainerwithAugmentation(
          args=training_args,
          model=model,
          train_dataset=dataset["train"],
          eval_dataset=dataset["val"],
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          callbacks=callbacks,
          tokenizer=processor.feature_extractor,
      )

      # Test dataloader
      if args.test_dataloader:
        # Save file
        trainer.data_collator.save = True
        train_dataloader = trainer.get_train_dataloader()
        val_dataloader = trainer.get_eval_dataloader(dataset["val"])

        for i in range(5):
          next(iter(train_dataloader))
          next(iter(val_dataloader))

        exit()

      # Start finetuning
      trainer.train()
      # Save best model
      trainer.save_model(args.output_dir)
      
      dict_logs = dataclasses.asdict(trainer.state)

      # No data augmentations for evaluation
      if list_degradations is not None:
        data_collator.list_degradations = None
      
      # Evaluate
      test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
      print("\n***** Evaluation Results *****")
      print(pd.Series(test_metrics))

      dict_logs["log_history"].append(test_metrics)

      # Save training log
      with open(os.path.join(args.output_dir,'training_logg.json'), 'w') as file:
          file.write(json.dumps(dict_logs, indent=4))
          print(f"\nLogging history saved at: {os.path.join(args.output_dir,'training_logg.json')}")

      # Save finetuned weights in output_dir
      trainer.save_model()

    ### Only perform evaluation ###
    else:
      # Set only evaluation related metrics
      training_args = Seq2SeqTrainingArguments(
          output_dir=args.output_dir,
          fp16=args.fp16,
          per_device_eval_batch_size=args.per_device_eval_batch_size,
          predict_with_generate=True,
          generation_max_length=225, 
          metric_for_best_model="wer",
          remove_unused_columns = False,
          push_to_hub=False,
      )
      # Get Trainer
      trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          eval_dataset=dataset["test"],
          data_collator=data_collator,
          compute_metrics=compute_metrics,
          tokenizer=processor.feature_extractor,
      )
      ### Evaluate robustness: loop over degradations ###
      if args.multiple_eval:
        evaluate_robustness(trainer=trainer, 
                            data_collator=data_collator, 
                            degradation_path=args.degradations_path,
                            output_dir=args.output_dir)

      else:
        ### Evaluate once on Test Dataset ###
        prediction_output = trainer.predict(test_dataset=dataset["test"],
                                            metric_key_prefix="test",
                                            max_length=225,
                                            num_beams=1) #Greedy search
        # Inverse tokenize predicted transcription
        transcriptions = processor.batch_decode(prediction_output.predictions, 
                                                skip_special_tokens=True)
        # Inverse tokenize target transcription
        labels = processor.batch_decode(prediction_output.label_ids,
                                        skip_special_tokens=True)
        metrics_dict = compute_metrics(pred=prediction_output, per_sample=True)
        # Save predictions and targets in dataframe
        df_predictions = pd.DataFrame(data = list(zip(labels, 
                                                      transcriptions,
                                                      metrics_dict["references"],
                                                      metrics_dict["predictions"],
                                                      metrics_dict["wer"])),
                                      columns = ["labels", 
                                                "transcribed",
                                                "labels_norm",
                                                "transcribed_norm",
                                                "wer"],
                                      index = dataset["test"]["path"]) # Returned in order
        df_predictions["path"] = df_predictions.index.map(os.path.basename)
        # Add metadata to results
        if args.merge_dataset:
          drop_columns = []
          for column in dataset["test"].column_names:
            for s in ["text", "transcri", "audio", "labels", "feature"]:
              if s in column:
                drop_columns.append(column)
          df_predictions = df_predictions.join(
                    dataset["test"].to_pandas().drop(
                      columns=drop_columns).set_index("path"))
        df_predictions.reset_index(drop=True, inplace=False)
        df_predictions.to_csv(args.output_dir+"/predictions.csv", index=False)

        # Compute WER
        metrics = prediction_output.metrics
        print("\n***** Evaluation Results *****")
        print(pd.Series(metrics))

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Basic arguments
    parser.add_argument(
        "--size", type=str, default="tiny",
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help="The size of the Whisper Architecture.",
    )
    parser.add_argument(
        "--finetuned", type=str2bool, default=False, 
        help="If True, load finetuned weights. Else, load pretrained models."
    )
    parser.add_argument(
        "--dataset", type=str, default="google/fleurs",
        choices=['google/fleurs', 'librispeech_asr'],
        help="Dataset name.",
    )
    parser.add_argument(
        "--lang", type=str, default="gl",
        choices=['en','fr','gl','fa','es'],
        help="Language code.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Paths to save results."
    )
    parser.add_argument(
        "--train", type=str2bool, default=False,
        help="True for finetuning and False for Evaluation.",
    )
    parser.add_argument(
        "--normalize", type=str, default="none",
        choices=["english_normalizer", "basic_normalizer", "lower", "none"],
        help="Normalizer to apply on text before evaluation.", 
    )
    # Evaluation
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=8,
        help="",
    )
    parser.add_argument(
        "--multiple_eval", type=str2bool, default=False,
        help=("If False, the dataset is evaluated once w/o degradation. "+
              "Else, multiple evaluations are performed with different combinations "+ 
              "of degradations applied to the given dataset. In this case, "+
              "--degradation_path is required."), 
    )
    parser.add_argument(
        "--degradations_path", type=str, default=None,
        help=("Path to a json file containing the degradations. "+
              "The format of this file differs between a single evaluation or "+ 
              "multiple ones. Please check the README for more."),
    )
    parser.add_argument(
        "--merge_dataset", type=str2bool, default=False,
        help= "If True, and single evaluation, add dataset information to predictions."
    )
    # Finetuning
    parser.add_argument(
        "--use_peft", type=str2bool, default=False,
        help="If True, finetune with LoRA."
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=32,
        help="Train batch size.", 
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Increase by 2x for every 2x decrease in batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5,
        help="Learning rate to specify for finetuning.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, help="", default=3
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500,
        help="Warmup steps for finetuning.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay for finetuning.",
    )
    parser.add_argument(
        "--patience", type=int, default=-1,
        help="Patience for Early Stopping."
    )
    parser.add_argument(
        "--early_stopping_threshold", type=float, default=1.0,
        help="Early sopping threshold.", 
    )
    parser.add_argument(
        "--gradient_checkpointing", type=str2bool, help="", default=False
    )
    parser.add_argument(
        "--eval_steps", type=int, help="", default=200
    )
    parser.add_argument(
        "--logging_steps", type=int, help="", default=20
    )
    # Advanced
    parser.add_argument(
        "--debug", type=str2bool, default=False,
        help="If True, the dataset will be reduced to 10 samples.", 
    )
    parser.add_argument(
        "--cpu_mode", type=str2bool, default=False,
        help="If True, run on CPU.",
    )
    parser.add_argument(
      "--model_name_or_path", type=str, default=None,
      help="Name of or path to model to load using PreTrainedModel from HuggingFace." 
    )    
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, default=None,
        help="Name of or path to the tokenizer to load using PreTrainedTokenizer from HuggingFace.",
    )
    parser.add_argument(
        "--task", type=str, default="transcribe",
        help="Task for fine-tuning."
    )
    parser.add_argument(
        "--fp16", type=str2bool, default=True,
        help="If True, applies mixed precision.", 
    )
    parser.add_argument(
        "--fix_forced_decoder_ids", type=str2bool, default=True,
        help=("If True, fixes the task and languages to the chosen ones."+
              "Else, they will be predicted."),
    )
    parser.add_argument(
        "--suppress_tokens", type=str2bool, help="", default=False
    )
    parser.add_argument(
        "--test_dataloader", type=str2bool, default=False,
        help="If True, outputs audio from the dataloader to test."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
