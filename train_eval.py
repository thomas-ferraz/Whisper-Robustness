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

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial

import numpy as np
import pandas as pd

import torch

from datasets import load_dataset, DatasetDict, Audio
import evaluate
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, 
                          WhisperProcessor, WhisperForConditionalGeneration)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
from peft import prepare_model_for_int8_training
from peft import PeftModel, LoraModel, LoraConfig, get_peft_model

import audio_degrader as ad

from data_utils import (prepare_audio, apply_degradation,
                                prepare_dataset,
                                DataCollator,
                                DataCollatorwithDegradation,
                                evaluate_robustness)



def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size", type=str, help="Model size", default="tiny"
    )
    parser.add_argument(
        "--finetuned", type=int, help="0 for Pretrained, 1 for Fintuned", default=0
    )    
    parser.add_argument(
        "--tokenizer_name", type=str, help="Initial tokenizer", default=None
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name", default="google/fleurs"
    )
    parser.add_argument(
        "--lang", type=str, help="Language code", default="gl"
    )
    parser.add_argument(
        "--task", type=str, help="Task for fine-tuning", default="transcribe"
    ) #maybe do a boolean

    parser.add_argument(
        "--output_dir", type=str, help="", default="./model"
    )
    parser.add_argument(
        "--cpu_mode", type=bool, help="", default=False
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, help="", default=32
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="Increase by 2x for every 2x decrease in batch size", default=1
    )
    parser.add_argument(
        "--learning_rate", type=float, help="", default=1e-5
    )
    parser.add_argument(
        "--warmup_steps", type=int, help="", default=500
    )
    parser.add_argument(
        "--weight_decay", type=float, help="", default=0.0
    )
    parser.add_argument(
        "--max_steps", type=int, help="", default=4000
    )
    parser.add_argument(
        "--gradient_checkpointing", type=int, help="", default=1
    )
    parser.add_argument(
        "--fp16", type=int, help="", default=1
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, help="", default=8
    )
    parser.add_argument(
        "--eval_steps", type=int, help="", default=200
    )
    parser.add_argument(
        "--logging_steps", type=int, help="", default=20
    )
    parser.add_argument(
        "--fix_forced_decoder_ids", type=int, help="", default=0
    )
    parser.add_argument(
        "--suppress_tokens", type=int, help="", default=0
    )
    parser.add_argument(
        "--train", type=int, help="0 for eval, 1 for train+eval", default=1
    )
    parser.add_argument(
        "--patience", type=int, help="", default=3
    )
    parser.add_argument(
        "--use_peft", type=int, help="", default=0
    )
    parser.add_argument(
        "--early_stopping_threshold", type=float, help="", default=1.0
    )
    parser.add_argument(
        "--eval_robustness", type=int, help="", default=0,
    )
    parser.add_argument(
        "--degradations_path", type=str, help="path to degradation json", 
        default=None,
    )
    parser.add_argument(
        "--debug", type=int, help="", default=0
    )
    parser.add_argument(
        "--predict", type=str, help="", default=0,
    )
    parser.add_argument(
        "--normalize", type=str, help="normalized wer", default="none",
    )
    # TO DO - Help comments in the arguments
    args = parser.parse_args()
    return args

lang_to_whisper = {"gl":"Galician", 
                   "fr":"French", 
                   "fa":"Persian", 
                   "libri_en": "English"}
lang_to_id = {"gl":"gl_es", 
              "fr":"fr_fr", 
              "fa":"fa_ir", 
              "libri_en":"clean"}

def load_finetuned(size="tiny", language="French"):
  """Download finetuned weights from drive"""
  if not os.path.isdir("./finetuned"):
    os.mkdir("./finetuned")
  # Load dictionary of ids
  with open("./finetuned_models.json") as file:
    dict_finetuned = json.load(file)  
  # Download 
  for f, file_id in dict_finetuned[size][language].items():
    gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t",
                 output=f"./finetuned/{f}",
                 use_cookies=False)


def compute_metrics(pred, tokenizer, metric_wer, normalize = "none"):

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if normalize == "whisper":
      pred_str = [tokenizer._normalize(text) for text in pred_str]
      label_str = [tokenizer._normalize(text) for text in label_str]
    elif normalize == "lower":
      pred_str = [text.lower() for text in pred_str]
      label_str = [text.lower() for text in label_str]     

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


def main():

    args = arg_parse()
    ## Verif params
    assert args.size in ["tiny", "base", "small", "medium", "larges"], (
      "Whisper sizes are 'tiny', 'base', 'small', 'medium' and 'larges'.")
    assert args.lang in ["gl","fr","fa","en","libri_en"], ( 
      "Supported languages are: 'gl', 'fr', 'fa', 'en' and 'libri_en'.")
    assert args.normalize in ["none", "whisper", "lower"], (
          "Normalize can be none, whisper or lower.")

    
    ### Load datasets ###
    dataset = DatasetDict()
    # Load train and validation splits for finetuning
    train = bool(args.train)
    if train:
      dataset["train"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                      split="train")
      dataset["val"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                      split="validation")
    # Load test set
    dataset["test"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                   split="test")

    ### CPU Test Mode ####
    if args.cpu_mode:
        args.max_steps = 10
        args.fp16 = 0
        args.warmup_steps = 1
        args.eval_steps = 2
        args.logging_steps = 1
        args.debug = 1

    ### Debug settings ###
    if bool(args.debug):
      print("\nDebug Mode.")
      for s, d in dataset.items():
        dataset[s] = dataset[s].select(list(range(0, 10)))
    
    print(dataset)

    ### Load pretrained/finetuned ###
    # Load finetuned weights and configurations from drive
    if bool(args.finetuned):
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
    # Load pretrained weights from Hugging Face
    else:
      # Set path of loaded model
      model_name_or_path = "openai/whisper-"+args.size
      architecture = model_name_or_path
      print(f"\nLoaded model: pretrained/{args.size}/{args.lang}\n")

    ### Load degradations ###
    eval_robustness = bool(args.eval_robustness)
    if (args.degradations_path is not None) and (not eval_robustness):
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
    # Force task and language
    if bool(args.fix_forced_decoder_ids):
      forced_decoder_ids = processor.get_decoder_prompt_ids(
                                      language=lang_to_whisper[args.lang], 
                                      task="transcribe")
      model.generation_config.forced_decoder_ids = forced_decoder_ids
    else: 
      model.config.forced_decoder_ids = None # To be predicted
    # Remove supressed tokens
    if not bool(args.suppress_tokens):
        model.config.suppress_tokens = []

    ### Prepare metric ###
    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer, 
                                                    metric_wer=metric,
                                                    normalize=args.normalize)

    ### Use PEFT ###
    if bool(args.use_peft):
      model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")

      config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
      model = get_peft_model(model, config)
      model.print_trainable_parameters()


    ### Original preprocessing pipeline (No data augmentation) ###
    if (list_degradations is None) and (not eval_robustness):
      # Preprocess audio: resamples
      dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
      # Extract features: compute log-magnitude Mel Spectrogram
      prepare_dataset_func = partial(prepare_dataset, 
                                     tokenizer=tokenizer, 
                                     feature_extractor=feature_extractor,
                                     dataset=args.dataset)
      dataset = dataset.map(prepare_dataset_func, 
                            remove_columns=dataset.column_names["test"], 
                            num_proc=2)
      # Prep data_collator
      data_collator = DataCollator(processor=processor)
    
    ### Preprocessing pipeline with data augmentation ###
    else:
      # All audio processing done in the DataCollator
      data_collator = DataCollatorwithDegradation(processor,
                                                  args.dataset,
                                                  list_degradations)
    ### Finetune ###
    if train:
      # Perform training and evaluation
      # Refere to https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
      training_args = Seq2SeqTrainingArguments(
          output_dir=args.output_dir,
          per_device_train_batch_size=args.per_device_train_batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps, 
          learning_rate=args.learning_rate,
          warmup_steps=args.warmup_steps,
          max_steps=args.max_steps, 
          gradient_checkpointing=bool(args.gradient_checkpointing),
          fp16=bool(args.fp16),
          evaluation_strategy="steps",
          per_device_eval_batch_size=args.per_device_eval_batch_size,
          predict_with_generate=True,
          generation_max_length=225,
          save_steps=args.eval_steps, 
          eval_steps=args.eval_steps,
          logging_steps=args.logging_steps,
          #report_to=["tensorboard"],
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
          push_to_hub=False,
          save_total_limit=2,
          weight_decay=args.weight_decay,
          remove_unused_columns= False, 
          label_names=["labels"] if bool(args.use_peft) else None, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
      )

      # Perform Early stopping
      early_stop = EarlyStoppingCallback(early_stopping_patience=args.patience,
                                       early_stopping_threshold=args.early_stopping_threshold)
      # Set Trainer
      trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          train_dataset=dataset["train"],
          eval_dataset=dataset["val"],
          data_collator=data_collator,
          compute_metrics=compute_metrics_func,
          callbacks=[early_stop],
          tokenizer=processor.feature_extractor,
      )
      # Start finetuning
      trainer.train()

      print("\nHistory")
      log_steps = trainer.state.log_history.copy()
      print("End History\n")
      
      # No data augmentations for evaluation
      if list_degradations is not None:
        data_collator.list_degradations = None
      
      # Evaluate
      test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
      print("\n***** Evaluation Results *****")
      print(pd.Series(test_metrics))

      log_steps.append(test_metrics)

      # Save training log
      with open(os.path.join(args.output_dir,'training_logg.json'), 'w') as file:
          file.write(json.dumps(log_steps, indent=4))
          print(f"\nLogging history saved at: {os.path.join(args.output_dir,'training_logg.json')}")

      # Save finetuned weights in output_dir
      trainer.save_model()

    ### Only perform evaluation ###
    else:
      # Set only evaluation related metrics
      training_args = Seq2SeqTrainingArguments(
          output_dir= args.output_dir,
          fp16=bool(args.fp16),
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
          compute_metrics=compute_metrics_func,
          tokenizer=processor.feature_extractor,
      )
      ### Evaluate robustness: loop over degradations ###
      if eval_robustness:
        evaluate_robustness(trainer=trainer, 
                            data_collator=data_collator, 
                            degradation_path=args.degradations_path,
                            output_dir=args.output_dir)

      else:
        ### Evaluate on a single degradation ###
        prediction_output = trainer.predict(dataset["test"],
                                      metric_key_prefix="test")
        generated_ids = prediction_output.predictions
        # Inverse tokenize predicted transcription
        transcriptions = processor.batch_decode(generated_ids, 
                                                      skip_special_tokens=True)
        # Inverse tokenize target transcription
        labels = processor.batch_decode(prediction_output.label_ids, 
                                                    skip_special_tokens=True)
        # Save predictions and targets in dataframe
        df_predictions = pd.DataFrame()
        df_predictions["labels"] = labels
        df_predictions["transcribed"] = transcriptions
        # Save normalized text
        if args.normalize in ["whisper", "lower"]:
          df_predictions["labels_norm"] = [tokenizer._normalize(text) 
                                            for text in labels]
          df_predictions["transcribed_norm"] = [tokenizer._normalize(text) 
                                            for text in transcriptions]
        
        df_predictions.to_csv(args.output_dir+"/predictions.csv")

        # Compute WER
        metrics = prediction_output.metrics
        print("\n***** Evaluation Results *****")
        print(pd.Series(metrics))


if __name__ == '__main__':
    main()
