# -*- coding: utf-8 -*-
"""whisper_eval.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x5YTcG93Xjh3F3kzTtHxBs1bEsY2YymE
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import json
import os

import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (WhisperFeatureExtractor, WhisperTokenizer, 
                          WhisperProcessor, WhisperForConditionalGeneration)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
import evaluate

import audio_degrader as ad
import numpy as np

import logging

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="Initial model", default="openai/whisper-tiny"
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
        "--test_cpu_mode", type=bool, help="", default=False
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
        "--dataset_streaming", type=int, help="", default=0
    )
    parser.add_argument(
        "--train", type=int, help="0->eval, 1->train+eval", default=1
    )
    parser.add_argument(
        "--degradation_path", type=str, help="path to degradation json", 
        default=None,
    )
    parser.add_argument
    # TO DO - Help comments in the arguments
    args = parser.parse_args()
    return args

lang_to_whisper = {"gl":"Galician", "fr":"French", "fa":"Persian"}
lang_to_id = {"gl":"gl_es", "fr":"fr_fr", "fa":"fa_ir"}

def prepare_audio(samples):
    # Normalize
    if np.abs(samples).max() > 1:
      rms_samples = np.sqrt(np.sum(np.power(samples, 2)))
      samples = samples/rms_samples
    return samples

def apply_degradation(degradation: List[str], samples, 
                      sample_rate_in: int = 16e3):
  """
  Function to apply degradations on one audio.
  Inputs:
  - degradation: List of strings. String format name_degradation,param1,param2
  - samples: Numpy array of samples
  - sample_rate_in: Sample rate of the audio. If not equal to 16k Hz, will be resampled
  """
  # Prep audio
  audio = ad.AudioArray(samples_in = samples, 
                      sample_rate_in = sample_rate_in,
                      sample_rate_process= 16e3, # Default rate
                      bits = 64) # Default resolution
  # Loop over degradations and apply
  degradation = ad.ParametersParser.parse_degradations_args(degradation)
  for d in degradation:
    audio.apply_degradation(d)

  return audio.samples, int(audio.sample_rate)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    def __init__(self, processor: Any, 
                 tokenizer: Any,
                 feature_extractor: Any,
                 list_degradations: List[Dict[str,float]] = None):
      self.processor = processor
      self.tokenizer = tokenizer
      self.feature_extractor = feature_extractor
      self.list_degradations = list_degradations
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:

        input_features = []
        label_features = []

        for data in batch:
          samples  = data["audio"]["array"]
          sample_rate_in = data["audio"]["sampling_rate"]
          samples = prepare_audio(samples)

          if self.list_degradations is not None:
            for dict_degradation in self.list_degradations:
                degradation = dict_degradation["degradation"]
                prob = dict_degradation["prob"]
                assert 0 <= prob <= 1
                # For random data augmentation
                sample = np.random.uniform()
                if sample > prob:
                  # Resample
                  samples, sample_rate = apply_degradation(["resample,16000"], 
                                                          samples, sample_rate_in)
                else:
                  # Apply degradation with probability = prob
                  samples, sample_rate = apply_degradation(degradation, samples, 
                                                    sample_rate_in)
          else:
            # Resample
            samples, sample_rate = apply_degradation(["resample,16000"], 
                                                     samples, sample_rate_in)

          assert sample_rate == 16e3
          # Test Audio for small batch size
          #ipd.display(ipd.Audio(samples, rate = sample_rate))
          
          # Compute log-Mel input features from input audio array 
          input_features.append({"input_features": 
                                 self.feature_extractor(samples, 
                                 sampling_rate=sample_rate).input_features[0]})
          # Tokenize transcription
          label = self.tokenizer(data["raw_transcription"]).input_ids
          label_features.append({"input_ids": label})

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred, tokenizer, metric_wer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main():

    args = arg_parse()

    # Load datasets
    dataset = DatasetDict()

    train = bool(args.train)
    if train:
      dataset["train"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                      split="train+validation", 
                                      streaming=bool(args.dataset_streaming))
    dataset["test"] = load_dataset(args.dataset, lang_to_id[args.lang], 
                                   split="test", 
                                   streaming=bool(args.dataset_streaming))
    print(dataset)
    # Load degradations
    if args.degradation_path is not None:
      with open(args.degradation_path) as json_file:
        list_degradations = json.load(json_file)
    else:
      list_degradations = None
    # Debug settings
    if args.test_cpu_mode:
        dataset["train"] = dataset["train"].select(list(range(0, 10)))
        dataset["test"] = dataset["test"].select(list(range(0, 10)))
        print(dataset)
        args.max_steps=10
        args.fp16=0
        args.warmup_steps=1
        args.eval_steps=2
        args.logging_steps=1
    # 
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, 
                            language=lang_to_whisper[args.lang], task=args.task)
    #
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, 
                            language=lang_to_whisper[args.lang], task=args.task)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor,
                      tokenizer=tokenizer, feature_extractor=feature_extractor,
                      list_degradations=list_degradations)

    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer, 
                                   metric_wer=metric)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if train:
      # Perform training and evaluation
      training_args = Seq2SeqTrainingArguments(
          output_dir= args.output_dir,# "./whisper-small-gl",  # change to a repo name of your choice
          per_device_train_batch_size=args.per_device_train_batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
          learning_rate=args.learning_rate,
          warmup_steps=args.warmup_steps,#500,
          max_steps=args.max_steps, #1000, #4000,
          gradient_checkpointing=bool(args.gradient_checkpointing),
          fp16=bool(args.fp16),
          evaluation_strategy="steps",
          per_device_eval_batch_size=args.per_device_eval_batch_size,
          predict_with_generate=True,
          generation_max_length=225,
          save_steps=args.eval_steps, #100, #1000
          eval_steps=args.eval_steps,#100, #1000
          logging_steps=args.logging_steps,#25,
          #report_to=["tensorboard"],
          load_best_model_at_end=True,
          metric_for_best_model="wer",
          greater_is_better=False,
          push_to_hub=False,
          save_total_limit=2,
          remove_unused_columns = False,
      )

      early_stop = EarlyStoppingCallback(3, 1.0)

      trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          train_dataset=dataset["train"],
          eval_dataset=dataset["test"],
          data_collator=data_collator,
          compute_metrics=compute_metrics_func,
          callbacks=[early_stop],
          tokenizer=processor.feature_extractor,
      )

      trainer.train()

      print("History")
      logic_steps=trainer.state.log_history
      print("End History")
      with open(os.path.join(args.output_dir,'training_logg.json'), 'w') as file:
          file.write(json.dumps(logic_steps, indent=4))
          print(f"Logging history saved at: {os.path.join(args.output_dir,'training_logg.json')}")

      metrics = trainer.evaluate()
      print(metrics)

      trainer.save_model()

    else:
      # Only perform evaluation
      training_args = Seq2SeqTrainingArguments(
          output_dir= "./whisper-small-gl",
          fp16=True,
          per_device_eval_batch_size=64,
          predict_with_generate=True,
          generation_max_length=225, #?
          metric_for_best_model="wer",
          remove_unused_columns = False,
          push_to_hub=False,
      )
      trainer = Seq2SeqTrainer(
          args=training_args,
          model=model,
          eval_dataset=dataset["test"],
          data_collator=data_collator,
          compute_metrics=compute_metrics_func,
          tokenizer=processor.feature_extractor,
      )

      metrics = trainer.evaluate()
      print(metrics)


if __name__ == '__main__':
    main()