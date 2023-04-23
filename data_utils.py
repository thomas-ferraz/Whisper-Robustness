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
import json
import time
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import sox
import audio_degrader as ad

import logging
from adversarial_attacks.whisper_attacker_feature_extractor import WhisperAttackerFeatureExtractor

dataset_text_name = {
    "google/fleurs": "raw_transcription",
    "librispeech_asr": "text"
}


def prepare_audio(samples):
    # Normalize
    if np.abs(samples).max() > 1:
        rms_samples = np.sqrt(np.sum(np.power(samples, 2)))
        samples = samples / rms_samples
    return samples


def prepare_dataset(batch, feature_extractor, tokenizer, dataset):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    text_name = dataset_text_name[dataset]
    batch["labels"] = tokenizer(batch[text_name]).input_ids
    return batch


def apply_degradation(degradation: List[str],
                      samples,
                      sample_rate_in: int = 16e3,
                      save_file=None,
                      verbose=0):
    """
  Function to apply degradations on one audio samples.
  Inputs:
  - degradation: List of strings. String format name_degradation,param1,param2.
  - samples: Numpy array of samples.
  - sample_rate_in: Sample rate of the audio. 
                    If not equal to 16k Hz, will be resampled.
  """
    st = time.time()
    # Prep audio
    audio = ad.AudioArray(
        samples_in=samples,
        sample_rate_in=sample_rate_in,
        sample_rate_process=16e3,  # Default rate
        bits=64)  # float16 for Whisper
    # Loop over degradations and apply
    degradation = ad.ParametersParser.parse_degradations_args(degradation)
    for d in degradation:
        audio.apply_degradation(d)
    et = time.time()

    if save_file is not None:
        tfm = sox.Transformer()
        tfm.set_output_format(rate=audio.sample_rate, bits=64, channels=1)
        tfm.build_file(input_array=audio.samples,
                       sample_rate_in=audio.sample_rate,
                       output_filepath=save_file)
    if verbose > 0:
        print(f"\nApplied degradations in {et-st:.3f} seconds.")
    return audio.samples, int(audio.sample_rate)


@dataclass
class DataCollator:
    """Normal collator"""

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{
            "input_features": feature["input_features"]
        } for feature in features]
        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id
            ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorwithDegradation:

    def __init__(self,
                 processor: Any,
                 dataset: str,
                 list_degradations: List[Dict[str, float]] = None):
        self.processor = processor
        self.list_degradations = list_degradations
        self.dataset = dataset

    def __call__(self, batch) -> Dict[str, torch.Tensor]:

        input_features = []
        label_features = []

        for data in batch:
            samples = data["audio"]["array"]
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
                        print(f"ouput_{data.id}.wav")
                        samples, sample_rate = apply_degradation(
                            ["resample,16000"], samples, sample_rate_in)
                    else:
                        # Apply degradation with probability = prob
                        samples, sample_rate = apply_degradation(
                            degradation, samples, sample_rate_in)
            else:
                # Resample
                samples, sample_rate = apply_degradation(["resample,16000"],
                                                         samples,
                                                         sample_rate_in)

            assert sample_rate == 16e3
            # Test Audio for small batch size
            #ipd.display(ipd.Audio(samples, rate = sample_rate))

            # Compute log-Mel input features from input audio array
            input_features.append({
                "input_fraeatures":
                self.processor.feature_extractor(
                    samples, sampling_rate=sample_rate).input_features[0]
            })
            # Tokenize transcription
            text_name = dataset_text_name[self.dataset]
            label = self.processor.tokenizer(data[text_name]).input_ids
            label_features.append({"input_ids": label})

        batch = self.processor.feature_extractor.pad(input_features,
                                                     return_tensors="pt")
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features,
                                                    return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id
            ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorAttacker:

    def __init__(self, processor: Any, dataset: Any):
        #TODO: see if this can be used
        processor.feature_extractor = WhisperAttackerFeatureExtractor()
        self.processor = processor

    def __call__(self, batch):
        input_features = []
        for data in batch:
            samples = torch.from_numpy(data["audio"]["array"]).float()
            samples_rate_in = data["audio"]["sampling_rate"]
            # TODO: add the labels
            # TODO: should we do the normalization here?
            # samples = prepare_audio(samples)
            sample_features = self.processor(
                samples, sampling_rate=samples_rate_in,
                return_tensors="pt").input_features
            sample_features = F.pad(
                input=sample_features[None],
                # TODO: change to parameters
                pad=(0, 3000 - 585),
                mode="constant",
                value=0.0)

            sample["input_features"].append(sample_features)
            sample["audio"].append(samples)
        sample["input_features"] = torch.tensor(sample["input_features"])
        sample["audio"] = torch.tensor(sample["audio"])
        return sample


def evaluate_robustness(trainer,
                        data_collator,
                        degradation_path,
                        output_dir=""):
    # Load degradations
    with open(degradation_path) as json_file:
        list_degradations = json.load(json_file)
    # Create dict for results
    df_results_all = []
    list_deg_str = []
    for dict_degradation in list_degradations:
        name_deg = dict_degradation["name"]
        dict_result_deg = {}

        if "param1" in dict_degradation.keys():
            list_param1 = dict_degradation["param1"]["values"]
            name_param1 = dict_degradation["param1"]["name"]

            for param1 in list_param1:
                if "param2" in dict_degradation.keys():
                    list_param2 = dict_degradation["param2"]["values"]
                    name_param2 = dict_degradation["param2"]["name"]

                    for param2 in list_param2:
                        deg_str = f"{name_deg},{param1},{param2}"
                        list_deg_str.append(deg_str)

                else:
                    deg_str = f"{name_deg},{param1}"
                    list_deg_str.append(deg_str)
        else:
            deg_str = f"{name_deg}"
            list_deg_str.append(deg_str)

    for deg_str in list_deg_str:

        print(f"\nEvaluating with: {deg_str}")
        dict_apply_deg = {"degradation": [deg_str], "prob": 1}
        data_collator.list_degradations = [dict_apply_deg]
        test_metrics = trainer.evaluate()

        # Save results
        split_str = deg_str.split(",")
        if not split_str[0] in dict_result_deg.keys():
            dict_result_deg[split_str[0]] = []

        dict_result = {}
        for i, p in enumerate(split_str[1:]):
            dict_result["param" + str(i + 1)] = p
        for k, v in test_metrics.items():
            dict_result[k.replace("eval_", "")] = v

        dict_result_deg[split_str[0]].append(dict_result)

        print(pd.Series(dict_result, name=split_str[0]))
        # Serializing json
        json_object = json.dumps(dict_result_deg, indent=4)
        with open(output_dir + "/results_evaluate_robustness.json",
                  "w") as outfile:
            outfile.write(json_object)
