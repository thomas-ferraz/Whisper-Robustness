# Copyright (c) 2023 Thomas Palmeira Ferraz, Hélène Maxcici, Teysir Baoueb
#
# Licensed under the MIT License (the "License");
# You may not use this file except in compliance with the License.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from functools import partial
import json

from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

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
        "--output_dir", type=str, help="", default="model"
    )
    # List of args
    # model
    # language
    # dataset
    args = parser.parse_args()
    return args

lang_to_whisper = {"gl":"Galician"}
lang_to_id = {"gl":"gl_es"}

def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["raw_transcription"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
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

    #logging.basicConfig(filename="std.log",
                        #format='%(asctime)s %(message)s',
                        #filemode='w')

    #logger = logging.getLogger()

    dataset = DatasetDict()

    dataset["train"] = load_dataset(args.dataset, lang_to_id[args.lang], split="train+validation")
    dataset["test"] = load_dataset(args.dataset, lang_to_id[args.lang], split="test")

    print(dataset)

    #dataset["train"] = dataset["train"].select(list(range(0, 10)))
    #dataset["test"] = dataset["test"].select(list(range(0, 10)))
    #print(dataset)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)


    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=lang_to_whisper[args.lang], task=args.task)



    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=lang_to_whisper[args.lang], task=args.task)



    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    prepare_dataset_func = partial(prepare_dataset, tokenizer=tokenizer, feature_extractor=feature_extractor)
    dataset = dataset.map(prepare_dataset_func, remove_columns=dataset.column_names["train"], num_proc=2)





    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer, metric_wer=metric)




    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []



    training_args = Seq2SeqTrainingArguments(
        output_dir= args.output_dir,# "./whisper-small-gl",  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=1,#500,
        max_steps=10, #1000, #4000,
        gradient_checkpointing=True,
        #fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100, #1000
        eval_steps=2,#100, #1000
        logging_steps=1,#25,
        #report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    print("History")
    logic_steps= trainer.state.log_history
    print("End History")
    with open('model/training_logg.json', 'w') as file:
        file.write(json.dumps(logic_steps, indent=4))

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == '__main__':
    main()