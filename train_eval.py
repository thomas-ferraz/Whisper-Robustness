
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
import os

from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import EarlyStoppingCallback
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
        "--weight_decay", type=float, help="", default=0.0
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
        "--fix_forced_decoder_ids", type=int, help="", default=0
    )
    parser.add_argument(
        "--suppress_tokens", type=int, help="", default=0
    )
    parser.add_argument(
        "--train", type=int, help="", default=1
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
        "--save_model", type=int, help="", default=1
    )
    # TO DO - Help comments in the arguments
    args = parser.parse_args()
    return args

lang_to_whisper = {"gl":"Galician", "fr":"French", "fa":"Persian"}
lang_to_id = {"gl":"gl_es", "fr":"fr_fr", "fa":"fa_ir"}

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


    dataset["train"] = load_dataset(args.dataset, lang_to_id[args.lang],
                                    split="train",
                                    streaming=bool(args.dataset_streaming))
    dataset["val"] = load_dataset(args.dataset, lang_to_id[args.lang],
                                  split="validation",
                                    streaming=bool(args.dataset_streaming))
    dataset["test"] = load_dataset(args.dataset, lang_to_id[args.lang],
                                   split="test",
                                   streaming=bool(args.dataset_streaming))

    print(dataset)

    if args.test_cpu_mode:
        dataset["train"] = dataset["train"].select(list(range(0, 10)))
        dataset["val"] = dataset["val"].select(list(range(0, 10)))
        dataset["test"] = dataset["test"].select(list(range(0, 10)))
        print(dataset)
        args.max_steps=10
        args.fp16=0
        args.warmup_steps=1
        args.eval_steps=2
        args.logging_steps=1

    if not args.tokenizer_name:
        args.tokenizer_name = args.model_name_or_path

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_name)


    tokenizer = WhisperTokenizer.from_pretrained(args.tokenizer_name, language=lang_to_whisper[args.lang], task=args.task)



    processor = WhisperProcessor.from_pretrained(args.tokenizer_name, language=lang_to_whisper[args.lang], task=args.task)



    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    prepare_dataset_func = partial(prepare_dataset, tokenizer=tokenizer, feature_extractor=feature_extractor)
    dataset = dataset.map(prepare_dataset_func, remove_columns=dataset.column_names["train"], num_proc=2)





    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)



    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics, tokenizer=tokenizer, metric_wer=metric)




    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    if not bool(args.fix_forced_decoder_ids):
        model.config.forced_decoder_ids = None
    if not bool(args.suppress_tokens):
        model.config.suppress_tokens = []

    if bool(args.use_peft):
        from peft import prepare_model_for_int8_training
        model = prepare_model_for_int8_training(model, output_embedding_layer_name="proj_out")
        from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
        config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
        model = get_peft_model(model, config)
        model.print_trainable_parameters()



    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,# "./whisper-small-gl",  # change to a repo name of your choice
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
        weight_decay=args.weight_decay,
        remove_unused_columns=False if bool(args.use_peft) else True, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"] if bool(args.use_peft) else None, # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    )

    early_stop = EarlyStoppingCallback(early_stopping_patience=args.patience,
                                       early_stopping_threshold=args.early_stopping_threshold)

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

    if bool(args.train):
        trainer.train()

    print("History")
    log_steps = trainer.state.log_history.copy()
    print("End History")

    test_metrics = trainer.evaluate(dataset["test"])
    #print(test_metrics)
    test_metrics = {k.replace("eval","test"):v for k,v in test_metrics.items()}
    print(test_metrics)
    log_steps.append(test_metrics)

    with open(os.path.join(args.output_dir,'training_logg.json'), 'w') as file:
        file.write(json.dumps(log_steps, indent=4))
        print(f"Logging history saved at: {os.path.join(args.output_dir,'training_logg.json')}")

    if bool(args.train) and bool(args.save_model):
        trainer.save_model()


if __name__ == '__main__':
    main()
