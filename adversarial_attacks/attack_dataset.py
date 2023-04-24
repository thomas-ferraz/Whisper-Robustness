import argparse
from functools import partial
import pandas as pd

from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

from adversarial_attacks.whisper_attacker_feature_extractor import WhisperAttackerFeatureExtractor
from data_utils import DataCollatorAttacker
from data_utils import DataCollator


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Adversarial Attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        default="hf-internal-testing/librispeech_asr_dummy",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch Sizer",
        default=4,
    )
    parser.add_argument(
        "--attack_iterations",
        type=int,
        help="Number of attack iterations each sample",
        default=1,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
        default="openai/whisper-tiny.en",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Epsilon in attack",
        default=0.2,
    )
    parser.add_argument("--output_dir", type=str, help="", default="./model")
    parser.add_argument("--per_device_eval_batch_size",
                        type=int,
                        help="",
                        default=8)
    args = parser.parse_args()
    return args


def compute_metrics(pred, tokenizer, metric_wer, normalize="whisper"):
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

    return wer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = load_dataset(args.dataset, "clean", split="validation")

    # load model and processor
    # TODO: eliminate pred_processor when reading transcripts
    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.feature_extractor = WhisperAttackerFeatureExtractor()
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    data_collator_attacker = DataCollatorAttacker(processor=processor,
                                                  model=model,
                                                  epsilon=0.02,
                                                  device=device)

    dataset_adversarial = Dataset.from_generator(
        generator=lambda: data_collator_attacker(dataset))

    # Prepare dataset

    prepare_dataset_func = partial(
        prepare_dataset,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dataset=args.dataset)
    tokenized_dataset = dataset_adversarial.map(prepare_dataset_func,
                                                num_proc=2)

    # Data collator
    data_collator = DataCollator(processor=processor)

    # Compute metrics
    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics,
                                   tokenizer=processor.tokenizer,
                                   metric_wer=metric,
                                   normalize="whisper")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        fp16=False,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        metric_for_best_model="wer",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        tokenizer=processor.feature_extractor,
    )

    prediction_output = trainer.predict(tokenized_dataset,
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
    df_predictions["labels_norm"] = [
        processor.tokenizer._normalize(text) for text in labels
    ]
    df_predictions["transcribed_norm"] = [
        processor.tokenizer._normalize(text) for text in transcriptions
    ]
    df_predictions.to_csv(args.output_dir + "/predictions.csv")

    metrics = prediction_output.metrics
    print("\n***** Evaluation Results *****")
    print(pd.Series(metrics))

    # Prepare dataset

    prepare_dataset_func = partial(
        prepare_dataset,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dataset=args.dataset)
    tokenized_dataset = dataset_adversarial.map(prepare_dataset_func,
                                                num_proc=2)

    # Data collator
    data_collator = DataCollator(processor=processor)

    # Compute metrics
    metric = evaluate.load("wer")
    compute_metrics_func = partial(compute_metrics,
                                   tokenizer=processor.tokenizer,
                                   metric_wer=metric,
                                   normalize="whisper")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        fp16=False,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        metric_for_best_model="wer",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
        tokenizer=processor.feature_extractor,
    )

    prediction_output = trainer.predict(tokenized_dataset,
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
    df_predictions["labels_norm"] = [
        processor.tokenizer._normalize(text) for text in labels
    ]
    df_predictions["transcribed_norm"] = [
        processor.tokenizer._normalize(text) for text in transcriptions
    ]
    df_predictions.to_csv(args.output_dir + "/predictions.csv")

    metrics = prediction_output.metrics
    print("\n***** Evaluation Results *****")
    print(pd.Series(metrics))


if __name__ == "__main__":
    args = arg_parse()
    main(args)