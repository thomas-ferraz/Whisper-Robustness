import argparse

from datasets import load_dataset, Dataset
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from adversarial_attacks.whisper_attacker_feature_extractor import WhisperAttackerFeatureExtractor
from data_utils import DataCollatorAttacker


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
    parser.add_argument("--snr", type=int, help="Fixed SNR", default=35)
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
    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.feature_extractor = WhisperAttackerFeatureExtractor()
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    data_collator_attacker = DataCollatorAttacker(processor=processor,
                                                  model=model,
                                                  snr=args.snr,
                                                  device=device)

    adversarial_dataset = Dataset.from_generator(
        generator=lambda: data_collator_attacker(dataset))

    save_to_dir = f"attaked_{args.snr}.hf"
    adversarial_dataset.save_to_disk(save_to_dir)


if __name__ == "__main__":
    args = arg_parse()
    main(args)