import argparse

from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    args = parser.parse_args()
    return args


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

    datacollator = DataCollatorAttacker(processor, model, device)

    dataset_adversarial = Dataset.from_generator(
        generator=lambda: datacollator(dataset))


if __name__ == "__main__":
    args = arg_parse()
    main(args)