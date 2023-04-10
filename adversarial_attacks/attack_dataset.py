import argparse

from datasets import load_dataset
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
        "--model",
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


def adversarial_attack(
    model,
    batch,
    attack_iterations,
    epsilon: int = 0.2,
):
    # TODO: test this process
    # TODO: modify the devices
    batch["audio"].requires_grad = True
    predicted_ids = model.generate(batch["input_features"])
    for _ in attack_iterations:
        out = model.forward(input_features=batch["input_features"],
                            labels=predicted_ids)
        loss = F.cross_entropy(out.logits.view(-1, model.config.vocab_size),
                               predicted_ids.view(-1))
        # TODO: WHAT IS PREDICTED_IDS
        model.zero_grad()
        loss.backward()
        audio_grad = batch["audio"].grad.data
        sign_data_grad = audio_grad.sign()
        perturbed_sound = batch["audio"] - epsilon * sign_data_grad
    return perturbed_sound


def main(args):
    # Create dataset
    dataset = load_dataset(args.dataset, "clean", split="validation")

    # Feature Extractor
    processor = WhisperProcessor.from_pretrained(args.model_name)

    datacollator = DataCollatorAttacker(processor=processor, dataset=dataset)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        collate_fn=datacollator)

    # Initialize the Model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Perform attack
    for batch in loader:
        perturbed_sound = adversarial_attack(model, batch,
                                             args.attack_iterations,
                                             args.epsilon)


if __name__ == "__main__":
    args = arg_parse()
    main(args)