import argparse

from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from adversarial_attacks.whisper_attacker_feature_extractor import WhisperAttackerFeatureExtractor
#from data_utils import DataCollatorAttacker


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
    pred_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.feature_extractor = WhisperAttackerFeatureExtractor()
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en")
    model.to(device)

    def data_collator_attacker(dataset):
        # TODO: testing as a datacollator object
        for data in dataset:
            samples = torch.from_numpy(data["audio"]["array"]).float()
            samples_rate_in = data["audio"]["sampling_rate"]

            # creating labels
            # TODO: fix column mapping
            labels_ids = processor.tokenizer(data["text"])["input_ids"]

            # adversarial attack
            # TODO: should we put this into a function?
            samples.requires_grad = True
            input_features = processor(
                samples, sampling_rate=samples_rate_in,
                return_tensors="pt").input_features.to(device)
            input_features = F.pad(input=input_features[None],
                                   pad=(0, 3000 - input_features.shape[1]),
                                   mode="constant",
                                   value=0.0)

            out = model.forward(input_features=input_features,
                                labels=labels_ids)
            loss = F.cross_entropy(
                out.logits.view(-1, model.config.vocab_size),
                labels_ids.view(-1))

            model.zero_grad()
            loss.backward()
            data_grad = samples.grad.data
            epsilon = 0.02
            sign_data_grad = data_grad.sign()
            perturbed_sound = samples - epsilon * sign_data_grad
            yield {
                "input_features": input_features,
                "audio": perturbed_sound.cpu()
            }

    dataset_adversarial = Dataset.from_generator(
        generator=lambda: data_collator_attacker(dataset))


if __name__ == "__main__":
    args = arg_parse()
    main(args)