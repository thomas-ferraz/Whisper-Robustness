from typing import List, Dict
from functools import partial
import argparse
import os
from omegaconf import OmegaConf

from evaluate import load
from datasets import load_dataset, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from adversarial_attacks.utils import LANGUAGE


def map_to_pred(batch, processor, model, device, labels):
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    input_features = processor(audio_arrays,
                               sampling_rate=16000,
                               return_tensors="pt").input_features
    predicted_ids = model.generate(input_features.to(device))
    transcription = processor.batch_decode(predicted_ids, normalize=True)
    labels = [processor.tokenizer._normalize(text) for text in batch[labels]]
    batch['text'] = labels
    batch["transcription"] = transcription
    return batch


def main(conf):
    device = conf.device
    lang = LANGUAGE[conf.lang]
    if conf.data.path:
        dataset = load_from_disk(conf.data.path)
    else:
        dataset = load_dataset(conf.data.name,
                               lang["data"],
                               split="validation")

    processor = WhisperProcessor.from_pretrained(conf.model)
    if lang["model"]:
        processor.get_decoder_prompt_ids(lang["model"], task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        conf.model).to(device)

    map_to_pred_func = partial(map_to_pred,
                               processor=processor,
                               model=model,
                               device=device,
                               labels=conf.data.labels)

    result = dataset.map(map_to_pred_func,
                         batched=True,
                         batch_size=conf.batch_size)
    wer = load("wer")

    #for t in zip(result["text"], result["transcription"]):
    #    print(t)

    wer_dataset = wer.compute(predictions=result["transcription"],
                              references=result["text"])
    print(f"WER Value: {wer_dataset}")

    save_dir = f"{conf.lang}_{conf.attack}_results.hf"
    print(f"Saving dataset results at {save_dir}")
    result.save_to_disk(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path to configuration file")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config_path)
    main(conf)