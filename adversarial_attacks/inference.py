from typing import List, Dict
from functools import partial
import argparse
import os
from omegaconf import OmegaConf

from evaluate import load
from datasets import load_dataset, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def map_to_pred(batch: List(Dict), processor: WhisperProcessor,
                model: WhisperForConditionalGeneration,
                device: str) -> List(Dict):
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    input_features = processor(audio_arrays,
                               sampling_rate=16000,
                               return_tensors="pt").input_features
    predicted_ids = model.generate(input_features.to(device))
    transcription = processor.batch_decode(predicted_ids, normalize=True)
    labels = [processor.tokenizer._normalize(text) for text in batch["text"]]
    batch['text'] = labels
    batch["transcription"] = transcription
    return batch


def main(conf):
    device = conf.device
    if conf.data.path is not None:
        dataset = load_from_disk(conf.data.path)
    else:
        dataset = load_dataset(conf.data.name,
                               conf.data.version,
                               split="validation")

    processor = WhisperProcessor.from_pretrained(conf.model.name)
    model = WhisperForConditionalGeneration.from_pretrained(
        conf.model.name).to(device)

    map_to_pred_func = partial(map_to_pred,
                               processor=processor,
                               model=model,
                               device=device)

    result = dataset.map(map_to_pred_func,
                         batched=True,
                         batch_size=conf.batch_size)
    wer = load("wer")

    #for t in zip(result["text"], result["transcription"]):
    #    print(t)

    wer_dataset = wer.compute(predictions=result["transcription"],
                              references=result["text"])
    print(f"WER Value: {wer_dataset}")

    save_dir = f"{conf.model.language}_{conf.data.tag}_results.hf"
    print(f"Saving dataset results at {save_dir}")
    result.save_to_disk(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to configuration file")
    args = parser.parse_args()

    conf = OmegaConf.load(args.config_path)
    main(conf)
