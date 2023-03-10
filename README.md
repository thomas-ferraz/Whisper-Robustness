# Improving Large-Scale Speech Recognition Robustness by Language Specialization

[IPOL DEMO](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000393)
Latest version: 1.0

The recent [Whisper](https://github.com/openai/whisper/discussions/654) proposes a multi-task weak supervised training on a large-scale dataset collected from the internet. Although the model presents important gains, especially in English, its robustness and limitations in multilingual and low-resource scenarios have not yet been sufficiently explored. In this work, we present a detailed description of this new model; we propose a method to mitigate the performance gap presented in low-resource languages; and we perform an in-depth study of its robustness across different noise types and languages. The results show the effectiveness of language specialization in reducing the performance gap in speech transcription and even in increasing in the model robustness on noisy data.

The code is divided mainly into to scripts. The [`train_eval.py`](https://github.com/thomas-ferraz/Whisper-Robustness/blob/main/train_eval.py) contains the main pipeline for the finetuning and evaluations. And the [`data_utils.py`](https://github.com/thomas-ferraz/Whisper-Robustness/blob/main/data_utils.py) contains preprocessing functions as well as the `DataCollator`.

## Prerequisites
```
pip install -r requirements.txt
```
## Evaluate Whisper with Degradations
##  Usage
```
python train_eval.py --size tiny --dataset google/fleurs --lang fr --output_dir tiny_fr  --normalize "lower" --train 1 
```
|Hyperparameters| Usage                        |
|---------------|------------------------------|
| `size`          | A string to specify the Whisper model size to use (e.g., tiny, small,..) |
| `finetuned`     | An integer value to specify whetherto use pre-trained model or the fine-tuned one (0: pre-trained, 1: fine-tuned)|
| `tokenizer_name`     | The initial tokenizer |
| `dataset`     | The dataset that will be used to run the code |
| `lang`     | The language code |
| `task`     | The task on which the model will be fine-tuned|
| `output_dir`     | The directory where to save the model |
| `cpu_mode`     | Runs on CPU if set to 1|
| `per_device_train_batch_size`     | The batch size per device during training|
| `gradient_accumulation_steps`     | Number of accumulation steps|
| `learning_rate`     | Learning rate |
| `warmup_steps`     | Number of warmup steps |
| `weight_decay`     | Weight decay |
| `max_steps`     | Maximum number of steps |
| `fp16`     |  Mixed precision if set to 1|
| `per_device_eval_batch_size` | Batch size per device during evaluation|
| `eval_steps`     | Number of evaluations |
| `logging_steps`     | Number of steps before displaying log messages |
| `fix_forced_decoder_ids`     | Fix the language when using the model|
| `train`     | Finetunes if set to 1 and evaluates if 0|
| `patience`     | Patience before early stopping|
| `use_peft`     | Use PEFT if  set to 1 |
| `early_stopping_threshold`     | Threshold for applying early stopping|
| `eval_robustness`     | Multiple evaluations if set to 1 |
| `degradations_path`     | Path to JSON file precising the degradations |
| `debug`     | Debug mode on 10 samples if set to 1|
| `normalize`     | Text standardization to apply  |

## Evaluate Whisper with Degradations
You can easily apply various degradations to a dataset to evaluate Whisper's Robustness.

### Prerequisites
The sox library should be installed separatly with `apt-get`.
```
apt-get update && apt-get install -y sox
```
### Usage
##### Single Evaluation
This is an example of the command line for a single evaluation of Whisper on a dataset.
```
python train_eval.py  --lang fr --output_dir results_fr --degradations degradations.json  --normalize lower --train 0 --fix_forced_decoder_ids 1
```
The degradations should be specified in a JSON file, similar to [`degradatiosn.json`](https://github.com/thomas-ferraz/Whisper-Robustness/blob/main/degradations.json). It contains a list of one dictionary, indicating the degradation string and the probability of applying it to audio in the datasets, which should be 1 in this case. More degradation strings can be found at the [audio_degrader](https://github.com/emilio-molina/audio_degrader) repository.
```json
[
  {
    "degradation": ["mix,sounds/ambience-pub.wav,6"],
    "prob": 1
  }
]
```
##### Multiple Evaluations
For multiple evaluations, the option `eval_robustness` should be set to 1.
```
python train_eval.py  --lang fr --output_dir results_dr --degradations evaluate_robustness.json --eval_robustness 1 --normalize "lower" --train 0  --fix_forced_decoder_ids 0
```
For this task, the JSON file should have a different format, similar to [`evaluate_robustness.json`](https://github.com/thomas-ferraz/Whisper-Robustness/blob/main/evaluate_robustness.json). Here the degradation string is split on commas into `name`, `param1` and if there is a `param2`.  
```json
[
  {
    "name": "mix",
    "param1":{
        "values": ["sounds/white-noise.wav",
                  "sounds/ambience-pub.wav", 
                  "sounds/helen.wav"],
        "name": "noise"
    },
    "param2":{
        "values": [100,40,35,30,25,20,15,10,5,0,-5,-10],
        "name": "snr"
    }
  }
]
```
## Authors
- Thomas Palmeira Ferraz - thomas.palmeira@telecom-paris.fr
- Helene Maxcici - helene.maxcici@ens-paris-saclay.fr
- Teysir Baoueb - teysir.baoueb@ensta-paris.fr

## Copyright and license information
Copyright (c) 2023 Thomas Palmeira Ferraz, Helene Maxcici, Teysir Baoueb

This code is licensed under the Apache License, Version 2.0 (the "License"); you may not use this code except in compliance with the License.

You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.