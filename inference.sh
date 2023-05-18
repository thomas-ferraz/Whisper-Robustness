export PYTHONPATH=$(pwd)
python adversarial_attacks/scripts/inference.py \
    --config_path=adversarial_attacks/configs/inference/inference_en_clean.yaml
python adversarial_attacks/scripts/inference.py \
    --config_path=adversarial_attacks/configs/inference/inference_en_40.yaml
python adversarial_attacks/scripts/inference.py \
    --config_path=adversarial_attacks/configs/inference/inference_en_35.yaml