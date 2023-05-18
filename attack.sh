export PYTHONPATH=$(pwd)
python adversarial_attacks/scripts/attack_dataset.py \
    --config_path=adversarial_attacks/configs/attack/attack_fr_40.yaml
python adversarial_attacks/scripts/attack_dataset.py \
    --config_path=adversarial_attacks/configs/attack/attack_fr_35.yaml