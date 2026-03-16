#!/usr/bin/env bash
# Upload selected checkpoints to HuggingFace Hub
# Usage: bash 0315-upload-checkpoint.sh

set -euo pipefail

REPO_ID="kaiwen2/vca-cosmospolicy"
CKPT_BASE="/home/user/kaiwen/cosmos-policy/training_output/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80/checkpoints"
CHECKPOINTS=("iter_000050000" "iter_000075000")

# Prompt for HF token
read -rsp "Enter your HuggingFace token: " HF_TOKEN
echo

pip install -q huggingface_hub

for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_path="$CKPT_BASE/$ckpt"
    if [ ! -d "$ckpt_path" ]; then
        echo "ERROR: $ckpt_path does not exist, skipping."
        continue
    fi
    echo "Uploading $ckpt (~5.5GB)..."
    HF_TOKEN="$HF_TOKEN" python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo('$REPO_ID', repo_type='model', exist_ok=True)
api.upload_folder(
    repo_id='$REPO_ID',
    folder_path='$ckpt_path',
    path_in_repo='$ckpt',
    repo_type='model',
)
print('Done: $ckpt')
"
done

echo "All uploads complete: https://huggingface.co/$REPO_ID"
