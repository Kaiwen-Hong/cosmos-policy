#!/usr/bin/env bash
# Upload checkpoints from inside Docker container (runs as root, no permission issues)
# Usage: bash 0316-upload-checkpoint-docker.sh

set -euo pipefail

REPO_ID="kaiwen2/vca-cosmospolicy"
CKPT_BASE="/workspace/training_output/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80/checkpoints"
CHECKPOINTS=("iter_000050000" "iter_000080000")

read -rsp "Enter your HuggingFace token: " HF_TOKEN
echo

for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_path="$CKPT_BASE/$ckpt"
    if [ ! -d "$ckpt_path" ]; then
        echo "ERROR: $ckpt_path does not exist, skipping."
        continue
    fi
    echo "Uploading $ckpt (~5.5GB)..."
    HF_TOKEN="$HF_TOKEN" uv run --extra cu128 --group aloha --python 3.10 python -c "
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
