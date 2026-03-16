# Second Run - 2026-03-13

## 1. Launch Docker Container

```bash
cd /home/user/kaiwen/cosmos-policy
docker run \
  -u root \
  -e HOST_USER_ID=$(id -u) \
  -e HOST_GROUP_ID=$(id -g) \
  -e HF_HOME=/home/cosmos/.cache/huggingface \
  -v $HOME/.cache:/home/cosmos/.cache \
  -v $(pwd):/workspace \
  --gpus all \
  --ipc=host \
  -it \
  --rm \
  -w /workspace \
  --entrypoint bash \
  cosmos-policy
```

## 2. Install Dependencies (inside container)

```bash
uv sync --extra cu128 --group aloha --python 3.10
```

## 3. Launch Training (inside container)

```bash
mkdir -p /workspace/training_output

export BASE_DATASETS_DIR=$(pwd)
export IMAGINAIRE_OUTPUT_ROOT=/workspace/training_output
export WANDB_API_KEY=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
export WANDB_PROJECT=cosmos-policy-v0

uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80" \
  2>&1 | tee /workspace/training_output/training.log
```

Training takes ~48 hours on 8xH100. Evaluate the 50K step checkpoint.
Checkpoints saved every 1000 steps to `$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/<experiment_name>/checkpoints/`.
