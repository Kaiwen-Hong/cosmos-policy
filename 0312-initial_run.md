# Initial Run - 2026-03-12

## 1. Build Docker Image

```bash
cd /home/user/kaiwen/cosmos-policy
docker build -t cosmos-policy docker
```

## 2. HuggingFace Login (on host, before launching container)

```bash
python -c "from huggingface_hub import login; login()"
```

Note: You need access to gated repos `nvidia/Cosmos-Predict2-2B-Video2World` and `nvidia/Cosmos-Policy-ALOHA-Predict2-2B`. Request access on their HuggingFace pages first.

## 3. Launch Docker Container

```bash
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

Note: `-e HF_HOME=/home/cosmos/.cache/huggingface` is required because container HOME is `/root` but the host cache is mounted at `/home/cosmos/.cache`.

## 4. Install ALOHA Dependencies (inside container)

```bash
uv sync --extra cu128 --group aloha --python 3.10
```

## 5. Download ALOHA Dataset (inside container)

```bash
uv run --extra cu128 --group aloha --python 3.10 hf download nvidia/ALOHA-Cosmos-Policy --repo-type dataset --local-dir ALOHA-Cosmos-Policy
export BASE_DATASETS_DIR=$(pwd)
```

Note: If rate-limited, login inside the container first:
```bash
uv run --extra cu128 --group aloha --python 3.10 python -c "from huggingface_hub import login; login()"
```

## 6. Launch ALOHA Training (inside container)

```bash
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

**Wandb & logging**: The experiment config enables `wandb` and `wandb_callback_actions` callbacks.
Without a valid API key, wandb will block all 8 torchrun workers waiting for interactive login.
- To enable (recommended): `export WANDB_API_KEY=<your-key>` (get it from https://wandb.ai/authorize)
- To disable: `export WANDB_MODE=disabled`
- **Important**: `stdout.log` is NOT created in the open-source version (gated behind `COSMOS_INTERNAL` flag).
  The detailed loss metrics (action L1, value loss, etc.) are only persisted via wandb.
  The `basic` callback prints `Iteration: N, average iter time: xx, total loss xx` to terminal every 5 steps,
  but this is lost if the terminal closes. Always use `| tee` to capture terminal output to a file.

Notes:
- **Training logs & checkpoints** are saved to `$IMAGINAIRE_OUTPUT_ROOT/{project}/{group}/{name}/`:
  - `checkpoints/` — model checkpoints (saved every 1000 steps)
  - `config.yaml` — saved config
  - `job_env.yaml`, `launch_info.yaml` — reproducibility info
  - Default output root is `/tmp/imaginaire4-output` (lost when container stops!), so we set `IMAGINAIRE_OUTPUT_ROOT=/workspace/training_output` to persist to the mounted volume
  - `stdout.log` is NOT auto-created (see above); use `| tee` to save terminal output
- Effective batch size = 25 (local) x 8 (GPUs) = 200 (no grad accum needed; 8xH100 is sufficient for ALOHA)
- Train until action L1 loss reaches ~0.010; evaluate the 50K step checkpoint
- LR schedule: warmup 2K steps, decay at 20K steps, then 5x lower constant LR
- Config: `cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py`
- Model input: 3 images (1 third-person + 2 wrist cameras), chunk_size=50, state_t=11
- The paper used 8 H100s for ~48 hours
- Best practice: test policy with the same GPU used to train it

## 7. Launch ALOHA Evaluation (inside container, after training)

ALOHA uses a server-client setup (unlike LIBERO/RoboCasa). See `ALOHA.md` for details.

- **Server side**: `cosmos_policy/experiments/robot/aloha/deploy.py`
- **Client side**: `cosmos_policy/experiments/robot/aloha/run_aloha_eval.py`

## Environment Info

- **GPU**: 8x NVIDIA H100 80GB HBM3
- **Docker**: 28.2.2
- **CUDA**: 13.1 (host) / 12.8.1 (container image)
- **Python**: 3.10.18 (managed by uv)
- **Packages**: 263 installed via uv
