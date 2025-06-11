# SC 24 - Tutorial on Distributed Training of Deep Neural Networks

[![Join slack](https://img.shields.io/badge/slack-axonn--users-blue)](https://join.slack.com/t/axonn-users/shared_invite/zt-2itbahk29-_Ig1JasFxnuVyfMtcC4GnA)

All the code for the hands-on exercies can be found in this repository. 

**Table of Contents**

* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Inference](#inference)

## Setup 

To request an account on Zaratan, please join slack at the link above, and fill [this Google form](https://forms.gle/MSVc3ARbgqwu2wUDA).

We have pre-built the dependencies required for this tutorial on Zaratan. This
will be activated automatically when you run the bash scripts.

Model weights and the training dataset have 
been downloaded in `/scratch/zt1/project/sc24/shared/`.

## Basics of Model Training

### Using PyTorch Lightning

```bash
CONFIG_FILE=configs/single_gpu.json sbatch --ntasks-per-node=1 --gres=gpu:a100:1 train.sh
```

### Mixed Precision
Open `configs/single_gpu.json` and change `precision` to `bf16-mixed` and then run - 

```bash
CONFIG_FILE=configs/single_gpu.json sbatch --ntasks-per-node=1 --gres=gpu:a100:1 train.sh
```


## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
CONFIG_FILE=configs/ddp.json sbatch --ntasks-per-node=4 --gres=gpu:a100:4 train.sh
```

### Fully Sharded Data Parallelism (FSDP)


```bash
CONFIG_FILE=configs/fsdp.json sbatch --ntasks-per-node=4 --gres=gpu:a100:4  train.sh
```

## Tensor Parallelism

```bash
CONFIG_FILE=configs/axonn.json sbatch --ntasks-per-node=4 --gres=gpu:a100:4 train.sh
```

## Inference

Add more prompts to `data/inference/prompts.txt` if you want. Then run

```bash
GPUS=1 CONFIG_FILE=configs/inference_yalis.json sbatch --ntasks-per-node=1 infer.sh
```

### With torch.compile

Open `infer.sh` and change `YALIS_DISABLE_COMPILE` from `1` to `0`. Then run 

```bash
GPUS=1 CONFIG_FILE=configs/inference_yalis.json sbatch --ntasks-per-node=1  infer.sh
```

### With cuda graphs

Open `infer.sh` and change `YALIS_DISABLE_DECODE_CUDAGRAPHS` from `1` to `0` (make sure torch compile is also enabled). Then run 

```bash
GPUS=1 CONFIG_FILE=configs/inference_yalis.json sbatch --ntasks-per-node=1  infer.sh
```

### With tensor parallelism

```bash
GPUS=4 CONFIG_FILE=configs/inference_yalis.json sbatch --ntasks-per-node=4 --gres=gpu:a100:4 infer.sh
```


### Online Inference with VLLM

For session host: Take an interactive session
```bash
sinteractive -N 1 -G -g gpu:a100:1 -c 32 -t 59 -A isc-aac --exclusive --mem=500G --reservation=isc
```

Then run:
```bash
bash vllm_serve.sh
```

For participants:
```bash
curl http://<Server IP>:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 32,
        "temperature": 0
    }'
```
