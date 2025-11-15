# Tutorial on Distributed Training of Deep Neural Networks

[![Join slack](https://img.shields.io/badge/slack-axonn--users-blue)](https://join.slack.com/t/axonn-users/shared_invite/zt-2itbahk29-_Ig1JasFxnuVyfMtcC4GnA)

All the code for the hands-on exercies can be found in this repository. 

**Table of Contents**

* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Inference](#inference)

## Setup 

To request an account on Zaratan, please join slack at the link above, and fill [this Google form](https://forms.gle/14g4ZTF4sJqUhg4o6).

We have pre-built the dependencies required for this tutorial on Zaratan. This
will be activated automatically when you run the bash scripts.

Model weights and the training dataset have 
been downloaded in `/scratch/zt1/project/sc25/shared/`.

## Basics of Model Training

### Using PyTorch Lightning

```bash
CONFIG_FILE=configs/single_gpu.json sbatch train_single.sh
```

### Mixed Precision
Open `configs/single_gpu.json` and change `precision` to `bf16-mixed` and then run - 

```bash
CONFIG_FILE=configs/single_gpu.json sbatch train_single.sh
```


## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
CONFIG_FILE=configs/ddp.json sbatch train_multi.sh
```

### Fully Sharded Data Parallelism (FSDP)


```bash
CONFIG_FILE=configs/fsdp.json sbatch train_multi.sh
```

## Tensor Parallelism

```bash
CONFIG_FILE=configs/axonn.json sbatch train_multi.sh
```

## Inference

Add more prompts to `data/inference/prompts.txt` if you want. Then run

```bash
CONFIG_FILE=configs/inference_yalis.json sbatch infer_single.sh
```

### With torch.compile

Open `infer.sh` and change `YALIS_DISABLE_COMPILE` from `1` to `0`. Then run 

```bash
CONFIG_FILE=configs/inference_yalis.json sbatch infer_single.sh
```

### With cuda graphs

Open `infer.sh` and change `YALIS_DISABLE_DECODE_CUDAGRAPHS` from `1` to `0` (make sure torch compile is also enabled). Then run 

```bash
CONFIG_FILE=configs/inference_yalis.json sbatch infer_single.sh
```

### With tensor parallelism

```bash
CONFIG_FILE=configs/inference_yalis.json sbatch infer_multi.sh
```

### Online Inference with VLLM

Query the vllm server we setup as follows:
```bash
# Usage: ./llm_request.sh <server_ip> "<prompt>" [max_tokens]

./llm_request.sh <vLLM Server IP> "San Francisco is a" 64
```

Change the prompt and the max tokens argument to play around with command
