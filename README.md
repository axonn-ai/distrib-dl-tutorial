# ISC 24 - Tutorial on Distributed Training of Deep Neural Networks

[![Join slack](https://img.shields.io/badge/slack-axonn--users-blue)](https://join.slack.com/t/axonn-users/shared_invite/zt-2itbahk29-_Ig1JasFxnuVyfMtcC4GnA)

All the code for the hands-on exercies can be found in this repository. 

**Table of Contents**

* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Pipeline Parallelism](#pipeline-parallelism)

## Setup 

To request an account on Zaratan, please join slack at the link above, and fill [this Google form](https://forms.gle/bowh2GWQaG34EZyq6).

We have pre-built the dependencies required for this tutorial on Zaratan. This
will be activated automatically when you run the bash scripts.

The training dataset i.e. [MNIST](http://yann.lecun.com/exdb/mnist/) has also
been downloaded in `/scratch/zt1/project/isc/shared/data/MNIST`.

## Basics of Model Training

### Using PyTorch

```bash
cd session_1_basics/
sbatch --reservation=isc2024 run.sh
```

### Mixed Precision

```bash
MIXED_PRECISION=true sbatch --reservation=isc2024 run.sh
```

### Activation Checkpointing

```bash
CHECKPOINT_ACTIVATIONS=true sbatch --reservation=isc2024 run.sh
```

## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
cd session_2_data_parallelism
sbatch --reservation=isc2024 run_ddp.sh
```

### Zero Redundancy Optimizer (ZeRO)


```bash
sbatch --reservation=isc2024 run_deepspeed.sh
```

## Intra-layer (Tensor) Parallelism

```bash
cd session_3_intra_layer_parallelism
sbatch --reservation=isc2024 run.sh
```

## Inter-layer (Pipeline) Parallelism


```bash
cd session_4_inter_layer_parallelism
sbatch --reservation=isc2024 run.sh
```

### Hybrid Inter-layer (Pipeline) + Data Parallelism

```bash
HYBRID_PARR=true sbatch --reservation=isc2024 run.sh
```
