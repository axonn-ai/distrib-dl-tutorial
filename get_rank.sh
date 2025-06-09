#!/bin/bash
# select_gpu_device wrapper script
export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}

export TORCHINDUCTOR_CACHE_DIR="/tmp/$USER/.cache/torchinductor/torchinductor_${RANK}"
export TRITON_HOME="/tmp/$USER/.cache/triton/triton_${RANK}"
export TRITON_CACHE_DIR="/tmp/$USER/.cache/triton/triton_${RANK}"

exec $*
