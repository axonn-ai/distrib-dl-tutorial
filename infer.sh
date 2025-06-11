#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:06:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A isc-aac
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH --reservation=isc
#SBATCH --error=/dev/null


export SCRATCH="/scratch/zt1/project/isc/shared/"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}"

# variables needed for torch.distributed
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

# nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1


echo "Copying python environment to fast node local storage"
start=`date +%s`
mkdir -p /tmp/tutorial_env
tar -xzf ${SCRATCH}/miniconda3.tar.gz -C /tmp/tutorial_env
end=`date +%s`
runtime=$((end-start))
echo "Copy completed. Time taken = ${runtime} s"

# activate environment
source /tmp/tutorial_env/bin/activate

CONFIG_FILE="${CONFIG_FILE:-configs/inference_yalis.json}"
GPUS="${GPUS:-1}"

export YALIS_DISABLE_COMPILE=1
export YALIS_DISABLE_DECODE_CUDAGRAPHS=1

# Run torchrun with specified number of GPUs
srun -N 1 -n ${GPUS} -u ./get_rank.sh python -u infer.py --config-file $CONFIG_FILE
