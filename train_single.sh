#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A nairr-class
#SBATCH --mem=100G
#SBATCH --reservation=nairr


export SCRATCH="/scratch/zt1/project/nairr/shared/"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# variables needed for torch.distributed
export MASTER_ADDR=$(hostname)
USER_ID=$(( 0x$(echo -n "$USER" | md5sum | cut -c1-8) % 76 ))
BASE_PORT=29500
export MASTER_PORT=$(( BASE_PORT + USER_ID ))
echo "MASTER_PORT=$MASTER_PORT"

echo "Copying python environment to fast node local storage"
start=`date +%s`
mkdir -p /tmp/${USER}/tutorial_env
tar -xzf ${SCRATCH}/miniconda3.tar.gz -C /tmp/${USER}/tutorial_env
end=`date +%s`
runtime=$((end-start))
echo "Copy completed. Time taken = ${runtime} s"

# activate environment
source /tmp/${USER}/tutorial_env/bin/activate

CONFIG_FILE="${CONFIG_FILE:-configs/single_gpu.json}"
echo $CONFIG_FILE

# Run torchrun with specified number of GPUs
srun -u python -u train.py --config-file $CONFIG_FILE
