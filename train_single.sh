#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A sc25-aac
#SBATCH --mem=100G
#SBATCH --reservation=sc25


export SCRATCH="/scratch/zt1/project/sc25/shared/"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# variables needed for torch.distributed
export MASTER_ADDR=$(hostname)
START_PORT=29500
PORT=$START_PORT

while true; do
    if netstat -tuln | grep -q ":$PORT "; then
        PORT=$((PORT+1))
    else
        export MASTER_PORT=$PORT
        echo "MASTER_PORT=$MASTER_PORT"
        break
    fi

    if [ $PORT -gt 65535 ]; then
        echo "No available ports"
        exit 1
    fi
done

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
