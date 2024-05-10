#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128
#SBATCH --time=00:05:00
#SBATCH -A bhatele-lab-cmsc 

module load python gcc/9.4.0 cuda openmpi/gcc
# CHANGE AS PER PROJECT
VENV_HOME="/scratch/zt1/project/bhatele-lab/shared/parallel-deep-learning"
source $VENV_HOME/tutorial-venv/bin/activate
DATA_DIR="$VENV_HOME/data"


cmd="torchrun --nproc_per_node 2 train_deepspeed.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --checkpoint-activations --deepspeed_config ./ds_config.json" 

echo "${cmd}"

eval "${cmd}"
