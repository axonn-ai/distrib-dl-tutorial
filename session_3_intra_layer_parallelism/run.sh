#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:05:00 
#SBATCH -A isc-aac

module load python gcc/9.4.0 cuda openmpi/gcc
VENV_HOME="/scratch/zt1/project/isc/shared/"

# Activate python virtual environment
source $VENV_HOME/tutorial-venv/bin/activate

DATA_DIR="$VENV_HOME/data"


G_INTRA_ROW=2
G_INTRA_COL=2
G_INTRA_DEPTH=1
G_DATA=1

torchrun --nproc_per_node 4 train.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-intra-d ${G_INTRA_DEPTH} --G-data 1  --micro-batch-size 4 --checkpoint-activations

