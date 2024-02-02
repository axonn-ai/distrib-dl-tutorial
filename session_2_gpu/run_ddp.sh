#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:45:00
#SBATCH -A bhatele-lab-aac 

DATA_DIR="/home/sathwik7/axonn"

module load gcc/9.4.0 openmpi/gcc
. /home/sathwik7/axonn/tutorial-venv/bin/activate

pip install mpi4py

## Command for DDP
cmd="mpirun -np 4 python train_ddp.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --checkpoint-activations"

echo "${cmd}"
eval "${cmd}"

