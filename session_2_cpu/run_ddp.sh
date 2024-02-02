#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 64
#SBATCH -p standard
#SBATCH --time=02:00:00
#SBATCH -A bhatele-lab-aac 

DATA_DIR="/home/sathwik7/scratch.bhatele-lab/tutorial-venv"

module load gcc/9.4.0 openmpi/gcc
. /home/sathwik7/scratch.bhatele-lab/tutorial-venv/bin/activate

export MASTER_PORT="6000"
export MASTER_ADDR="localhost"

# Set Environment Vars
export OMP_NUM_THREADS=64

## Command for DDP
cmd="mpirun -np 2 python train_ddp.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64 --checkpoint-activations"

echo "${cmd}"
eval "${cmd}"

