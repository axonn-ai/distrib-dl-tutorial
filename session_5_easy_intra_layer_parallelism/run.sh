#!/bin/bash
#SBATCH -n 4 
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -c 16
#SBATCH -p standard
#SBATCH --time=02:00:00 
#SBATCH -A bhatele-lab-aac 

DATA_DIR="/home/sathwik7/scratch.bhatele-lab/tutorial-venv"

module load gcc/9.4.0
module load openmpi
. /home/sathwik7/scratch.bhatele-lab/tutorial-venv/bin/activate

G_INTRA_ROW=2
G_INTRA_COL=2

# Set Environment Variables
export OMP_NUM_THREADS=16

cmd="srun -n 4 python train.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-data 1  --micro-batch-size 4 --checkpoint-activations"

echo ${cmd}
eval ${cmd}

