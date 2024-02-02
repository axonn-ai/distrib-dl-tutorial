#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -p standard 
#SBATCH --time=01:15:00 
#SBATCH -A bhatele-lab-aac 

DATA_DIR="/home/sathwik7/scratch.bhatele-lab/tutorial-venv/"

module load gcc/9.4.0
module load openmpi
. /home/sathwik7/scratch.bhatele-lab/tutorial-venv/bin/activate

INSTALL_PATH="/home/sathwik7/scratch.bhatele-lab/tutorial-venv/"
export PATH="${PATH}:${INSTALL_PATH}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PATH}/lib"

## Command for DDP


HYBRID_PARR="${HYBRID_PARR:=false}"

G_INTER=4

if [ ${HYBRID_PARR} == "true" ]; then
	G_INTER=2
fi

G_DATA=$(( 4 / G_INTER ))

echo ${G_DATA}
echo ${G_INTER}

cmd="mpirun -np 4 python train_axonn_inter_layer.py --num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.0001 --image-size 64 --G-inter ${G_INTER} --G-data ${G_DATA} --micro-batch-size 4 --checkpoint-activations"

echo ${cmd}
eval ${cmd}

