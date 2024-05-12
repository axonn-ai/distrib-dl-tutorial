#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node 128 
#SBATCH --time=00:05:00 
#SBATCH -A isc-aac

module load python gcc/9.4.0 cuda
VENV_HOME="/scratch/zt1/project/isc/shared/"

# Activate python virtual environment
source $VENV_HOME/tutorial-venv/bin/activate

DATA_DIR="$VENV_HOME/data"


module load cuda

INSTALL_PATH="/scratch/zt1/project/isc/shared/openmpi/build/"
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

