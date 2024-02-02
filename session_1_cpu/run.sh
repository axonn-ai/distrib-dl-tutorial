#!/bin/bash
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p standard
#SBATCH --time=06:00:00
#SBATCH -A bhatele-lab-aac

DATA_DIR="/home/sathwik7/scratch.bhatele-lab/tutorial-venv"

. /home/sathwik7/scratch.bhatele-lab/tutorial-venv/bin/activate

SCRIPT=train.py
ARGS="--num-layers 4 --hidden-size 2048 --data-dir ${DATA_DIR} --batch-size 32 --lr 0.001 --image-size 64"

MIXED_PRECISION="${MIXED_PRECISION:=false}"
CHECKPOINT_ACTIVATIONS="${CHECKPOINT_ACTIVATIONS:=false}"

if [ ${MIXED_PRECISION} == "true" ]; then
	SCRIPT="train_mp.py"
fi

if [ ${CHECKPOINT_ACTIVATIONS} == "true" ]; then
	SCRIPT="train_mp.py"
	ARGS="${ARGS} --checkpoint-activations"
fi

cmd="python ${SCRIPT} ${ARGS}"
echo $cmd

python ${SCRIPT} ${ARGS}

