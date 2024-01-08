#!/bin/bash
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=2
#SBATCH --account=m2404_g
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00

ulimit -c unlimited

DIR=`pwd`
NNODES=1
#GPUS=$(( NNODES * 4 ))
GPUS=2
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
#export NCCL_DEBUG=INFO
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1

. ~/venvs/axonnenv_dev/bin/activate

G_INTRA_ROW=1
G_INTRA_COL=1
G_INTRA_DEP=2

SCRIPT="python -u train_convnet_intra_layer.py --data-dir ${SCRATCH} --batch-size 32 --lr 0.001 --image-size 64 --G-intra-r ${G_INTRA_ROW} --G-intra-c ${G_INTRA_COL} --G-intra-d ${G_INTRA_DEP} --G-data 1  --micro-batch-size 8 --checkpoint-activations"

cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=2 ${SCRIPT}"

echo ${cmd}
eval ${cmd}

