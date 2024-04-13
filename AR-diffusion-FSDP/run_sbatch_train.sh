#!/bin/bash

#SBATCH --job-name=train_ARD
#SBATCH --output=/home/<username>/logs/train_ARD_%j.out
#SBATCH --error=/home/<username>/logs/train_ARD_%j.err
#SBATCH --time=1-0
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=18G
#SBATCH --partition=gpu-a100
#SBATCH --mail-user=test@test.com
#SBATCH --mail-type=ALL

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export STEPS=100000

module purge

module load miniconda3
module load cuda12.1/toolkit/12.1.1
module load cuda12.1/fft/12.1.1
module load cuda12.1/blas/12.1.1
module load cudnn8.9-cuda12.1/8.9.4.25
module load slurm

conda activate /home/<username>/test-fsdp
cd /home/<username>/ProphetNet-FSDP/AR-diffusion-FSDP

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $SLURM_NTASKS_PER_NODE \
--rdzv_id $SLURM_JOB_ID \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:0 \
train_utils/trainer_main.py \
model.name="bert-base-uncased" \
batch_size=128 \
grad_accum=3 \
total_steps=$STEPS \
data.name=xsum \
tgt_len=50 \
max_pos_len=512 \
lr=1e-4 \
lr_step=40000 \
dropout=0.1 \
in_channels=512 \
out_channels=512 \
time_channels=128 \
eval_interval=3000 \
log_interval=1 \
schedule_sampler="xy_uniform" \
time_att=true \
att_strategy="txl" \
use_AMP=true 
