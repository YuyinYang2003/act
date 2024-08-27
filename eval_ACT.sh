#!/bin/bash
module load anaconda/2022.10
source activate aloha

# Transfer Cube task
python3 imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir /ailab/user/yangyuyin/act/ckpt \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0 \
--eval