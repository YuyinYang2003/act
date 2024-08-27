#!/bin/bash
source activate aloha

# Transfer Cube task
python3 imitate_episodes.py \
--task_name sim_coordinated_lift_tray \
--ckpt_dir /ailab/user/yangyuyin/act/ckpt/sim_coordinated_lift_tray \
--policy_class ACT --kl_weight 10 --chunk_size 75 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0