#! /bin/bash

cd ../

# HHAR 2 dataset
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_multi_task_hhar3' --gpu_device=4,5,6,7 --train_device='all' --held_out_num_groups=3 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="multi_task_transform" --dataset_name hhar-2.0-0.5.dat
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_en_co_hhar3' --gpu_device=0 --train_device='all' --held_out_num_groups=3 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="en_co_training"  --dataset_name hhar-2.0-0.5.dat
python3  hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_random_hhar3' --gpu_device=4,5,6,7 --train_device='all' --held_out_num_groups=3 --n_process_per_gpu=1  --eval_mode base_model --baseline="random"   --dataset_name hhar-2.0-0.5.dat

# PAMAP2 dataset
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_multi_task_pamap4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="multi_task_transform" --dataset_name pamap2-2.0-1.0-v2.dat
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_en_co_pamap4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="en_co_training"  --dataset_name pamap2-2.0-1.0-v2.dat
python3  hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_random_pamap4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --eval_mode base_model --baseline="random"   --dataset_name pamap2-2.0-1.0-v2.dat

# Opportunity dataset
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_multi_task_opp4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="multi_task_transform" --dataset_name opportunity-2.0-0.0.dat
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_en_co_opp4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="en_co_training"  --dataset_name opportunity-2.0-0.0.dat
python3  hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_random_opp4' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --eval_mode base_model --baseline="random"   --dataset_name opportunity-2.0-0.0.dat

# Real-World dataset
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_en_co_rw5' --gpu_device=7 --train_device='all' --held_out_num_groups=5 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="en_co_training"  --dataset_name realworld-3.0-0.0.dat
python3  hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_random_rw5' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --eval_mode base_model --baseline="random"   --dataset_name realworld-3.0-0.0.dat
python3 hparam_tuning_mp_ssl_baseline.py --exp_name='ssl_baseline_multi_task_rw5' --gpu_device=0 --train_device='all' --held_out_num_groups=4 --n_process_per_gpu=1  --training_batch_size=512 --eval_mode base_model --baseline="multi_task_transform" --dataset_name realworld-3.0-0.0.dat
