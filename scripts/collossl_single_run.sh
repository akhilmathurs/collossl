#! /bin/bash

working_directory=/mnt/data/gsl/runs/ # change working directory
dataset_path=/mnt/data/gsl/ # change path to dataset files

exp_name=collossl_single_run
train_device=chest
eval_device=chest
dataset_name=realworld-3.0-0.0.dat

python3 ../contrastive_training.py --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

