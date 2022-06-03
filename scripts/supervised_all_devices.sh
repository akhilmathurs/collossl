#! /bin/bash 

positions=(thigh waist shin forearm head upperarm chest) #rw
# positions=(rua lla rshoe lshoe back) #opp
# positions=(s3mini_1 gear_1 lgwatch_1) #hhar
# positions=(hand chest ankle) #pamap2

types=(supervised)
mkdir ${exp_name}
dataset_name=realworld-3.0-0.0.dat
held_out_num_groups=5
#'realworld-3.0-0.0.dat', 'pamap2-2.0-1.0-v2.dat', 'hhar-2.0-0.5.dat', 'opportunity-2.0-0.0.dat'

exp_name=supervised_all_devices

working_directory=/mnt/data/gsl/runs/ # change working directory
dataset_path=/mnt/data/gsl/ # change path to dataset files

mkdir ${exp_name}
for device in ${positions[@]}; do
  for type in ${types[@]}; do
    python3 ../hparam_tuning_mp.py --exp_name=${exp_name}  --baseline="supervised_all_devices" --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0,1,2,3,4,5,6,7  --held_out_num_groups=${held_out_num_groups} --n_process_per_gpu=2 --training_mode=${type} --train_device=${device} --training_batch_size=512 --training_epochs 200 --dataset_name ${dataset_name}
  done
done


for device in ${positions[@]}; do
  for type in ${types[@]}; do
    python3 ../plot_results.py --${type} /mnt/data/gsl/runs/${device}/${exp_name}/${type}/logs/hparam_tuning_*/results_summary.csv --show_error_bar=False --out_file ${exp_name}/${device}_${type}.png 
  done
done