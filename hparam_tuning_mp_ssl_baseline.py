'''

1) To view the results in tensorboard, run the following inside docker with the appropriate values. Port number is the port you opened while running the docker container. 

tensorboard --logdir={working_directory}/logs/hparam_tuning_{exp_name}/ --port 6055 --bind_all

E.g., tensorboard --logdir=/mnt/data/gsl/runs/logs/hparam_tuning_waist_baseline/ --port 6055 --bind_all


2) Once tensorboard is running, you can ssh into the machine with port forwarding. Change the port as per your docker scripts

ssh -L 6055:localhost:6055 <server_machine_name>

Then go to your browser and open localhost:6055

'''


# thigh/supervised/models/

# thigh/supervised/logs/

# thigh/supervised/results

from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from contrastive_training import train, fine_tune_evaluate
from tabulate import tabulate
import argparse, os, datetime
import csv
import numpy as np
import multiprocessing as mp
import copy
import hashlib
import time
import sys
import signal

import load_data
from common_parser import get_parser
import simclr_utitlities
import contrastive_training
import ssl_baseline

parser = get_parser()
args = parser.parse_args()
if args.eval_device is None:
    args.eval_device = args.train_device
if args.fine_tune_device is None:
    args.fine_tune_device = args.train_device

if args.training_mode == 'single' or args.training_mode == 'multi':
    args.eval_mode = 'base_model'

# args = vars(args) #convert to a dict 


'''
Specify the training hyperparams here
'''

HP_LR_DECAY = hp.HParam('learning_rate_decay', hp.Discrete(['cosine']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-5]))
HP_OPTIM = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_TAKE = hp.HParam('take', hp.Discrete([1.0])) #percentage of the dataset used for training
HP_ARCH = hp.HParam('architecture', hp.Discrete(['1d_conv'])) # iterate over training architectures. TODO: Not implemented yet
# HP_AUG = hp.HParam('augmentations', hp.Discrete(['none'])) #add various augmentations. TODO: not implemented yet
HP_DEVICE_SELECTION_STRATEGY = hp.HParam('dev_sel_strategy', hp.Discrete(['none']))
HP_WEIGHTED_COLLOSSL = hp.HParam('weighted_collossl', hp.Discrete([False]))
HP_NEG_SAMPLE_SIZE = hp.HParam('neg_sample_size', hp.Discrete([1]))

### -------------------------- These are hard-coded for realworld. Needs changes for opportunity
if args.dataset_name.find('realworld')!=-1:
  if args.held_out_num_groups is None:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])) # iterate over training architectures. TODO: Not implemented yet
  else:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(args.held_out_num_groups)]))
  HP_EVAL_DEVICE = hp.HParam('eval_device', hp.Discrete(['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin']))
elif args.dataset_name.find('opportunity')!=-1:
  if args.held_out_num_groups is None:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(4)])) # iterate over training architectures. TODO: Not implemented yet
  else:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(args.held_out_num_groups)]))
  HP_EVAL_DEVICE = hp.HParam('eval_device', hp.Discrete(['rua','lla','rshoe','lshoe','back']))
elif args.dataset_name.find('pamap2')!=-1:
  if args.held_out_num_groups is None:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(8)])) # iterate over training architectures. TODO: Not implemented yet
  else:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(args.held_out_num_groups)]))
  HP_EVAL_DEVICE = hp.HParam('eval_device', hp.Discrete(['hand', 'chest', 'ankle']))
elif args.dataset_name.find('hhar')!=-1:
  if args.held_out_num_groups is None:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(6)])) # iterate over training architectures. TODO: Not implemented yet
  else:
    HP_HELD_OUT = hp.HParam('held_out', hp.Discrete([i for i in range(args.held_out_num_groups)]))
  HP_EVAL_DEVICE = hp.HParam('eval_device', hp.Discrete(['s3mini_1', 'gear_1', 'lgwatch_1']))

if args.baseline == 'multi_task_transform':
  HP_FINETUNE_TAKE = hp.HParam('fine_tune_take', hp.Discrete([0.1,0.25,0.5,0.75,1.0]))

  HP_AUG = hp.HParam('augmentations', hp.Discrete(['none']))
  HP_TEMPERATURE = hp.HParam('contrastive_temperature', hp.Discrete([-1.]))

  HP_MULTI_SAMPLING = hp.HParam('multi_sampling_mode', hp.Discrete(['sync_all']))
  HP_DYNAMIC_DEVICE_SELECTION = hp.HParam('dynamic_device_selection', hp.Discrete([0]))
elif args.baseline == 'en_co_training':
  HP_FINETUNE_TAKE = hp.HParam('fine_tune_take', hp.Discrete([0.1,0.25,0.5,0.75,1.0]))

  HP_AUG = hp.HParam('augmentations', hp.Discrete(['none']))
  HP_TEMPERATURE = hp.HParam('contrastive_temperature', hp.Discrete([-1.]))

  HP_MULTI_SAMPLING = hp.HParam('multi_sampling_mode', hp.Discrete(['sync_all']))
  HP_DYNAMIC_DEVICE_SELECTION = hp.HParam('dynamic_device_selection', hp.Discrete([0]))
elif args.baseline == 'random':
  HP_FINETUNE_TAKE = hp.HParam('fine_tune_take', hp.Discrete([0.1,0.25,0.5,0.75,1.0]))

  HP_AUG = hp.HParam('augmentations', hp.Discrete(['none']))
  HP_TEMPERATURE = hp.HParam('contrastive_temperature', hp.Discrete([-1.]))

  HP_MULTI_SAMPLING = hp.HParam('multi_sampling_mode', hp.Discrete(['sync_all']))
  HP_DYNAMIC_DEVICE_SELECTION = hp.HParam('dynamic_device_selection', hp.Discrete([0]))


HP_POS_NEG_DEVICES = hp.HParam('positive_negative_devices', hp.Discrete(["([], [])"]))

METRIC_F1_MACRO = 'f1_macro'
METRIC_F1_WEIGHTED = 'f1_weighted'


working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)
if not os.path.exists(working_directory):
    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
    os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)  

start_time = str(int(datetime.datetime.now().timestamp()))
hparams = [HP_LR_DECAY, HP_LR, HP_TAKE, HP_FINETUNE_TAKE, HP_OPTIM, HP_ARCH, HP_AUG, HP_TEMPERATURE, HP_DEVICE_SELECTION_STRATEGY, HP_WEIGHTED_COLLOSSL, HP_POS_NEG_DEVICES, HP_MULTI_SAMPLING, HP_HELD_OUT, HP_DYNAMIC_DEVICE_SELECTION, HP_EVAL_DEVICE, HP_NEG_SAMPLE_SIZE]
metrics = [hp.Metric(METRIC_F1_MACRO, display_name='f1_macro'), hp.Metric(METRIC_F1_WEIGHTED, display_name='f1_weighted')]

fieldnames = [p.name for p in hparams] + [m._display_name for m in metrics]


# working_directory = args.working_directory if args.working_directory.endswith("/") else args.working_directory + "/"
    
# start_time = datetime.datetime.now()
# start_time_str = start_time.strftime("%Y%m%d-%H%M%S")



def consumer(consumer_id, job_queue, results_queue, setup_args):
  job = None
  try:
    # with tf.device(f'/gpu:{consumer_id}'):
      print(f'{consumer_id} {os.getpid()} Starting consumer ')
      results_queue.put({
        'type': 'init',
        'data': os.getpid()
      })
      consumer_vars = consumer_setup(consumer_id, setup_args)
      while(True):
          job = job_queue.get(timeout=60)
          print(f"{consumer_id} {os.getpid()} get job ") # {job}
          if job is None:
              break
          else:
              return_value = process_job(consumer_vars, job)
              results_queue.put({
                'type': 'job_finished',
                'data': return_value
              })
      print(f"{consumer_id} {os.getpid()} exitting")


  finally:
    if job is not None:
      job_queue.put(job)

    results_queue.put(None)
    print(f'Stopping consumer {consumer_id} {os.getpid()}')


def consumer_setup(consumer_id, setup_args):
  print(consumer_id, setup_args)
  # sys.stdout = open(str(consumer_id) + ".out", "w")
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  if args.baseline == 'en_co_training':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = setup_args["CUDA_VISIBLE_DEVICES"]
  os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      try:
          for sel_gpu in gpus:
              tf.config.experimental.set_memory_growth(sel_gpu, True)
              tf.config.experimental.set_virtual_device_configuration(
                sel_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
              )
      except RuntimeError as e:
          print(e)

  dataset_full = load_data.Data(args.dataset_path, args.dataset_name)
      
  return {
    'id': consumer_id,
    'tf_dataset_full': dataset_full
  }

def process_job(consumer_vars, job):
  all_results = []

  tf_dataset_full = consumer_vars['tf_dataset_full']
  
  args = job['args']
  # tf_dataset_full = load_data.Data(args.dataset_path, args.dataset_name, held_out=args.held_out)

  session_num = 0

  if args.baseline == 'en_co_training' or args.baseline == 'multi_task_transform' or args.baseline == 'random':
    
    for eval_device in HP_EVAL_DEVICE.domain.values:
      args.eval_device = eval_device
      args.train_device = args.eval_device
      args.fine_tune_device = args.eval_device
      pos_neg_devices = args.pos_neg_devices

      if args.baseline == 'multi_task_transform':
        tf.keras.backend.clear_session()
        trained_model_save_path = ssl_baseline.train_multi_task_transform(tf_dataset_full, args)

      for ft_take in HP_FINETUNE_TAKE.domain.values:
        args.fine_tune_take = ft_take
        hparams = {
            HP_LR_DECAY: args.learning_rate_decay,
            HP_LR: args.learning_rate,
            HP_OPTIM: args.optimizer,
            HP_TAKE: args.take,
            HP_FINETUNE_TAKE: ft_take,
            HP_ARCH: args.model_arch,
            HP_AUG: args.data_aug,
            HP_TEMPERATURE: args.contrastive_temperature,
            HP_POS_NEG_DEVICES: pos_neg_devices,
            HP_MULTI_SAMPLING: args.multi_sampling_mode,
            HP_HELD_OUT: args.held_out,
            HP_EVAL_DEVICE: eval_device,
            HP_DYNAMIC_DEVICE_SELECTION: args.dynamic_device_selection,
            HP_DEVICE_SELECTION_STRATEGY: args.device_selection_strategy,
            HP_WEIGHTED_COLLOSSL: args.weighted_collossl,
            HP_NEG_SAMPLE_SIZE: args.neg_sample_size,
        }

        print(args)

        run_name = "run-%d-%d" % (consumer_vars['id'], int(datetime.datetime.now().timestamp() * 1000.0))
        with tf.summary.create_file_writer(working_directory + '/logs/hparam_tuning_' + start_time + "/" + run_name + "-" + str(ft_take)).as_default():
          hp.hparams(hparams)

          if args.baseline == 'multi_task_transform':
            test_f1_macro, test_f1_weighted, tsne_image = fine_tune_evaluate(tf_dataset_full, trained_model_save_path, args)
          elif args.baseline == 'en_co_training':
            test_f1_macro, test_f1_weighted, = ssl_baseline.train_and_evaluate_en_co_training(tf_dataset_full, args)
          elif args.baseline == 'random':
            test_f1_macro, test_f1_weighted = ssl_baseline.Random(tf_dataset_full, args)
        result_dict = {}
        for k, v in hparams.items():
          result_dict[k.name] = v
        result_dict[METRIC_F1_MACRO] = test_f1_macro
        result_dict[METRIC_F1_WEIGHTED] = test_f1_weighted

        with open(working_directory + '/logs/hparam_tuning_' + start_time + '/results_summary.csv', 'a', newline='') as csv_file:
          csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          csv_writer.writerow(result_dict)

        session_num += 1
    return session_num
  #@Ian: Is the below code useful?
  all_devices = list(tf_dataset_full.info['device_list'])

    # trained_model_save_path = ''
  if args.baseline == 'multi_task_transform':
    trained_model_save_path = ssl_baseline.train_multi_task_transform(tf_dataset_full, args)
  else:
    trained_model_save_path = train(tf_dataset_full, args)

  for eval_device in HP_EVAL_DEVICE.domain.values:

    run_name = "run-%d-%d" % (consumer_vars['id'], int(datetime.datetime.now().timestamp() * 1000.0))
    apriori_device_selection = False
    if args.dynamic_device_selection==0 and (len(args.positive_devices)==0 or len(args.negative_devices)==0): # apriori device selection
      apriori_device_selection = True
      positive_devices = [contrastive_training.all_devices[j] for j in contrastive_training.positive_indices]
      negative_devices = [contrastive_training.all_devices[j] for j in contrastive_training.negative_indices]
      pos_neg_devices = str(tuple([positive_devices, negative_devices]))
    else:
      pos_neg_devices = args.pos_neg_devices

    for ft_take in HP_FINETUNE_TAKE.domain.values:
      hparams = {
          HP_LR_DECAY: args.learning_rate_decay,
          HP_LR: args.learning_rate,
          HP_OPTIM: args.optimizer,
          HP_TAKE: args.take,
          HP_FINETUNE_TAKE: ft_take,
          HP_ARCH: args.model_arch,
          HP_AUG: args.data_aug,
          HP_TEMPERATURE: args.contrastive_temperature,
          HP_POS_NEG_DEVICES: pos_neg_devices,
          HP_MULTI_SAMPLING: args.multi_sampling_mode,
          HP_HELD_OUT: args.held_out,
          HP_EVAL_DEVICE: eval_device,
          HP_DYNAMIC_DEVICE_SELECTION: args.dynamic_device_selection,
          HP_DEVICE_SELECTION_STRATEGY: args.device_selection_strategy,
          HP_WEIGHTED_COLLOSSL: args.weighted_collossl,
          HP_NEG_SAMPLE_SIZE: args.neg_sample_size,
      }

      
      current_time = str(datetime.datetime.now())                  
      print('--- Count: %d.....Starting trial %s at time %s' % (session_num, run_name,current_time))
      hyper = "_".join([str(hparams[h]) for h in hparams])
      print(hyper)

      with tf.summary.create_file_writer(working_directory + '/logs/hparam_tuning_' + start_time + "/" + run_name + "-" + str(ft_take)).as_default():
          hp.hparams(hparams)  # record the values used in this trial

          args.eval_device = eval_device
          args.fine_tune_take = ft_take
          
          print(args)

          # test_f1_macro, test_f1_weighted, tsne_image = np.random.randint(100)/100, np.random.randint(100)/100, None
          test_f1_macro, test_f1_weighted, tsne_image = fine_tune_evaluate(tf_dataset_full, trained_model_save_path, args)

          tf.summary.scalar(METRIC_F1_MACRO, test_f1_macro, step=1)
          tf.summary.scalar(METRIC_F1_WEIGHTED, test_f1_weighted, step=1)
          if tsne_image is not None:
            tf.summary.image("T-SNE", tsne_image, step=1)
          
          if args.dynamic_device_selection==1: #getting dynamic device selection stats and logging them
            tmp = simclr_utitlities.dds_pair_stats
            tmp_list = []
            for key in tmp.keys():
              pos_device = [contrastive_training.all_devices[j] for j in key[0]]  
              neg_device = [contrastive_training.all_devices[j] for j in key[1]]  
              tmp_list.append([str([pos_device, neg_device]), tmp[key]])
            tmp_list.sort(key= lambda x: x[1], reverse=True)
            with open(f"{working_directory}/logs/hparam_tuning_{start_time}/device_stats_{eval_device}.txt", 'a') as f:
              f.write(str(args))
              f.write(f"\n====== Dynamic Device Selection stats ======\n")
              f.write(tabulate(tmp_list, headers=['Pos neg device pair', 'count']))
              f.write('\n')
            print(tabulate(tmp_list, headers=['Pos neg device pair', 'count']))
            simclr_utitlities.dds_pair_stats = {}

            tmp = simclr_utitlities.dds_pos_device_stats
            tmp_list = [[contrastive_training.all_devices[key], tmp[key]] for key in tmp.keys()]
            tmp_list.sort(key= lambda x: x[1], reverse=True)
            with open(f"{working_directory}/logs/hparam_tuning_{start_time}/device_stats_{eval_device}.txt", 'a') as f:
              f.write(tabulate(tmp_list, headers=['Pos device', 'count']))
              f.write('\n')
            print(tabulate(tmp_list, headers=['Pos device', 'count']))
            simclr_utitlities.dds_pos_device_stats = {}

            tmp = simclr_utitlities.dds_neg_device_stats
            tmp_list = [[contrastive_training.all_devices[key], tmp[key]] for key in tmp.keys()]
            tmp_list.sort(key= lambda x: x[1], reverse=True)
            with open(f"{working_directory}/logs/hparam_tuning_{start_time}/device_stats_{eval_device}.txt", 'a') as f:
              f.write(tabulate(tmp_list, headers=['Neg device', 'count']))
              f.write('\n')
            print(tabulate(tmp_list, headers=['Neg device', 'count']))
            simclr_utitlities.dds_neg_device_stats = {}

          result_dict = {}
          for k, v in hparams.items():
            result_dict[k.name] = v
          result_dict[METRIC_F1_MACRO] = test_f1_macro
          result_dict[METRIC_F1_WEIGHTED] = test_f1_weighted

          # all_results.append(result_dict)
          
          with open(working_directory + '/logs/hparam_tuning_' + start_time + '/results_summary.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow(result_dict)

          # time.sleep(1)

          session_num += 1

    if apriori_device_selection: #resetting the values for next run 
      positive_devices = []
      negative_devices = []
      pos_neg_devices = '([], [])'
  
  return session_num

if __name__ == '__main__':

  all_gpu_devices = args.gpu_device.split(',')
  os.environ['PYTHONHASHSEED']='42'
  # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  tf.random.set_seed(42)
  np.random.seed(42)
  # tf_dataset_full = load_data.Data(args.dataset_path, args.dataset_name)
  # session_num = 0
  if not os.path.exists(working_directory + '/logs/hparam_tuning_' + start_time + '/'):
    os.mkdir(working_directory + '/logs/hparam_tuning_' + start_time + '/')
  with open(working_directory + '/logs/hparam_tuning_' + start_time + '/results_summary.csv', 'w', newline='') as csv_file:
      csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      csv_writer.writeheader()

  maximum_queue_size = 300
  num_process = args.n_process_per_gpu * len(all_gpu_devices)
  active_consumer_counter = num_process

  job_queue = mp.Queue()
  results_queue = mp.Queue(maximum_queue_size)
  processes = [mp.Process(target=consumer, args=(i, job_queue, results_queue, {"CUDA_VISIBLE_DEVICES": all_gpu_devices[i % len(all_gpu_devices)].strip() })) for i in range(num_process)]
  process_pids = []

  try:
    print("Putting Jobs...")
    for held_out_user in HP_HELD_OUT.domain.values:
      for lr_decay in HP_LR_DECAY.domain.values:
        for lr in HP_LR.domain.values:
          for optim in HP_OPTIM.domain.values:
            for take in HP_TAKE.domain.values:
              for arch in HP_ARCH.domain.values:
                for aug in HP_AUG.domain.values:
                  for temp in HP_TEMPERATURE.domain.values:
                    for pos_neg_devices in HP_POS_NEG_DEVICES.domain.values:
                      for multi_sampling_mode in HP_MULTI_SAMPLING.domain.values:
                        for dynamic_device_selection in HP_DYNAMIC_DEVICE_SELECTION.domain.values:
                          for device_selection_strategy in HP_DEVICE_SELECTION_STRATEGY.domain.values: 
                            for weighted_collossl in HP_WEIGHTED_COLLOSSL.domain.values:
                              for neg_sample_size in HP_NEG_SAMPLE_SIZE.domain.values:
                                run_name = "run-pretrain-%d" % int(datetime.datetime.now().timestamp() * 1000.0)

                                args.learning_rate_decay = lr_decay
                                args.learning_rate = lr
                                args.optimizer = optim
                                args.take = take
                                args.model_arch = arch
                                args.data_aug = aug
                                args.contrastive_temperature = temp
                                positive_devices, negative_devices = eval(pos_neg_devices)
                                args.pos_neg_devices = pos_neg_devices
                                args.positive_devices = positive_devices
                                args.negative_devices = negative_devices
                                args.run_name = run_name
                                args.start_time = start_time
                                args.held_out = held_out_user
                                args.multi_sampling_mode = multi_sampling_mode
                                args.dynamic_device_selection = dynamic_device_selection
                                args.device_selection_strategy = device_selection_strategy
                                args.weighted_collossl = weighted_collossl
                                args.neg_sample_size = neg_sample_size

                                if dynamic_device_selection==1 and (len(positive_devices)!=0 and len(negative_devices)!=0): #invalid configurations
                                  continue

                                job_queue.put({'args': copy.copy(args)})
                                time.sleep(0.01) # To avoid name collision

    print(f"{os.getpid()} Server - starting consumers")
    for p in processes:
      p.start()

    for _ in range(num_process):
      job_queue.put(None)
    print("finished putting jobs")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for sel_gpu in gpus:
    #             tf.config.experimental.set_memory_growth(sel_gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    with tf.summary.create_file_writer(working_directory + '/logs/hparam_tuning_' + start_time).as_default():
      hp.hparams_config(
          hparams=hparams,
          metrics=metrics,
      )
    # with open(working_directory + '/logs/hparam_tuning_' + start_time + '/results_summary.csv', 'w', newline='') as csv_file:
    #   csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #   csv_writer.writeheader()

    while(True):
      job_results = results_queue.get()
      print("Job results", job_results)
      if job_results is None:
        active_consumer_counter -= 1
        if active_consumer_counter == 0:
          break
      elif job_results['type'] == 'init':
        process_pids.append(job_results['data'])
      
    print('Closing workers')
    for p in processes:
      p.join()

  except KeyboardInterrupt:
    print("Interrupted from Keyboard")
  finally:
    print("Terminating Processes", processes)
    for p in processes:
      try:
        p.terminate()
      except Exception as e: 
        print(f"Unable to terminate process {p}, processes might still exist.", e)
    print("Killing Processes", process_pids)
    for pid in process_pids:
      try:
        # os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGKILL)
      except Exception as e:
        print(f"Unable to kill process {pid}, processes might still exist.", e)
    
    try:
      job_queue.close()
      results_queue.close()
    except Exception as e: 
      print("Unable to close job queues, processes might still be open", e)

    
    print('Finished')
    print(working_directory + '/logs/hparam_tuning_' + start_time + '/results_summary.csv')
    
