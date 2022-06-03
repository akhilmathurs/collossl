# %%
## Python libraries import

from device_selection import get_pos_neg_apriori 
import os
import pickle
import scipy
import datetime
import time
import argparse
import numpy as np
import tensorflow as tf
from common_parser import get_parser
# %%

import sys
import simclr_models
import simclr_utitlities
import transformations
import visual_utils

# %%
## Data loading script import
import load_data

## Loss function import
from loss_fn import *

def get_random_shuffle_indices(length, seed):
    rng = np.random.default_rng(seed=seed)
    index_list = np.arange(length, dtype=int)
    rng.shuffle(index_list)
    return index_list

def shuffle_array(array, seed, inplace=True):
    if inplace:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(array)
    else:
        length = array.shape[0]
        rng = np.random.default_rng(seed=seed)
        index_list = np.arange(length, dtype=int)
        rng.shuffle(index_list)
        return array[index_list]

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_group_held_out_users(all_users, group_index, num_groups):
    group_length = round(len(all_users) / num_groups)
    groups = [all_users[i * group_length: (i + 1) * group_length] if i < num_groups - 1 else all_users[i * group_length:] for i in range(num_groups)]
    return groups[group_index]


class BatchedRandomisedDataset:
    def __init__(self, data, batch_size, seed=42, randomised=True, axis=0, post_process_func=None, name=""):
        self.name = name
        self.data = data
        self.batch_size = batch_size
        self.axis = axis
        self.seed = seed
        self.data_len = data.shape[self.axis]
        self.num_batches = ceiling_division(self.data_len, batch_size)
        self.randomised = randomised
        self.rng = np.random.default_rng(seed=seed)
        self.post_process_func = post_process_func
        # if post_process_func is None:
        #     post_process_func = lambda x: x
        
    def reset_dataset(self):
        if self.randomised:
            index_list = np.arange(self.data_len, dtype=int)
            self.rng.shuffle(index_list)
            self.shuffled_dataset = self.data[index_list]
            self.output_dataset = self.shuffled_dataset
        else:
            self.output_dataset = self.data
        if not self.axis == 0:
            self.output_dataset = np.moveaxis(self.output_dataset, self.axis, 0)
        self.i = 0

    def __len__(self):
        return self.num_batches
    
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     if self.i < self.num_batches:
    #         i = self.i
    #         self.i += 1
    #         if self.axis == 0:
    #             return self.post_process_func(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size])
    #         else:
    #             return self.post_process_func(np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis + 1))
    #     else:
    #         raise StopIteration

    def __iter__(self):
        self.reset_dataset()

        def gen():
            if self.post_process_func is None:
                if self.axis == 0:
                    for i in range(self.num_batches):
                        yield self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size]
                else:
                    for i in range(self.num_batches):
                        yield np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis)
            else:
                if self.axis == 0:
                    for i in range(self.num_batches):
                        yield self.post_process_func(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size])
                else:
                    for i in range(self.num_batches):
                        yield self.post_process_func(np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis))
        return gen()

class SequenceEagerZippedDataset(tf.keras.utils.Sequence):
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis
        self.reset_dataset()

    def reset_dataset(self):
        if self.stack_batches:
            self.data = [
                np.stack(zipped_batch, axis=self.stack_axis) for zipped_batch in zip(*tuple(self.datasets))
            ]
        else:
            self.data = [
                zipped_batch for zipped_batch in zip(*tuple(self.datasets))
            ]

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def on_epoch_end(self):
        self.reset_dataset()

class ZippedDataset:
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    # def reset_dataset(self):
    #     self.iters = [iter(dataset) for dataset in self.datasets]
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     batch = [next(it) for it in self.iters]
    #     if self.stack_batches:
    #         return np.stack(batch, axis=self.stack_axis)
    #     else:
    #         return tuple(batch)

    def __iter__(self):
        def gen():
            if self.stack_batches:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield np.stack(zipped_batch, axis=self.stack_axis)
            else:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield zipped_batch
        return gen()

class ConcatenatedDataset:
    def __init__(self, batched_randomised_datasets):
        self.datasets = batched_randomised_datasets
        self.num_datasets = len(self.datasets)

    # def reset_dataset(self):
    #     self.iters = [iter(dataset) for dataset in self.datasets]
    #     self.iter_i = 0
    
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     batch = None
    #     while (batch is None):
    #         try:
    #             batch = next(self.iters[self.iter_i])
    #         except StopIteration:
    #             batch = None
    #             self.iter_i += 1
    #             if self.iter_i == self.num_datasets:
    #                 raise StopIteration
    #     return batch
    
    def __iter__(self):
        def gen():
            for dataset in self.datasets:
                for batch in dataset:
                    yield batch
        return gen()

all_devices, positive_indices, negative_indices = [], [], []
def train(dataset_full, args):

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)
    
    input_shape = dataset_full.input_shape

    # tf_train_full = tf_dataset_full.ds_train[args.train_device].map(lambda x, y, i: (x, y))

    # output_shape = len(np.unique([y for x, y in tf_train_full])) # Infer number of classes from training data (slow)
    output_shape = len(dataset_full.info['session_list'])
    print("input shape", input_shape)
    print("output shape", output_shape)


    # %%
    ## Setup working folder

    # working_directory = args.working_directory if args.working_directory.endswith("/") else args.working_directory + "/"
    # if not os.path.exists(working_directory):
    #     os.mkdir(working_directory)
    # start_time = datetime.datetime.now()
    # start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)
    if not os.path.exists(working_directory):
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)


    if not hasattr(args, 'start_time'):
        args.start_time = str(int(datetime.datetime.now().timestamp()))
    if not hasattr(args, 'run_name'):
        args.run_name = f"run-{args.start_time}"

    ## Model Architecture

    if args.trained_model_path is None:
        if args.model_arch == '1d_conv':
            base_model, last_freeze_layer = simclr_models.create_base_model(input_shape, model_name="1d_conv")
    else:
        base_model = tf.keras.models.load_model(args.trained_model_path)
        last_freeze_layer = args.trained_model_last_freeze_layer

# %%
## Training hyperparameters

    if args.training_mode != 'none':
        batch_size = args.training_batch_size
        decay_steps = args.training_decay_steps
        epochs = args.training_epochs
        temperature = args.contrastive_temperature
        initial_lr = args.learning_rate

    # %%

    # %%
        ## Prepare for training (creation of learning rate decay, optimizer, neural network)

        tf.keras.backend.set_floatx('float32')

        if args.learning_rate_decay == 'cosine':
            lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        elif args.learning_rate_decay == 'none':
            lr_decayed_fn = initial_lr

        if args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_decayed_fn)
        elif args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)


        if args.data_aug == 'none':
            transformation_function = lambda x: x
        elif args.data_aug == 'rotate':
            transform_func_structs = [
                ([0,1,2], transformations.rotation_transform_vectorized),
                ([3,4,5], transformations.rotation_transform_vectorized)
            ]
            transformation_function = simclr_utitlities.generate_slicing_transform_function(
                transform_func_structs, slicing_axis=2, concatenate_axis=2
            )
        elif args.data_aug == 'sensor_noise':
            transformation_function = lambda x: transformations.scaling_transform_vectorized(transformations.noise_transform_vectorized(x))
        

    # %%
        if args.training_mode == 'multi':

            simclr_model = base_model
            simclr_model.summary()
            ## Multi-device Training
            print('='*20+"Multi-device Training"+'='*20)

            # Design notes:
            ## The training loop is expected to function as follows:
            ## It samples n samples from each device (n = batch size)
            ## Each of these n samples are passed through the model
            ## And then the loss function is called, which should accept the anchor embeddings (n x e), positive embeddings (p x n x e) and negative embeddings (q x n x e)
            ## p = number of positive devices, q = number of negative devices
            ## The model accepts a (n x t x c) windowed time series, and outputs embeddings in shape (n x e)
            ## To allow for better generalizability, the sampling should be agnostic to the positive/negative assignments (all devices are sampled at the same time)
            ## And when obtaining the loss, it should allow for different assignments by concatenating different device embeddings based on arguments passed as positive/negative indices

            # Index mappings
            # all_devices = ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin']
            global all_devices, positive_indices, negative_indices
            all_devices = [args.train_device]
            all_devices.extend(args.positive_devices)
            all_devices.extend(args.negative_devices)
            print("Anchor:", args.train_device, "Positives:", args.positive_devices, "Negatives:", args.negative_devices)
            if len(args.positive_devices)==0 or len(args.negative_devices)==0: # device_selection will be called
                all_devices = list(dataset_full.info['device_list'])
                all_devices.remove(args.train_device)
                all_devices = [args.train_device] + all_devices
                
            anchor_index = 0
            positive_indices = np.arange(len(args.positive_devices)) + 1
            negative_indices = np.arange(len(args.negative_devices)) + 1 + len(args.positive_devices)

            training_set_device = [(np.concatenate([dataset_full.device_user_ds[d][u][0] for u in dataset_full.device_user_ds[d] if u not in held_out_users], axis=0)) for d in all_devices]
            # training_full_size = len(training_set_device[0])
            # train_samples_count = int(training_full_size * args.take)

            if args.dynamic_device_selection==0 and (len(args.positive_devices)==0 or len(args.negative_devices)==0):
                training_set_stacked = np.stack(training_set_device, axis=0)
                tf_train_contrast = BatchedRandomisedDataset(training_set_stacked, batch_size, randomised=False, axis=1, name="distances")
                positive_indices, negative_indices, distances = get_pos_neg_apriori(tf_train_contrast, all_devices, strategy=args.device_selection_strategy)
            else:
                distances = None

            if args.multi_sampling_mode == 'sync_all':
                # if distances is not None:
                #     distances = tf.convert_to_tensor(distances, dtype=tf.float64)
                pass
            else:
                # Unsynchronised sampling for different groupings of devices     
                if distances is not None:
                    overlap_indices = list(set.intersection(set(positive_indices), set(negative_indices)))
                    overlap_devices = [all_devices[i] for i in overlap_indices] 
                    all_device_length = len(all_devices)

                    #remove the overlap indices from their original order, to prevent them from becoming positive_devices/aligned
                    negative_indices = [x for x in negative_indices if x not in set(overlap_indices)]
                    negative_devices = [all_devices[i] for i in negative_indices]

                    #Allow for duplicate anchor dataset for negative sampling
                    #Add anchor as a negative device at the end  
                    negative_indices = negative_indices + [*range(all_device_length, all_device_length + len(overlap_indices) +  1, 1)]                    
                    all_devices = all_devices + [args.train_device] + overlap_devices

                    distances[all_device_length] = (min(distances.values())/2.0) 
                    for i,d in enumerate(overlap_indices):
                        distances[all_device_length+i+1] = (distances[d]) #add the distance of the  overlapped element at the end of the list
                        # del distances[d-1] #remove the overlapped element from its original place
                    
                    # distances = tf.convert_to_tensor(distances, dtype=tf.float64)
                            
            user_device_dataset = []
            for u in dataset_full.info['user_list']:
                dataset_per_user = []
                if u not in held_out_users:
                # if args.held_out is None or u != dataset_full.info['user_list'][args.held_out]:
                    for d in all_devices:
                        X = transformation_function(dataset_full.device_user_ds[d][u][0])
                        len_x = X.shape[0]
                        X_shuffled = shuffle_array(X, seed=42, inplace=False) 
                        dataset_per_user.append(X_shuffled[: int(len_x * args.take)])
                    user_device_dataset.append(dataset_per_user)

            tf_train_contrast_list = []
            for user_dataset in user_device_dataset:
                device_dataset_shuffled = []
                for device_index, device_dataset in enumerate(user_dataset):
                                            
                    if args.multi_sampling_mode == 'sync_all':
                        seed = 42
                    elif args.multi_sampling_mode == 'unsync_neg':
                        if device_index == anchor_index:
                            seed = 42
                        elif device_index in positive_indices:
                            seed = 42
                        else: 
                            seed = 43 + device_index
                            
                    elif args.multi_sampling_mode == 'unsync_all':
                        seed = 42 + device_index

                    #For Dynamic Device Selection, we want the datasets to be synced here. They will be shuffled later
                    if args.dynamic_device_selection==1 and args.multi_sampling_mode != 'unsync_all':  
                        seed = 42

                    shuffled = BatchedRandomisedDataset(device_dataset, batch_size=batch_size, seed=seed)
                    device_dataset_shuffled.append(shuffled)

                tf_train_contrast_list.append(ZippedDataset(device_dataset_shuffled, stack_batches=True))

            tf_train_concat = ConcatenatedDataset(tf_train_contrast_list)


            weighted_loss_function = lambda a_e, p_e, p_w, n_e, n_w: weighted_group_contrastive_loss_with_temp(a_e, p_e, p_w, n_e, n_w, temperature=temperature)

            # if args.dynamic_device_selection==0 and (len(args.positive_devices)==0 or len(args.negative_devices)==0):
            #     positive_indices, negative_indices = get_pos_neg_apriori(tf_train_contrast, all_devices)

            index_mappings = (anchor_index, positive_indices, negative_indices)
            if not os.path.exists(f"{working_directory}/models/{args.run_name}.hdf5"):
                trained_model_save_path = f"{working_directory}/models/{args.run_name}.hdf5"
                trained_model_low_loss_save_path = f"{working_directory}/models/{args.run_name}_lowest.hdf5"
                trained_model, epoch_losses = simclr_utitlities.group_supervised_contrastive_train_model(simclr_model, tf_train_concat, transformation_function, optimizer, index_mappings, distances, weighted_loss_function, args.device_selection_strategy, args, weighted=args.weighted_collossl, epochs=epochs, verbose=1, training=True, temporary_save_model_path=trained_model_low_loss_save_path)
                trained_model.save(trained_model_save_path)
                trained_model_save_path = trained_model_low_loss_save_path

            else:
                trained_model_save_path = f"{working_directory}/models/{args.run_name}.hdf5"

        elif args.training_mode == 'supervised':
            
            supervised_model = simclr_models.create_full_classification_model_from_base_model(base_model, output_shape, optimizer=optimizer, model_name="TPN", intermediate_layer=-1, last_freeze_layer=-1)
            
            full_model_save_path = f"{working_directory}/models/{args.run_name}_full.hdf5"
            best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_model_save_path,
                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
            )


            if args.baseline=='supervised_all_devices':
                all_devices = list(dataset_full.info['device_list'])
                training_set_device_X = np.concatenate([dataset_full.device_user_ds[d][u][0] for d in all_devices for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
                training_set_device_y = np.concatenate([dataset_full.device_user_ds[d][u][1] for d in all_devices for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
            else:
                training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.train_device][u][0] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
                training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.train_device][u][1] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)

            training_full_size = training_set_device_y.shape[0]
            train_samples_count = int(training_full_size * args.take)
            shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
            training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
            training_set_device_y = tf.keras.utils.to_categorical(training_set_device_y[shuffle_indices][:train_samples_count], num_classes=output_shape)

            train_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[int(train_samples_count * 0.2):], batch_size, post_process_func=transformation_function, seed=42), BatchedRandomisedDataset(training_set_device_y[int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
            val_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[:int(train_samples_count * 0.2)], batch_size, seed=42), BatchedRandomisedDataset(training_set_device_y[:int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
            
            callbacks = [best_model_callback]

            
            supervised_model.fit(
                x = train_split,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=val_split
                # verbose=0
            )

            print(full_model_save_path)
            best_supervised_model = tf.keras.models.load_model(full_model_save_path)

            feature_extractor_model = simclr_models.extract_intermediate_model_from_base_model(best_supervised_model, intermediate_layer=7)
            trained_model_save_path = f"{working_directory}/models/{args.run_name}.hdf5"
            feature_extractor_model.save(trained_model_save_path)

            if args.eval_mode == 'base_model':
                trained_model_save_path = trained_model_save_path
            elif args.eval_mode == 'full_model':
                trained_model_save_path = full_model_save_path

    return trained_model_save_path

    

def fine_tune_evaluate(dataset_full, trained_model_save_path, args):

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)


    input_shape = dataset_full.input_shape
    
    output_shape = len(dataset_full.info['session_list'])
    working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)

    if args.trained_model_path is None:
        if args.model_arch == '1d_conv':
            _, last_freeze_layer = simclr_models.create_base_model(input_shape, model_name="1d_conv")
    else:
        # base_model = tf.keras.models.load_model(args.trained_model_path)
        last_freeze_layer = args.trained_model_last_freeze_layer

    if args.eval_mode != 'none':

        if args.training_mode != 'none':
            eval_model_path = trained_model_save_path
        else:
            eval_model_path = args.trained_model_path
        
        # %%
        ## Prepare for evaluations


        # dataset_full.device_user_test
        np_test_x = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][0] for u in sorted(dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
        np_test_y = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][1] for u in sorted(dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
        # np_test_x = np.concatenate([dataset_full.device_user_train[args.eval_device][u][0] for u in sorted(dataset_full.device_user_train[args.eval_device].keys()) if (args.held_out is None or u == dataset_full.info['user_list'][args.held_out])], axis=0)
        # np_test_y = np.concatenate([dataset_full.device_user_train[args.eval_device][u][1] for u in sorted(dataset_full.device_user_train[args.eval_device].keys()) if (args.held_out is None or u == dataset_full.info['user_list'][args.held_out])], axis=0)
        np_test_y = tf.keras.utils.to_categorical(np_test_y, num_classes=output_shape)
        np_test = (np_test_x, np_test_y)

        if args.eval_mode == 'base_model':

            total_epochs = args.fine_tune_epochs
            batch_size = args.fine_tune_batch_size


            training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][0] for u in sorted(dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)
            training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][1] for u in sorted(dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)

            training_full_size = training_set_device_y.shape[0]
            train_samples_count = int(training_full_size * args.fine_tune_take)
            shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
            training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
            training_set_device_y = tf.keras.utils.to_categorical(training_set_device_y[shuffle_indices][:train_samples_count], num_classes=output_shape)

            train_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[int(train_samples_count * 0.2):], batch_size, seed=42), BatchedRandomisedDataset(training_set_device_y[int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
            val_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[:int(train_samples_count * 0.2)], batch_size, seed=42), BatchedRandomisedDataset(training_set_device_y[:int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)

            # %%
            ## Full HAR Model

            tag = "full_eval"

            eval_model = tf.keras.models.load_model(eval_model_path)
            # lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=1e-4, decay_steps=1000)
            optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-4)
            full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(eval_model, output_shape, optimizer_fine_tune, model_name="TPN", intermediate_layer=-1, last_freeze_layer=last_freeze_layer)

            full_eval_best_model_file_name = f"{working_directory}/models/{args.run_name}_{tag}.hdf5"
            best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
            )

            training_history = full_evaluation_model.fit(
                x = train_split,
                epochs=total_epochs,
                callbacks=[best_model_callback],
                validation_data=val_split
            )

            full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name)

            results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
            results_last_epoch = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True)

            with open(f"{working_directory}/results/{args.run_name}.txt", 'a') as f:
                f.write("====== Full Evaluation ======\n")
                f.write("Model with lowest validation Loss:\n")
                f.write(str(results_lowest_loss) + "\n")
                f.write("Model in last epoch:\n")
                f.write(str(results_last_epoch) + "\n")

            print("Model with lowest validation Loss:")
            print(results_lowest_loss)
            print("Model in last epoch")
            print(results_last_epoch)


            # %% t-SNE
            eval_model = tf.keras.models.load_model(eval_model_path)
            tsne_image =  None

            if args.output_tsne:
                embeddings = eval_model.predict(np_test[0], batch_size=600)
                tsne_projections = visual_utils.fit_transform_tsne(embeddings)
                tsne_figure = visual_utils.plot_tsne(tsne_projections, np_test[1], label_name_list=dataset_full.info['session_list'])
                tsne_image = visual_utils.plot_to_image(tsne_figure)

            if results_lowest_loss['F1 Macro'] >= results_last_epoch['F1 Macro']:
                return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted'], tsne_image
            else:
                return results_last_epoch['F1 Macro'], results_last_epoch['F1 Weighted'], tsne_image

        elif args.eval_mode == 'full_model':  # This is to be run only with supervised setting
            assert args.training_mode == 'supervised', "args.eval_model = full_model can only be run with supervised training, use args.eval_model = base_model"
            full_eval_best_model = tf.keras.models.load_model(eval_model_path)

            results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)

            with open(f"{working_directory}/results/{args.run_name}.txt", 'a') as f:
                f.write("\n====== Args ======\n")
                f.write(str(args) + "\n")
                f.write("====== Full Evaluation ======\n")
                f.write(str(results_lowest_loss) + "\n")

            print("Results:")
            print(results_lowest_loss)

            # %% t-SNE
            tsne_image = None

            if args.output_tsne:
                feature_extractor_model = simclr_models.extract_intermediate_model_from_base_model(full_eval_best_model, intermediate_layer=7)
                embeddings = feature_extractor_model.predict(np_test[0], batch_size=600)
                tsne_projections = visual_utils.fit_transform_tsne(embeddings)
                tsne_figure = visual_utils.plot_tsne(tsne_projections, np_test[1], label_name_list=dataset_full.info['session_list'])
                tsne_image = visual_utils.plot_to_image(tsne_figure)

            return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted'], tsne_image

if __name__ == '__main__':
    
    parser = get_parser()

    ## Prepare full dataset
    args = parser.parse_args()

    if args.eval_device is None:
        args.eval_device = args.train_device
    if args.fine_tune_device is None:
        args.fine_tune_device = args.train_device
    print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    os.environ['PYTHONHASHSEED']='42'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.random.set_seed(42)
    np.random.seed(42)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
        except RuntimeError as e:
            print(e)

    dataset_full = load_data.Data(args.dataset_path, args.dataset_name, load_path=None, held_out=args.held_out)

    # train_and_evaluate(tf_dataset_full, args)
    trained_model_save_path = train(dataset_full, args)
    if not hasattr(args, 'start_time'):
        args.start_time = str(int(datetime.datetime.now().timestamp()))
    if not hasattr(args, 'run_name'):
        args.run_name = f"run-{args.start_time}"
    fine_tune_evaluate(dataset_full, trained_model_save_path, args)
