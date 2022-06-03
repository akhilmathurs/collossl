import os
import pickle
import scipy
import datetime
import time
import argparse
import numpy as np
import tensorflow as tf
import gc

from common_parser import get_parser
import load_data
import transformations
from contrastive_training import BatchedRandomisedDataset, SequenceEagerZippedDataset, get_random_shuffle_indices, shuffle_array, fine_tune_evaluate, get_group_held_out_users
import simclr_models
import simclr_utitlities

import scipy.stats
import sklearn.tree
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.ensemble

def create_individual_transform_dataset(X, transform_funcs, other_labels=None, multiple=1, is_transform_func_vectorized=True, verbose=1):
    label_depth = len(transform_funcs)
    transform_x = []
    transform_y = []
    other_y = []
    if is_transform_func_vectorized:
        for _ in range(multiple):
            
            transform_x.append(X)
            ys = np.zeros((len(X), label_depth), dtype=int)
            transform_y.append(ys)
            if other_labels is not None:
                other_y.append(other_labels)

            for i, transform_func in enumerate(transform_funcs):
                if verbose > 0:
                    print(f"Using transformation {i} {transform_func}")
                transform_x.append(transform_func(X))
                ys = np.zeros((len(X), label_depth), dtype=int)
                ys[:, i] = 1
                transform_y.append(ys)
                if other_labels is not None:
                    other_y.append(other_labels)
        if other_labels is not None:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), np.concatenate(other_y, axis=0)
        else:
            return np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0), 
    else:
        for _ in range(multiple):
            for i, sample in enumerate(X):
                if verbose > 0 and i % 1000 == 0:
                    print(f"Processing sample {i}")
                    gc.collect()
                y = np.zeros(label_depth, dtype=int)
                transform_x.append(sample)
                transform_y.append(y)
                if other_labels is not None:
                    other_y.append(other_labels[i])
                for j, transform_func in enumerate(transform_funcs):
                    y = np.zeros(label_depth, dtype=int)
                    # transform_x.append(sample)
                    # transform_y.append(y.copy())

                    y[j] = 1
                    transform_x.append(transform_func(sample))
                    transform_y.append(y)
                    if other_labels is not None:
                        other_y.append(other_labels[i])
        if other_labels is not None:
            np.stack(transform_x), np.stack(transform_y), np.stack(other_y)
        else:
            return np.stack(transform_x), np.stack(transform_y)

def map_multitask_y(y, output_tasks):
    multitask_y = {}
    for i, task in enumerate(output_tasks):
        multitask_y[task] = y[:, i]
    return multitask_y

def multitask_train_test_split(dataset, test_size=0.1, random_seed=42):
    dataset_size = len(dataset[0])
    indices = np.arange(dataset_size)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    test_dataset_size = int(dataset_size * test_size)
    return dataset[0][indices[test_dataset_size:]], dict([(k, v[indices[test_dataset_size:]]) for k, v in dataset[1].items()]), dataset[0][indices[:test_dataset_size]], dict([(k, v[indices[:test_dataset_size]]) for k, v in dataset[1].items()])


def attach_multitask_transform_head(core_model, output_tasks, optimizer, with_har_head=False, har_output_shape=None, num_units_har=1024, model_name="multitask_transform"):
    """
    Note: core_model is also modified after training this model (i.e. the weights are updated)
    """
    inputs = tf.keras.Input(shape=core_model.input.shape[1:], name='input')
    intermediate_x = core_model(inputs)
    outputs = []
    losses = [tf.keras.losses.BinaryCrossentropy() for _ in output_tasks]
    for task in output_tasks:
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_'+task)(intermediate_x)
        pred = tf.keras.layers.Dense(1, activation='sigmoid', name=task)(x)
        outputs.append(pred)


    if with_har_head:
        x = tf.keras.layers.Dense(num_units_har, activation='relu')(intermediate_x)
        x = tf.keras.layers.Dense(har_output_shape)(x)
        har_pred = tf.keras.layers.Softmax(name='har')(x)

        outputs.append(har_pred)
        losses.append(tf.keras.losses.CategoricalCrossentropy())

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['accuracy']
    )
    
    return model

def extract_core_model(composite_model):
    return composite_model.layers[1]

def train_multi_task_transform(dataset_full, args):

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)

    # Setup
    input_shape = dataset_full.input_shape
    output_shape = len(dataset_full.info['session_list'])
    print("input shape", input_shape)
    print("output shape", output_shape)


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

    if args.trained_model_path is None:
        if args.model_arch == '1d_conv':
            base_model, last_freeze_layer = simclr_models.create_base_model(input_shape, model_name="1d_conv")
    else:
        base_model = tf.keras.models.load_model(args.trained_model_path)
        last_freeze_layer = args.trained_model_last_freeze_layer

    batch_size = args.training_batch_size
    decay_steps = args.training_decay_steps
    epochs = args.training_epochs
    temperature = args.contrastive_temperature
    initial_lr = args.learning_rate

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

    # Dataset Preparation
    def six_d_separate_rotation(X):
        return np.concatenate([transformations.rotation_transform_vectorized(X[:, :, :3]), transformations.rotation_transform_vectorized(X[:, :, 3:])], axis=2)

    transform_funcs_vectorized = [
        transformations.noise_transform_vectorized, 
        transformations.scaling_transform_vectorized, 
        # transformations.rotation_transform_vectorized, 
        six_d_separate_rotation,
        transformations.negate_transform_vectorized, 
        transformations.time_flip_transform_vectorized, 
        transformations.time_segment_permutation_transform_improved, 
        transformations.time_warp_transform_low_cost, 
        transformations.channel_shuffle_transform_vectorized
    ]
    transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']

    transform_model = attach_multitask_transform_head(base_model, output_tasks=transform_funcs_names, optimizer=optimizer)
    transform_model.summary()

    training_set_device = np.concatenate([dataset_full.device_user_ds[args.train_device][u][0] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
    shuffle_array(training_set_device, seed=42) 
    training_full_size = training_set_device.shape[0]
    train_samples_count = int(training_full_size * args.take)
    training_set_device_partial = training_set_device[:train_samples_count]

    multitask_transform_dataset = create_individual_transform_dataset(training_set_device_partial, transform_funcs_vectorized)

    multitask_transform_train = (multitask_transform_dataset[0], map_multitask_y(multitask_transform_dataset[1], transform_funcs_names))
    multitask_split = multitask_train_test_split(multitask_transform_train, test_size=0.10, random_seed=42)
    multitask_train = (multitask_split[0], multitask_split[1])
    multitask_val = (multitask_split[2], multitask_split[3])

    # Training
    full_model_save_path = f"{working_directory}/models/{args.run_name}_multi_task.hdf5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_model_save_path,
        monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
    )
    callbacks = [best_model_callback]

    transform_model.fit(
        x = multitask_train[0],
        y = multitask_train[1],
        epochs=epochs,
        callbacks=callbacks,
        validation_data=multitask_val,
        batch_size=batch_size,
        verbose=1
    )

    print(full_model_save_path)
    best_supervised_model = tf.keras.models.load_model(full_model_save_path)

    feature_extractor_model = extract_core_model(best_supervised_model)
    trained_model_save_path = f"{working_directory}/models/{args.run_name}.hdf5"
    feature_extractor_model.save(trained_model_save_path)

    return trained_model_save_path

# ----------------------------

# En-Co-Training
def get_norm(window):
    return np.linalg.norm(window, axis=1)

def get_mean(window):
    return np.mean(window, axis=0)

def get_correlation(window):
    corr_matrix = np.corrcoef(window, rowvar=False)
    corr_matrix = np.where(np.isnan(corr_matrix), 0, corr_matrix)
    return np.array([corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]])

def get_interquartile_range(window):
    return scipy.stats.iqr(window, axis=0)

def get_mean_absolute_deviation(window):
    mean = np.mean(window, axis=0)
    deviation = np.absolute(window - mean)
    mad = np.mean(deviation, axis=0)
    return mad

def get_root_mean_square(window):
    square = np.square(window)
    mean = np.mean(square, axis=0)
    return np.sqrt(mean)

def get_std_deviation(window):
    return np.std(window, axis=0)

def get_variance(window):
    return np.var(window, axis=0)

def get_spectral_energy_no_dc(window):
    no_dc = window - np.mean(window, axis=0)
    fft_coef = np.fft.fft(no_dc, axis=0)
    return np.mean(np.square(np.absolute(fft_coef)), axis=0)

def extract_dataset_features(X, extraction_functions, pre_process_function):
    X_features = []
    X_scaled = pre_process_function(X)
    for sample in X_scaled:
        sample_features = []
        for func in extraction_functions:
            sample_features.append(func(sample))
        X_features.append(np.concatenate(sample_features))
    return np.stack(X_features)

def scale_pre_process(dataset, unit_conversion):
    return dataset * unit_conversion / scipy.constants.g



def train_and_evaluate_en_co_training(dataset_full, args):

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)


    # Setup
    input_shape = dataset_full.input_shape
    output_shape = len(dataset_full.info['session_list'])
    print("input shape", input_shape)
    print("output shape", output_shape)


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

    
    extraction_functions = [
        get_mean,
        get_correlation,
        get_interquartile_range,
        get_mean_absolute_deviation,
        get_root_mean_square,
        get_std_deviation,
        get_variance,
        get_spectral_energy_no_dc
    ]

    # Dataset Preparation
    training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.train_device][u][0] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
    training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.train_device][u][1] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)

    training_full_size = training_set_device_y.shape[0]
    train_samples_count = int(training_full_size * args.fine_tune_take)
    shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
    training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
    training_set_device_y = training_set_device_y[shuffle_indices][:train_samples_count]

    split = 0.5
    train_x = training_set_device_X[:int(len(training_set_device_X) * split)]
    train_y = training_set_device_y[:int(len(training_set_device_y) * split)]

    unlabelled_x = training_set_device_X[int(len(training_set_device_X) * split):]

    test_x = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][0] for u in sorted(dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
    test_y = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][1] for u in sorted(dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)

    train_x_features = extract_dataset_features(train_x, extraction_functions, lambda x: x)
    test_x_features = extract_dataset_features(test_x, extraction_functions, lambda x: x)
    unlabelled_x_features = extract_dataset_features(unlabelled_x, extraction_functions, lambda x: x)

    iterations = 20
    pool_size = len(unlabelled_x_features) // 10
    sample_extract_per_class = int(pool_size * 0.15 / output_shape)


    

    current_training_set_x = train_x_features
    current_training_set_y = train_y
    current_unlabelled_set_x = unlabelled_x_features

    evaluation_results = []
    models = []

    unlabelled_pool = None
    for i in range(iterations):
        # Pool Replenish
        np.random.seed(42 + i)
        np.random.shuffle(current_unlabelled_set_x)
        if unlabelled_pool is None:
            num_samples_extraction = pool_size
            unlabelled_pool = current_unlabelled_set_x[:num_samples_extraction]
        else:
            num_samples_extraction = pool_size - len(unlabelled_pool)
            unlabelled_pool = np.concatenate([unlabelled_pool, current_unlabelled_set_x[:num_samples_extraction]], axis=0)
        
        current_unlabelled_set_x = current_unlabelled_set_x[num_samples_extraction:]


        # Classifiers
        clfs = [
            ('decision_tree', sklearn.tree.DecisionTreeClassifier()),
            ('gaussian_nb', sklearn.naive_bayes.GaussianNB()),
            ('knn', sklearn.neighbors.KNeighborsClassifier(3))
        ]

        ensemble_clf = sklearn.ensemble.VotingClassifier(estimators=clfs, voting='soft')
        
        # Training
        ensemble_clf.fit(current_training_set_x, current_training_set_y)
        pred = ensemble_clf.predict(test_x_features)

        
        results = simclr_utitlities.evaluate_model_simple(pred, test_y, is_one_hot=False)
        evaluation_results.append(results)
        models.append(ensemble_clf)

        
        print(results)

        # Individual prediction
        predictions = []
        for name, _ in clfs:
            clf = ensemble_clf.named_estimators_[name]
            predictions.append(clf.predict(unlabelled_pool))

        predictions = np.array(predictions)

        # Agreement checking, check if all clfs agree
        equality_check = predictions[0]
        for prediction_row in predictions[1:]:
            equality_check = np.where(equality_check == prediction_row, equality_check, -1)

        starting_class = 0

        if np.sum(equality_check >= starting_class) > 0:
            # obtain indexes of agreed samples
            all_class_extraction_indexes = []
            
            for c in range(starting_class, output_shape):
                
                match_class_indexes = np.argwhere(equality_check == c)[:, 0]
                
                if len(match_class_indexes) > 0:
                    extraction_indexes = match_class_indexes[:sample_extract_per_class]
                    all_class_extraction_indexes.append(extraction_indexes)
                    # print(c, len(extraction_indexes))
                else:
                    # print(c, 0)
                    pass
            all_class_extraction_indexes = np.concatenate(all_class_extraction_indexes)
            
            # extract samples by indexes
            all_extractions_x = np.take(unlabelled_pool, all_class_extraction_indexes, axis=0)
            all_extractions_y = np.take(equality_check, all_class_extraction_indexes)
            
            # add to training set
            current_training_set_x = np.concatenate([current_training_set_x, all_extractions_x], axis=0)
            current_training_set_y = np.concatenate([current_training_set_y, all_extractions_y])

            # remove from pool
            unlabelled_pool = np.delete(unlabelled_pool, all_class_extraction_indexes, axis=0)
            
            print(f'{i} - training_x: {current_training_set_x.shape}, remaining_unlabelled:{current_unlabelled_set_x.shape}, pool: {unlabelled_pool.shape}, extraction: {len(all_class_extraction_indexes)}')
        else:
            # reset pool
            unlabelled_pool = None
            print(f'{i}!- training_x: {current_training_set_x.shape}, remaining_unlabelled:{current_unlabelled_set_x.shape}, pool: None, extraction: 0')
    
    # Training
    full_model_save_path = f"{working_directory}/models/{args.run_name}_enco.pkl"
    with open(full_model_save_path, 'wb') as f:
        pickle.dump(models[-1], f)

    return evaluation_results[-1]['F1 Macro'], evaluation_results[-1]['F1 Weighted']


def Random(dataset_full, args):
    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(
            dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)

    input_shape = dataset_full.input_shape
    output_shape = len(dataset_full.info['session_list'])
    print("input shape", input_shape)
    print("output shape", output_shape)

    working_directory = os.path.join(
        args.working_directory, args.train_device, args.exp_name, args.training_mode)
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
            base_model, last_freeze_layer = simclr_models.create_base_model(
                input_shape, model_name="1d_conv")
    else:
        base_model = tf.keras.models.load_model(args.trained_model_path)
        last_freeze_layer = args.trained_model_last_freeze_layer

    if args.model_arch == '1d_conv':
        base_model, last_freeze_layer = simclr_models.create_base_model(
            input_shape, model_name="1d_conv")

    np_test_x = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][0] for u in sorted(
        dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
    np_test_y = np.concatenate([dataset_full.device_user_ds[args.eval_device][u][1] for u in sorted(
        dataset_full.device_user_ds[args.eval_device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
    if 0 not in np.unique(np_test_y):  # for PAMAP2 dataset
        np_test_y = np_test_y-1
    np_test_y = tf.keras.utils.to_categorical(
        np_test_y, num_classes=output_shape)
    np_test = (np_test_x, np_test_y)

    total_epochs = args.fine_tune_epochs
    batch_size = args.fine_tune_batch_size

    tag = "linear_eval"
    eval_model = base_model
    linear_evaluation_model = simclr_models.create_linear_model_from_base_model(
        eval_model, output_shape)

    linear_eval_best_model_file_name = f"{working_directory}/models/{args.run_name}_{tag}.hdf5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
                                                             monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
                                                             )

    training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][0] for u in sorted(
        dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)
    training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][1] for u in sorted(
        dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)

    training_full_size = training_set_device_y.shape[0]
    train_samples_count = int(training_full_size * args.fine_tune_take)
    shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
    training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
    training_set_device_y = tf.keras.utils.to_categorical(
        training_set_device_y[shuffle_indices][:train_samples_count], num_classes=output_shape)

    train_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[int(train_samples_count * 0.2):], batch_size,
                                             seed=42), BatchedRandomisedDataset(training_set_device_y[int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
    val_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[:int(train_samples_count * 0.2)], batch_size, seed=42),
                                           BatchedRandomisedDataset(training_set_device_y[:int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)

    callbacks = [best_model_callback]
    # %%
    ## Linear Evaluation
    """
    training_history = linear_evaluation_model.fit(
        x=train_split,
        epochs=total_epochs,
        callbacks=callbacks,
        validation_data=val_split
        # verbose=0
    )

    linear_eval_best_model = tf.keras.models.load_model(
        linear_eval_best_model_file_name)

    results_lowest_loss = simclr_utitlities.evaluate_model_simple(
        linear_eval_best_model.predict(np_test[0]), np_test[1], is_one_hot=True, return_dict=True)
    results_last_epoch = simclr_utitlities.evaluate_model_simple(
        linear_evaluation_model.predict(np_test[0]), np_test[1], is_one_hot=True, return_dict=True)

    with open(f"{working_directory}/results/{args.run_name}.txt", 'w') as f:
        f.write("====== Args ======\n")
        f.write(str(args) + "\n")
        f.write("====== Linear Evaluation ======\n")
        f.write("Model with lowest validation Loss:\n")
        f.write(str(results_lowest_loss) + "\n")
        f.write("Model in last epoch:\n")
        f.write(str(results_last_epoch) + "\n")
        f.write("\n")

    print("Model with lowest validation Loss:")
    print(results_lowest_loss)
    print("Model in last epoch")
    print(results_last_epoch)
    """
    # %%
    ## Full HAR Model

    tag = "full_eval"

    eval_model = base_model
    optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=0.001)
    full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(
        eval_model, output_shape, optimizer_fine_tune, model_name="TPN", intermediate_layer=-1, last_freeze_layer=last_freeze_layer)

    full_eval_best_model_file_name = f"{working_directory}/models/{args.run_name}_{tag}.hdf5"
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
                                                             monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
                                                             )

    training_history = full_evaluation_model.fit(
        x=train_split,
        epochs=total_epochs,
        callbacks=[best_model_callback],
        validation_data=val_split
    )

    full_eval_best_model = tf.keras.models.load_model(
        full_eval_best_model_file_name)

    results_lowest_loss = simclr_utitlities.evaluate_model_simple(
        full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
    results_last_epoch = simclr_utitlities.evaluate_model_simple(
        full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True)

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
    eval_model = base_model
    tsne_image = None

    if args.output_tsne:
        embeddings = eval_model.predict(np_test[0], batch_size=600)
        tsne_projections = visual_utils.fit_transform_tsne(embeddings)
        tsne_figure = visual_utils.plot_tsne(
            tsne_projections, np_test[1], label_name_list=dataset_full.info['session_list'])
        tsne_image = visual_utils.plot_to_image(tsne_figure)
        return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted'], tsne_image
    
    return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted']


if __name__ == '__main__':
    parser = get_parser()
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
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.random.set_seed(42)
    np.random.seed(42)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
        except RuntimeError as e:
            print(e)

    dataset_format = 'np' # 'np', 'functional', 'tf'
    if dataset_format == 'functional':
        dataset_full = load_data.Data(args.dataset_path, args.dataset_name, load_path=None, held_out=args.held_out, format='np')
        converted_dataset = functional_dataset_pipeline.make_structured_dataset(dataset_full)
        dataset_full = functional_dataset_pipeline.FunctionalDatasetPipeline(converted_dataset, original_dataset=dataset_full, input_shape=dataset_full.input_shape, info=dataset_full.info, verbose=1)
    else:
        dataset_full = load_data.Data(args.dataset_path, args.dataset_name, load_path=None, held_out=args.held_out, format=dataset_format)


    if args.baseline == 'multi_task_transform':
        trained_model_save_path = train_multi_task_transform(dataset_full, args)
        print(fine_tune_evaluate(dataset_full, trained_model_save_path, args))
    elif args.baseline == 'en_co_training':
        print(train_and_evaluate_en_co_training(dataset_full, args))
    elif args.baseline == 'random':
        print(Random(dataset_full,args))

