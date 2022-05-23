from device_selection import get_pos_neg
import numpy as np
import tensorflow as tf
import sklearn.metrics
tf.random.set_seed(42)
np.random.seed(42)
# import data_pre_processing

__author__ = "C. I. Tang"
__copyright__ = """Copyright (C) 2020 C. I. Tang"""

"""
This file includes software licensed under the Apache License 2.0, modified by C. I. Tang.

Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def generate_composite_transform_function_simple(transform_funcs):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions
            the function is composed by applying 
            transform_funcs[0] -> transform_funcs[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """
    for i, func in enumerate(transform_funcs):
        print(i, func)
    def combined_transform_func(sample):
        for func in transform_funcs:
            sample = func(sample)
        return sample
    return combined_transform_func

def generate_combined_transform_function(transform_funcs, indices=[0]):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions

        indices
            list of indices corresponding to the transform_funcs
            the function is composed by applying 
            function indices[0] -> function indices[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """

    for index in indices:
        print(transform_funcs[index])
    def combined_transform_func(sample):
        for index in indices:
            sample = transform_funcs[index](sample)
        return sample
    return combined_transform_func

def generate_slicing_transform_function(transform_func_structs, slicing_axis=2, concatenate_axis=2):
    """
    Create a transformation function with slicing by applying different transformation functions to different slices.
    The output arrays are then concatenated at the specified axis.

    Parameters:
        transform_func_structs
            list of transformation function structs
            each transformation functions struct is a 2-tuple of (indices, transform_func)

            each transformation function is applied by
                transform_func(np.take(data, indices, slicing_axis))
            
            all outputs are concatenated in the output axis (concatenate_axis)

            Example:
                transform_func_structs = [
                    ([0,1,2], transformations.rotation_transform_vectorized),
                    ([3,4,5], transformations.time_flip_transform_vectorized)
                ]

        slicing_axis = 2
            the axis from which the slicing is applied
            (see numpy.take)

        concatenate_axis = 2
            the axis which the transformed array (tensors) are concatenated
            if it is None, a list will be returned

    Returns:
        slicing_transform_func
            a slicing transformation function 
    """
    def slicing_transform_func(sample):
        all_slices = []
        for indices, transform_func in transform_func_structs:
            trasnformed_slice = transform_func(np.take(sample, indices, slicing_axis))
            all_slices.append(trasnformed_slice)
        if concatenate_axis is None:
            return all_slices
        else:
            return np.concatenate(all_slices, axis=concatenate_axis)
    return slicing_transform_func


def get_NT_Xent_loss_gradients(model, samples_transform_1, samples_transform_2, normalize=True, temperature=1.0, weights=1.0, training=False):
    """
    A wrapper function for the NT_Xent_loss function which facilitates back propagation

    Parameters:
        model
            the deep learning model for feature learning 

        samples_transform_1
            inputs samples subject to transformation 1
        
        samples_transform_2
            inputs samples subject to transformation 2

        normalize = True
            normalise the activations if true

        temperature = 1.0
            hyperparameter, the scaling factor of the logits
            (see NT_Xent_loss)
        
        weights = 1.0
            weights of different samples
            (see NT_Xent_loss)

    Return:
        loss
            the value of the NT_Xent_loss

        gradients
            the gradients for backpropagation
    """
    with tf.GradientTape() as tape:
        hidden_features_transform_1 = model(samples_transform_1, training=training)
        hidden_features_transform_2 = model(samples_transform_2, training=training)
        loss = NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=normalize, temperature=temperature, weights=weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients



def simclr_train_model(model, dataset, optimizer, batch_size, transformation_function, temperature=1.0, epochs=100, is_trasnform_function_vectorized=False, verbose=0, is_tf_dataset=False, training=False):
    """
    Train a deep learning model using the SimCLR algorithm

    Parameters:
        model
            the deep learning model for feature learning 

        dataset
            the numpy array for training (no labels)
            the first dimension should be the number of samples
        
        optimizer
            the optimizer for training
            e.g. tf.keras.optimizers.SGD()

        batch_size
            the batch size for mini-batch training

        transformation_function
            the stochastic (probabilistic) function for transforming data samples
            two different views of the sample is generated by applying transformation_function twice

        temperature = 1.0
            hyperparameter of the NT_Xent_loss, the scaling factor of the logits
            (see NT_Xent_loss)
        
        epochs = 100
            number of epochs of training
            
        is_trasnform_function_vectorized = False
            whether the transformation_function is vectorized
            i.e. whether the function accepts data in the batched form, or single-sample only
            vectorized functions reduce the need for an internal for loop on each sample

        verbose = 0
            debug messages are printed if > 0

    Return:
        (model, epoch_wise_loss)
            model
                the trained model
            epoch_wise_loss
                list of epoch losses during training
    """

    epoch_wise_loss = []

    for epoch in range(epochs):
        step_wise_loss = []

        if is_tf_dataset:
            batched_dataset = dataset
        else:
            # Randomly shuffle the dataset
            shuffle_indices = np_random_shuffle_index(len(dataset))
            shuffled_dataset = dataset[shuffle_indices]

            # Make a batched dataset
            batched_dataset = get_batched_dataset_generator(shuffled_dataset, batch_size)

        for data_batch in batched_dataset:

            # Apply transformation
            if is_trasnform_function_vectorized:
                transform_1 = transformation_function(data_batch)
                transform_2 = transformation_function(data_batch)
            else:
                transform_1 = np.array([transformation_function(data) for data in data_batch])
                transform_2 = np.array([transformation_function(data) for data in data_batch])

            # Forward propagation
            loss, gradients = get_NT_Xent_loss_gradients(model, transform_1, transform_2, normalize=True, temperature=temperature, weights=1.0, training=training)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        
        if verbose > 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return model, epoch_wise_loss


def evaluate_model_simple(pred, truth, is_one_hot=True, return_dict=True):
    """
    Evaluate the prediction results of a model with 7 different metrics
    Metrics:
        Confusion Matrix
        F1 Macro
        F1 Micro
        F1 Weighted
        Precision
        Recall 
        Kappa (sklearn.metrics.cohen_kappa_score)

    Parameters:
        pred
            predictions made by the model

        truth
            the ground-truth labels
        
        is_one_hot=True
            whether the predictions and ground-truth labels are one-hot encoded or not

        return_dict=True
            whether to return the results in dictionary form (return a tuple if False)

    Return:
        results
            dictionary with 7 entries if return_dict=True
            tuple of size 7 if return_dict=False
    """

    if is_one_hot:
        truth_argmax = np.argmax(truth, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
    else:
        truth_argmax = truth
        pred_argmax = pred

    test_cm = sklearn.metrics.confusion_matrix(truth_argmax, pred_argmax)
    test_f1 = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro')
    test_precision = sklearn.metrics.precision_score(truth_argmax, pred_argmax, average='macro')
    test_recall = sklearn.metrics.recall_score(truth_argmax, pred_argmax, average='macro')
    test_kappa = sklearn.metrics.cohen_kappa_score(truth_argmax, pred_argmax)

    test_f1_micro = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='micro')
    test_f1_weighted = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='weighted')

    if return_dict:
        return {
            'Confusion Matrix': test_cm, 
            'F1 Macro': test_f1, 
            'F1 Micro': test_f1_micro, 
            'F1 Weighted': test_f1_weighted, 
            'Precision': test_precision, 
            'Recall': test_recall, 
            'Kappa': test_kappa
        }
    else:
        return (test_cm, test_f1, test_f1_micro, test_f1_weighted, test_precision, test_recall, test_kappa)

"""
The following section of this file includes software licensed under the Apache License 2.0, by The SimCLR Authors 2020, modified by C. I. Tang.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""

@tf.function
def NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    The normalised temperature-scaled cross entropy loss function of SimCLR Contrastive training
    Reference: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    https://github.com/google-research/simclr/blob/master/objective.py

    Parameters:
        hidden_features_transform_1
            the features (activations) extracted from the inputs after applying transformation 1
            e.g. model(transform_1(X))
        
        hidden_features_transform_2
            the features (activations) extracted from the inputs after applying transformation 2
            e.g. model(transform_2(X))

        normalize = True
            normalise the activations if true

        temperature
            hyperparameter, the scaling factor of the logits
        
        weights
            weights of different samples

    Return:
        loss
            the value of the NT_Xent_loss
    """
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    batch_size = tf.shape(hidden_features_transform_1)[0]
    h1 = hidden_features_transform_1
    h2 = hidden_features_transform_2

    if normalize:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    # Suppresses the logit of the repeated sample, which is in the diagonal of logit_aa
    # i.e. the product of h1[x] . h1[x]
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    
    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b

    return loss

def np_random_shuffle_index(length):
    """
    Get a list of randomly shuffled indices
    (From data_pre_processing)
    """
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def get_batched_dataset_generator(data, batch_size):
    """
    Create a data batch generator
    Note that the last batch might not be full
    (From data_pre_processing)

    Parameters:
        data
            A numpy array of data

        batch_size
            the (maximum) size of the batches

    Returns:
        generator<numpy array>
            a batch of the data with the same shape except the first dimension, which is now the batch size
    """

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

    # return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))


dds_pair_stats, dds_pos_device_stats, dds_neg_device_stats = {}, {}, {}
def group_supervised_contrastive_train_model(model, datasets, transformation_function, optimizer, index_mappings, distances, loss_function, strategy, args, weighted=False, epochs=100, verbose=0, patience=20, training=False, temporary_save_model_path=None):
    """
    Train a deep learning model with group-supervised contrastive training (still under testing)

    Parameters:
        model
            the deep learning model for feature learning 

        dataset
            an iterable object which, for each iteration returns an iterable object where each of them is a batched input to the model
            i.e.
                for data_batch in dataset:
                    for d in data_batch:
                        model(d)
            should work
            An example for a model which accepts input of shape (?, 100, 6), 
                dataset is a batched tf dataset object which returns objects in shape (n, ?, 100, 6)
                where n should be the total number of devices, ? is the batch size

        optimizer
            the optimizer for training
            e.g. tf.keras.optimizers.SGD()

        index_mappings
            a tuple containing anchor_index, positive_indices, negative_indices
            these correspond to the mappings of the devices indices to their corresponding groupings in the group-supervised training
            anchor_index should be an integer, positive_indices and negative_indices should be both lists of integers

        loss_function
            the loss function for group-supervised training
            it should accept three arguments: anchor_embedding, positive_embeddings, negative_embeddings
            anchor_embedding has shape (?, e)
            positive_embeddings has shape (p, ?, e)
            negative_embeddings has shape (n, ?, e)
            ? refers to the batch size (variable), e is the embedding size of the model, p is the number of positive devices, n is the number of negative devices

        verbose = 0
            debug messages are printed if > 0

    Return:
        (model, epoch_wise_loss)
            model
                the trained model
            epoch_wise_loss
                list of epoch losses during training
    """

    
    anchor_index, positive_indices, negative_indices = index_mappings

    epoch_wise_loss = []
    global dds_pair_stats, dds_pos_device_stats, dds_neg_device_stats #these variables are just for book-keeping, have no effect on training
    dds = True if (len(positive_indices) == 0 or len(negative_indices) == 0) else False
    patience_count = 0
    mean_epoch_loss = 0
    for epoch in range(epochs):
        step_wise_loss = []

        for enum, data_batch in enumerate(datasets):
            
            loss = 0
            if dds:
                # if (enum+1)%5==0:
                #     num_devices = data_batch.shape[0]
                #     anchor_index = ((enum+1)//5)%num_devices
                if args.multi_anchor:
                    num_devices = data_batch.shape[0]
                    anchor_index = np.random.randint(0,high=num_devices)
                positive_indices, negative_indices, distances = get_pos_neg(data_batch, anchor_index, strategy=strategy)
                # distances = tf.convert_to_tensor(distances, dtype=tf.float64) #distances is a dict now
                positive_indices.sort()
                negative_indices.sort()

                # book-keeping, no involvement in training
                tmp = tuple([tuple(positive_indices), tuple(negative_indices)])
                if tmp in dds_pair_stats:
                    dds_pair_stats[tmp] += 1
                else:
                    dds_pair_stats[tmp] = 1
                
                for p in positive_indices:
                    if p in dds_pos_device_stats:
                        dds_pos_device_stats[p] += 1
                    else:
                        dds_pos_device_stats[p] = 1
                
                for n in negative_indices:
                    if n in dds_neg_device_stats:
                        dds_neg_device_stats[n] += 1
                    else:
                        dds_neg_device_stats[n] = 1
                
            with tf.GradientTape() as tape:
                embeddings = []
                for device_index, d in enumerate(data_batch):
                    if (device_index == anchor_index) or (device_index in positive_indices) or (device_index in negative_indices):
                        embeddings.append(model(d, training=training))
                    else:
                        embeddings.append(None)
                anchor_embedding = embeddings[anchor_index]
                positive_embeddings = tf.stack(tuple([embeddings[index] for index in positive_indices]), axis=0)
                negative_embeddings = tf.stack(tuple([embeddings[index] for index in negative_indices]), axis=0)
                if args.multi_sampling_mode == 'sync_all':
                    # multi_negative_embeddings = negative_embeddings
                    multi_negative_embeddings_tmp = []
                    # this for loop sequence mimics the repeat function, useful for negative_weights later
                    for i in range(negative_embeddings.shape[0]): 
                        for j in range(args.neg_sample_size):
                            multi_negative_embeddings_tmp.append(negative_embeddings[i])
                    multi_negative_embeddings = tf.stack(tuple(multi_negative_embeddings_tmp), axis=0)
                else: 
                    # unsync_neg 
                    multi_negative_embeddings_tmp = []
                    perm_idx = np.arange(negative_embeddings[0].shape[0])                    
                    # this for loop sequence mimics the repeat function, useful for negative_weights later
                    for i in range(negative_embeddings.shape[0]): 
                        for j in range(args.neg_sample_size):
                            np.random.shuffle(perm_idx)
                            multi_negative_embeddings_tmp.append(tf.gather(negative_embeddings[i], indices=perm_idx, axis=0))
                            # multi_negative_embeddings_tmp.append(tf.random.shuffle(negative_embeddings[i], seed=i*100+j)) #strangely this gives error :/
                    multi_negative_embeddings = tf.stack(tuple(multi_negative_embeddings_tmp), axis=0)
                if weighted:
                    positive_weights = tf.stack(tuple([tf.broadcast_to(1.0, [embeddings[index].shape[0]]) for index in positive_indices]), axis=0)                
                    negative_weights = tf.stack(tuple([tf.broadcast_to(1./distances[index], [embeddings[index].shape[0]]) for index in negative_indices]), axis=0)
                    negative_weights = tf.repeat(negative_weights, repeats=args.neg_sample_size, axis=0) #note both the times tf.repeat is used so wts are consistent with embeddings
                else:
                    positive_weights = tf.stack(tuple([tf.broadcast_to(1.0, [embeddings[index].shape[0]]) for index in positive_indices]), axis=0)                
                    negative_weights = tf.stack(tuple([tf.broadcast_to(1.0, [embeddings[index].shape[0]]) for index in negative_indices]), axis=0)
                    negative_weights = tf.repeat(negative_weights, repeats=args.neg_sample_size, axis=0) #note both the times tf.repeat is used so wts are consistent with embeddings

                if distances == None:
                    positive_weights = tf.stack(tuple([tf.broadcast_to(1.0, [embeddings[index].shape[0]]) for index in positive_indices]), axis=0)                
                    negative_weights = tf.stack(tuple([tf.broadcast_to(1.0, [embeddings[index].shape[0]]) for index in negative_indices]), axis=0)
                    negative_weights = tf.repeat(negative_weights, repeats=args.neg_sample_size, axis=0) #note both the times tf.repeat is used so wts are consistent with embeddings
                # breakpoint()
                negative_weights = tf.cast(negative_weights, dtype='float32')
                max_weight = tf.maximum(tf.reduce_max(positive_weights), tf.reduce_max(negative_weights))
                positive_weights = positive_weights / max_weight
                negative_weights = negative_weights / max_weight

                loss = loss_function(anchor_embedding, positive_embeddings, positive_weights, multi_negative_embeddings, negative_weights)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if verbose > 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        if temporary_save_model_path is not None:
            if epoch == 0:
                print(f"First epoch, saving lowest loss model to {temporary_save_model_path}")
                model.save(temporary_save_model_path)
            elif epoch_wise_loss[-1] < min(epoch_wise_loss[:-1]):
                print(f"New lowest loss {epoch_wise_loss[-1]:.3f} < {min(epoch_wise_loss[:-1]):.3f}, saving lowest loss model to {temporary_save_model_path}")
                model.save(temporary_save_model_path)

        if epoch > 60:
            if epoch_wise_loss[-1] >= min(epoch_wise_loss[:-1]):
                patience_count +=1
            else:
                patience_count = 0
            
            if patience_count > patience:
                print(f"Early Stopping at {epoch+1}")
                return model, epoch_wise_loss
                # break


    return model, epoch_wise_loss
