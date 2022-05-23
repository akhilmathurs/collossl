import distances
import load_data
from common_parser import get_parser
import tensorflow as tf
import numpy as np
import csv
tf.random.set_seed(42)
np.random.seed(42)

parser = get_parser()
args = parser.parse_args()


def get_pos_neg(data_batch, anchor_index=0, strategy='hard_negative'):
    """
        Selects positive and negative devices according to the device selection metric and device selection logic
        
        Parameters:
            data_batch
                Tensorflow dataset batched separated at axis 0 along all the devices that is to be considered for selection
            anchor_index
                index of the anchor device in data_batch
        
        Return:
            positive_devices, negative_devices
                Indices of respective devices in the data_batch parameter

    """
    all_indices = np.delete(np.arange(data_batch.shape[0]), anchor_index) # rest devices
    source_ds = data_batch[anchor_index] # anchor device 
    mmd = tf.keras.metrics.Mean(name='mmd')
    pairwise_distances = []
    for target in all_indices:
        candidate_ds = data_batch[target]
        acc_x, gyro_x = tf.split(source_ds, num_or_size_splits=2, axis=2)
        acc_norm_x = tf.sqrt(tf.reduce_sum(tf.square(acc_x), axis=2))

        acc_y, gyro_y = tf.split(candidate_ds, num_or_size_splits=2, axis=2)
        acc_norm_y = tf.sqrt(tf.reduce_sum(tf.square(acc_y), axis=2))

        if args.device_selection_metric=='mmd_acc_per_channel':
            mmd(sum([distances.mmd_loss(acc_x[:,:,i], acc_y[:,:,i]) for i in range(3)])/3) #assumes that acc data is in first 3 indices
        elif args.device_selection_metric=='mmd_acc_norm':
            mmd(distances.mmd_loss(acc_norm_x, acc_norm_y))

        pairwise_distances.append(mmd.result().numpy())
        # TF changed the API between 2.4 and 2.5, so TF2.4 uses reset_states and TF2.5 uses reset_state
        mmd.reset_states()
    device_order = np.array([all_indices[i] for i in np.argsort(pairwise_distances)])
    pairwise_distances_dict = {i:j for i,j in zip(all_indices, pairwise_distances)}
    return device_selection_logic(device_order, pairwise_distances_dict, strategy=strategy)

def get_pos_neg_apriori(dataset, all_devices, anchor_index=0, strategy='hard_negative'):
    """
        Selects positive and negative devices according to the device selection metric and device selection logic
        
        Parameters:
            dataset
                Tensorflow dataset containing all devices
                Assuming that anchor device is at index 0
        
        Return:
            positive_devices, negative_devices
                Indices of respective devices in the data_batch parameter

    """
    all_indices = np.delete(np.arange(len(all_devices)), anchor_index) # rest devices
    pairwise_distances = []
    for target in all_indices:
        mmd = tf.keras.metrics.Mean(name='mmd')
        for data_batch in dataset:
            source_ds = data_batch[anchor_index] 
            candidate_ds = data_batch[target]

            acc_x, gyro_x = tf.split(source_ds, num_or_size_splits=2, axis=2)
            acc_norm_x = tf.sqrt(tf.reduce_sum(tf.square(acc_x), axis=2))

            acc_y, gyro_y = tf.split(candidate_ds, num_or_size_splits=2, axis=2)
            acc_norm_y = tf.sqrt(tf.reduce_sum(tf.square(acc_y), axis=2))

            if args.device_selection_metric=='mmd_acc_per_channel':
                mmd(sum([distances.mmd_loss(acc_x[:,:,i], acc_y[:,:,i]) for i in range(3)])/3) #assumes that acc data is in first 3 indices
            elif args.device_selection_metric=='mmd_acc_norm':
                mmd(distances.mmd_loss(acc_norm_x, acc_norm_y))
        pairwise_distances.append(mmd.result().numpy())
    device_order = np.array([all_indices[i] for i in np.argsort(pairwise_distances)])
    pairwise_distances_dict = {i:j for i,j in zip(all_indices, pairwise_distances)}
    return device_selection_logic(device_order, pairwise_distances_dict, strategy=strategy)

def device_selection_logic(device_order, pairwise_distances, strategy='hard_negative'):
    """
        Params:
            device_order
                Order of devices sorted in increasing order according to distance 

        return
            positive_devices and negative_devices indices

    """
    ## TODO: Implement device selection logic on device_order var
    ## Template implementation of Hard negative sampling

    if strategy=='closest_only':
        positive_devices = negative_devices = [device_order[0]]
    elif strategy=='closest_pos_all_neg':
        positive_devices = [device_order[0]]
        negative_devices = device_order.tolist()
        negative_devices.sort()    
    elif strategy=='hard_negative':
        positive_devices = [device_order[0]]
        negative_devices = device_order[1:3].tolist()
        negative_devices.sort()    
    elif strategy=='harder_negative':
        positive_devices = [device_order[3]]
        negative_devices = device_order[0:2].tolist()
        negative_devices.sort()
    elif strategy=='closest_pos_rest_neg':
        positive_devices = [device_order[0]]
        negative_devices = device_order[1:].tolist()
        negative_devices.sort()
    elif strategy=='closest_two':
        positive_devices = [device_order[0]]
        negative_devices = [device_order[1]]
    elif strategy=='closest_two_reverse':
        positive_devices = [device_order[1]]
        negative_devices = [device_order[0]]
    elif strategy=='random_selection':
        positive_devices = [device_order[np.random.randint(len(device_order))]]
        negative_devices = [device_order[np.random.randint(len(device_order))]]
        pairwise_distances = {key:1.0 for key in pairwise_distances.keys()}
    elif strategy=='mid_selection':
        positive_devices = device_order[1:3].tolist()
        negative_devices = device_order[3:].tolist()
    elif strategy=='closest_pos_random_neg': 
        positive_devices = [device_order[0]]
        negative_devices = [device_order[np.random.randint(len(device_order))]]
    return positive_devices, negative_devices, pairwise_distances

def test_mmd():

    device_list = ['thigh', 'forearm', 'head', 'chest', 'upperarm', 'waist', 'shin']

    csv_file =  open('mmd_both.csv', 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['source/target'] + device_list)
    writer.writeheader()

    mmd = tf.keras.metrics.Mean(name='mmd')
    tf_dataset_full = load_data.Data(args.dataset_path, args.dataset_name)


    for source in device_list:
        source_ds = tf_dataset_full.ds_train[source].map(lambda x, y, i: x)
        result_dict = {}
        for target in device_list:
            candidate_ds = tf_dataset_full.ds_train[target].map(lambda x, y, i: x)
            for x,y in tf.data.Dataset.zip((source_ds, candidate_ds)).batch(128):
                acc_x, gyro_x = tf.split(x, num_or_size_splits=2, axis=2)
                acc_norm_x = tf.sqrt(tf.reduce_sum(tf.square(acc_x), axis=2))
                # gyro_norm_x = tf.sqrt(tf.reduce_sum(tf.square(gyro_x), axis=2))

                acc_y, gyro_y = tf.split(y, num_or_size_splits=2, axis=2)
                acc_norm_y = tf.sqrt(tf.reduce_sum(tf.square(acc_y), axis=2))
                # gyro_norm_y = tf.sqrt(tf.reduce_sum(tf.square(gyro_y), axis=2))

                mmd(distances.mmd_loss(acc_norm_x, acc_norm_y))

            result_dict[target] = mmd.result().numpy()
            mmd.reset_states()
        result_dict['source/target'] = source
        writer.writerow(result_dict)

# test_mmd()
