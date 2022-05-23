import tensorflow as tf

def pairwise_distance(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError("Both inputs should be matrices.")

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError("The number of features should be the same.")

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    # By making the `inner' dimensions of the two matrices equal to 1 using
    # broadcasting then we are essentially substracting every pair of rows
    # of x and y.
    # x will be num_samples x num_features x 1,
    # and y will be 1 x num_features x num_samples (after broadcasting).
    # After the substraction we will get a
    # num_x_samples x num_features x num_y_samples matrix.
    # The resulting dist will be of shape num_y_samples x num_x_samples.
    # and thus we need to transpose it again.
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_rbf_kernel(x, y):
    r"""Computes a Gaussian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    # The values usually stays within (-5 ~ 10)
    sigmas = tf.constant((1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 100))

    beta = 1.0 / (2.0 * tf.expand_dims(sigmas, 1))
    dist = pairwise_distance(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-1.0 * s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_rbf_kernel):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
        is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = (
        tf.reduce_mean(kernel(x, x))
        + tf.reduce_mean(kernel(y, y))
        - 2 * tf.reduce_mean(kernel(x, y))
    )
    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name="value")
    return cost


def mmd_loss(source_samples, target_samples, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
        source_samples: a tensor of shape [num_samples, num_features].
        target_samples: a tensor of shape [num_samples, num_features].
        weight: the weight of the MMD loss.
        scope: optional name scope for summary tags.
    Returns:
        a scalar tensor representing the MMD loss value.
    """
    with tf.name_scope("mmd_loss"):
        loss_value = maximum_mean_discrepancy(
            source_samples, target_samples, kernel=gaussian_rbf_kernel
        )
        loss_value = tf.maximum(1e-4, loss_value)
    return loss_value

def js_divergence(source_samples, target_samples):
    return 0.5 * (tf.keras.losses.KLD(source_samples, target_samples) + 
            tf.keras.losses.KLD(target_samples, source_samples))

if __name__ == "__main__":
  from tensorflow.keras.layers import Lambda

  #Add a Lambda layer to do MMD computation
#   MMD = Lambda(lambda x: [mmd_loss(x[0], x[1])], (1,), name="MMD")

  #pass your source and target domain features on which you want to compute MMD 
  #add this loss to your total loss and backpropagate
  
  #source_feature = feature_extractor(source_data)
  #target_feature = feature_extractor(target_data)  
  #loss = MMD([source_feature, target_feature]) 

  source_data = tf.random.uniform(shape=(2, 3), maxval=2, dtype=tf.float64)
  target_data = tf.random.uniform(shape=(2, 3), maxval=4, dtype=tf.float64)

  print(js_divergence(source_data, target_data))

