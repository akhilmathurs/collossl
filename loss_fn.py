import tensorflow as tf
tf.random.set_seed(42)

# @tf.function
def weighted_group_contrastive_loss_with_temp(anchor_embedding, positive_embeddings, positive_weights, negative_embeddings, negative_weights, temperature=2):
    """
    the loss function for group-supervised training 
    it has extra tempearture argument 
    it accepts three arguments: anchor_embedding, positive_embeddings, negative_embeddings
    anchor_embedding has shape (?, e)
    positive_embeddings has shape (p, ?, e)
    negative_embeddings has shape (n, ?, e)
    ? refers to the batch size (variable), e is the embedding size of the model, p is the number of positive devices, n is the number of negative devices
    """
    anchor = tf.convert_to_tensor(anchor_embedding)
    pos_embs = tf.convert_to_tensor(positive_embeddings, anchor.dtype)
    neg_embs = tf.convert_to_tensor(negative_embeddings, anchor.dtype)
    sim = tf.keras.losses.CosineSimilarity(
        axis=-1, reduction=tf.keras.losses.Reduction.NONE)

    pos_sim = sim(tf.broadcast_to(anchor, pos_embs.shape), pos_embs)/temperature  # (p,?)
    neg_sim = sim(tf.broadcast_to(anchor, neg_embs.shape), neg_embs)/temperature  # (n,?)
    numerator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0))  # (?)
    denominator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0) +
                         tf.reduce_sum(tf.exp(neg_sim) * tf.cast(negative_weights, neg_sim.dtype), axis=0)) # (?)
    return tf.math.reduce_mean(denominator - numerator)
