import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
import numpy as np
import io
import tensorflow as tf

def fit_transform_tsne(embeddings, perplexity=30., random_state=None, verbose=1):
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=verbose, random_state=None)
    tsne_projections = tsne_model.fit_transform(embeddings)
    return tsne_projections

def plot_tsne(tsne_projections, labels_one_hot, label_name_list=None):
    labels_argmax = np.argmax(labels_one_hot, axis=1)
    unique_labels = np.unique(labels_argmax)

    figure = plt.figure(figsize=(16,8))
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        palette=sns.color_palette("hsv", len(unique_labels)),
        s=50,
        alpha=1.0,
        rasterized=True
    )
    plt.xticks([], [])
    plt.yticks([], [])


    plt.legend(loc='lower left', ncol=2) # bbox_to_anchor=(0.25, -0.3),
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        if label_name_list is None:
            legend.get_texts()[j].set_text(str(j)) 
        else:
            legend.get_texts()[j].set_text(label_name_list[label]) 

    return figure
        
def plot_to_image(figure):
    """
    https://www.tensorflow.org/tensorboard/image_summaries
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    