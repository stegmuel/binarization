import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # Load the graph and placeholders
    graph = load_graph('../save_files/frozen_graph.pb')
    X = graph.get_tensor_by_name('prefix/X:0')
    y = graph.get_tensor_by_name('prefix/output/Sigmoid:0')

    # Load inputs
    train_images, train_images_gt, validation_images, validation_images_gt = load_images(20)

    with tf.Session(graph=graph) as session:
        feed_dict_pred = {X: validation_images}
        image_pred = session.run(y, feed_dict=feed_dict_pred)

        image_pred = np.squeeze(image_pred[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(image_pred)
        axes[1].imshow(np.squeeze(validation_images_gt[0]))
        plt.show()

