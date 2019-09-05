from models_TF import *
import numpy as np
from utils import *
import decimal
import matplotlib.pyplot as plt
from tensorflow.python.tools import freeze_graph
from classes import *


def flatten_tensor(tensor):
    input_shape = tensor.shape
    num_features = input_shape[1:4].num_elements()
    layer_flat = tf.reshape(tensor=tensor, shape=[-1, num_features])
    return layer_flat


def dice_accuracy(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / denominator


def dice_loss(y_true, y_pred):
    return 1 - dice_accuracy(y_true, y_pred)


def jaccard_accuracy(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred) - numerator
    return numerator / denominator


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_accuracy(y_true, y_pred)


def bce_accuracy(y_true, y_pred):
    return 1 - bce_loss(y_true, y_pred)


def bce_loss(y_true, y_pred):
    return tf.losses.log_loss(flatten_tensor(y_true), flatten_tensor(y_pred))


# def train(train_images, train_images_gt, epochs=10):
#     for epoch in range(epochs):
#         feed_dict_train = {X: train_images, y: train_images_gt}
#         session.run(optimizer, feed_dict=feed_dict_train)
#         acc = session.run(accuracy, feed_dict=feed_dict_train)
#         message = "Train accuracy at epoch {} is {}"
#         print(message.format(epoch, acc))


def train_batch(train_images, train_images_gt, validation_images, validation_images_gt, epochs, batch_size=10):
    batches = train_images.shape[0] // batch_size
    for epoch in range(epochs):
        for i in range(batches):
            X_batch = train_images[batch_size*i:batch_size*(i+1)]
            y_batch = train_images_gt[batch_size*i:batch_size*(i+1)]

            feed_dict_train = {X: X_batch, y: y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        j_acc, d_acc, b_acc = get_test_accuracy(validation_images, validation_images_gt)
        message = "epoch: {}, jaccard: {}, dice: {}, bce: {}"
        print(message.format(epoch, j_acc, d_acc, b_acc))


def train_with_generator(image_name_lst, batch_size, images_dir, images_gt_dir, image_dim, epochs, train_size):
    train_limit = int(0.9 * len(image_name_lst))
    train_lst = image_name_lst[:train_limit]
    validation_lst = image_name_lst[train_limit:]
    shuffle(train_lst)

    train_generator = ImageGenerator(image_name_lst, batch_size, images_dir, images_gt_dir, image_dim)
    batches = len(image_name_lst) // batch_size
    for epoch in range(epochs):
        for i in range(batches):
            X_batch, y_batch = train_generator.__getitem__(i)
            feed_dict_train = {X: X_batch, y: y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)


def get_test_accuracy(validation_images, validation_images_gt):
    feed_dict_test = {X: validation_images, y: validation_images_gt}
    j_acc, d_acc, b_acc = session.run([j_accuracy, d_accuracy, b_accuracy], feed_dict=feed_dict_test)
    return j_acc, d_acc, b_acc


def combine_cost(y_true, y_pred):
    j_loss = jaccard_loss(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    b_loss = bce_loss(y_true, y_pred)
    return 1/3 * (j_loss + d_loss + b_loss)


if __name__ == '__main__':
    train_images, train_images_gt, validation_images, validation_images_gt = load_images(100)

    # Define placeholder variables
    X = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='y')

    # Load the model
    model = UNet(X)

    # Define cost
    cost = bce_loss(y, model.output)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Define accuracy metrics
    j_accuracy = jaccard_accuracy(y, model.output)
    d_accuracy = dice_accuracy(y, model.output)
    b_accuracy = bce_accuracy(y, model.output)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Delete previous save
    if os.path.exists('../save_files/'):
        shutil.rmtree('../save_files/')

    # initialize Variables
    summary = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    model_path = '../models/UNet_TF.ckpt'

    with tf.Session() as session:
        # Restore from previous checkpoint
        # saver.restore(session, model_path)
        # print("Model restored.")
        session.run(init_op)

        # export Graph
        tf.train.write_graph(session.graph_def, '../save_files/', "Graph.pb")

        train_batch(train_images, train_images_gt, validation_images, validation_images_gt, epochs=3, batch_size=10)

        # Save the variables to disk.
        # save_path = saver.save(session, '../models/UNet_TF.ckpt')
        checkpoint_file = os.path.join('../save_files/', 'UNet.ckpt')
        saver.save(session, checkpoint_file)

        print("Model saved in path: %s" % checkpoint_file)

        feed_dict_pred = {X: validation_images, y: validation_images_gt}
        pred = model.output
        image_pred = session.run(pred, feed_dict=feed_dict_pred)

        image_pred = np.squeeze(image_pred[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(image_pred)
        axes[1].imshow(np.squeeze(validation_images_gt[0]))
        plt.show()

        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        # Freeze the graph
        freeze_graph.freeze_graph(input_graph='../save_files/Graph.pb',
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=checkpoint_file,
                                  output_node_names='output/Sigmoid',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='../save_files/frozen_graph.pb',
                                  clear_devices=False,
                                  initializer_nodes="")



