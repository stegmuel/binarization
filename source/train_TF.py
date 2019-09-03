from models_TF import *
import numpy as np
from utils import *
import decimal


def flatten_tensor(tensor):
    input_shape = tensor.shape
    num_features = input_shape[1:4].num_elements()
    layer_flat = tf.reshape(tensor=tensor, shape=[-1, num_features])
    return layer_flat


def jaccard_accuracy(y_true, y_pred):
    eps = 1.0
    y_true_flat = flatten_tensor(y_true)
    y_pred_flat = flatten_tensor(y_pred)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = tf.reduce_sum(y_true_flat + y_pred_flat, axis=1)
    return tf.reduce_mean(intersection / union, axis=0)


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_accuracy(y_true, y_pred)


def load_images(num_images):
    data_dir = '../data/train_data'
    train_list_path = '../data/full_train.lst'
    validation_list_path = '../data/full_validation.lst'
    prepare_training(data_dir, train_list_path, validation_list_path, num_images, ratio=0.9)
    train_images_names, train_images_gt_names = get_images_names(os.path.join('../training', 'train.lst'))
    validation_images_names, validation_images_gt_names = \
        get_images_names(os.path.join('../training', 'validation.lst'))

    train_images = []
    train_images_gt = []
    for train_image_name, train_image_gt_name in zip(train_images_names, train_images_gt_names):
        train_image = np.load(os.path.join(data_dir, train_image_name))
        train_image = normalize_image(train_image)
        train_image = np.expand_dims(train_image, axis=2)
        train_images.append(train_image)

        train_image_gt = np.load(os.path.join(data_dir, train_image_gt_name))
        train_image_gt = normalize_image(train_image_gt, gt_bool=True)
        train_image_gt = np.expand_dims(train_image_gt, axis=2)
        train_images_gt.append(train_image_gt)
    train_images = np.asarray(train_images)
    train_image_gt = np.asarray(train_images_gt)

    validation_images = []
    validation_images_gt = []
    for validation_image_name, validation_image_gt_name in zip(validation_images_names, validation_images_gt_names):
        validation_image = np.load(os.path.join(data_dir, validation_image_name))
        validation_image = normalize_image(validation_image)
        validation_image = np.expand_dims(validation_image, axis=2)
        validation_images.append(validation_image)

        validation_image_gt = np.load(os.path.join(data_dir, validation_image_gt_name))
        validation_image_gt = normalize_image(validation_image_gt, gt_bool=True)
        validation_image_gt = np.expand_dims(validation_image_gt, axis=2)
        validation_images_gt.append(validation_image_gt)
    validation_images = np.asarray(validation_images)
    validation_images_gt = np.asarray(validation_images_gt)
    return train_images, train_images_gt, validation_images, validation_images_gt


def train(train_images, train_images_gt, epochs=10):
    for epoch in range(epochs):
        feed_dict_train = {X: train_images, y: train_images_gt}
        session.run(optimizer, feed_dict=feed_dict_train)
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        message = "Train accuracy at epoch {} is {}"
        print(message.format(epoch, acc))


def train_batch(train_images, train_images_gt, validation_images, validation_images_gt, epochs, batch_size=10):
    batches = train_images.shape[0] // batch_size
    for epoch in range(epochs):
        for i in range(batches):
            X_batch = train_images[batch_size*i:batch_size*(i+1)]
            y_batch = train_images_gt[batch_size*i:batch_size*(i+1)]

            feed_dict_train = {X: X_batch, y: y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        acc = get_test_accuracy(validation_images, validation_images_gt)
        message = "Validation accuracy at epoch {} is {}"
        print(message.format(epoch, acc))


def get_test_accuracy(validation_images, validation_images_gt):
    feed_dict_test = {X: validation_images, y: validation_images_gt}
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    return acc


if __name__ == '__main__':
    train_images, train_images_gt, validation_images, validation_images_gt = load_images(20)

    # Define placeholder variables
    X = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='y')

    # Load the model
    model = UNet(X)

    # Define cost
    cost = jaccard_loss(y, model.output)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    accuracy = jaccard_accuracy(y, model.output)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    model_path = '../models/UNet_TF.ckpt'

    with tf.Session() as session:
        session.run(init_op)
        # saver.restore(session, model_path)
        # print("Model restored.")

        train_batch(train_images, train_images_gt, validation_images, validation_images_gt, epochs=10, batch_size=10)

        # Save the variables to disk.
        save_path = saver.save(session, '../models/UNet_TF.ckpt')
        print("Model saved in path: %s" % save_path)


