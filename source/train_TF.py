from models_TF import *
from tensorflow.python.tools import freeze_graph
from classes import *
import argparse


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
    current_train_lst = train_lst[:int(0.9 * train_size)]
    current_validation_lst = validation_lst[:int(0.1 * train_size)]
    train_generator = ImageGenerator(current_train_lst, batch_size, images_dir, images_gt_dir, image_dim)
    batches = len(current_train_lst) // batch_size
    for epoch in range(epochs):
        for i in range(batches):
            X_batch, y_batch = train_generator.__getitem__(i)
            feed_dict_train = {X: X_batch, y: y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        j_acc, d_acc, b_acc = test_accuracy(current_validation_lst, images_dir, images_gt_dir, image_dim)
        message = "epoch: {}, jaccard: {}, dice: {}, bce: {}"
        print(message.format(epoch, j_acc, d_acc, b_acc))


def test_accuracy(validation_lst, images_dir, images_gt_dir, image_dim):
    X_test = []
    y_test = []
    for image_name in validation_lst:
        image = np.asarray(Image.open(os.path.join(images_dir, image_name)), dtype=np.uint8)
        image_gt = np.asarray(Image.open(os.path.join(images_gt_dir, image_name)), dtype=np.uint8)
        H_c = image.shape[0] // 2
        W_c = image_gt.shape[1] // 2
        image = image[H_c - image_dim // 2: H_c + image_dim // 2, W_c - image_dim // 2: W_c + image_dim // 2]
        image_gt = image_gt[H_c - image_dim // 2: H_c + image_dim // 2, W_c - image_dim // 2: W_c + image_dim // 2]
        image = np.expand_dims(normalize_image(image), axis=2)
        image_gt = np.expand_dims(normalize_image(image_gt, gt_bool=True), axis=2)
        X_test.append(image)
        y_test.append(image_gt)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    feed_dict_test = {X: X_test, y: y_test}
    j_acc, d_acc, b_acc = session.run([j_accuracy, d_accuracy, b_accuracy], feed_dict=feed_dict_test)
    return j_acc, d_acc, b_acc


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
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Train a UNet model to binarize images.')
    parser.add_argument('images_dir', metavar='images_dir', type=str, help='Path to train images.')
    parser.add_argument('ground_truth_dir', metavar='ground_truth_dir', type=str,
                        help='Path to ground truth images having the same name as the train images.')
    parser.add_argument('models_path', metavar='models_path', type=str, help='Path to the models directory.')
    parser.add_argument('train_size', metavar='train_size', type=int,
                        help='Number of images to be used during a training session')
    parser.add_argument('epochs', metavar='epochs', type=int,
                        help='Number of times the model will train over all train images')
    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        help='number of images that passes through the network before one back-propagation. '
                             'Larger batch size leads to more reliable gradient steps but one batch must fit in memory.'
                             'Choose batch_size such that it divides the number of images.')
    parser.add_argument('output_model', metavar='output_model', type=str,
                        help='Name under which to save the newly trained model.')
    parser.add_argument('input_model', metavar='input_model', nargs='?', type=str,
                        help='Name of a pre trained model to resume the training from.')
    args = parser.parse_args()

    images_dir = args.images_dir
    images_gt_dir = args.ground_truth_dir

    # Read images' name from directory
    image_name_lst = [image_name for image_name in os.listdir(images_dir)]

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
    models_path = args.models_path

    with tf.Session() as session:
        if args.input_model:
            input_model = os.path.join(models_path, args.input_model)
            saver.restore(session, input_model)
            print("Model restored.")
        else:
            session.run(init_op)
            print("Model initialized")

        # export Graph
        #tf.train.write_graph(session.graph_def, '../save_files/', "Graph.pb")

        train_with_generator(image_name_lst, args.batch_size, images_dir, images_gt_dir, 128, args.epochs,
                             args.train_size)

        # Save the variables to disk.
        output_model = os.path.join(models_path, args.output_model)
        save_path = saver.save(session, output_model)
        #save_path = saver.save(session, '../models/UNet_TF.ckpt')
        # checkpoint_file = os.path.join('../save_files/', 'UNet.ckpt')
        saver.save(session, save_path)
        print("Model saved in path: %s" % save_path)

        # #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        #
        # # Freeze the graph
        # freeze_graph.freeze_graph(input_graph='../save_files/Graph.pb',
        #                           input_saver='',
        #                           input_binary=False,
        #                           input_checkpoint=checkpoint_file,
        #                           output_node_names='output/Sigmoid',
        #                           restore_op_name='save/restore_all',
        #                           filename_tensor_name='save/Const:0',
        #                           output_graph='../save_files/frozen_graph.pb',
        #                           clear_devices=False,
        #                           initializer_nodes="")



