from utils import *
from PIL import Image
from models_TF import *
import argparse


if __name__ == '__main__':
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Binarize the input image using a UNet architecture.')
    parser.add_argument('image', metavar='image', type=str, help='Path to input image.')
    parser.add_argument('model', metavar='model', type=str,
                        help='Path to model (ex: model.ckpt) to be used for binarization.')
    parser.add_argument('save_path', metavar='save_path', type=str, help='Path to save the binary image.')
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert('L'), dtype=np.uint8)

    # Split image to fit model's input requirements
    images, shape, padding = split_image(image)
    images = np.expand_dims(images, axis=3)

    # Define placeholder variables
    X = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X')

    # Load the mode
    model = UNet(X)

    # Define the predictions
    predictions = model.output

    # ...
    saver = tf.train.Saver()

    with tf.Session() as session:
        # Restore from previous checkpoint
        saver.restore(session, args.model)

        # Get the predictions
        feed_dict_pred = {X: images}
        pred_images = np.squeeze(session.run(predictions, feed_dict=feed_dict_pred))

    binary_image = merge_predictions(pred_images, shape)
    binary_image = binary_image[padding['upper']:-padding['lower'], padding['left']:-padding['right']]
    plt.imshow(binary_image, cmap='gray')
    plt.show()
    binary_image = Image.fromarray(binary_image)
    binary_image.save(args.save_path)
