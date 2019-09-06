from keras.models import load_model
import os
from utils import *
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import tensorflow as tf


def predict_images(images, model):
    images = np.expand_dims(images, axis=3)
    pred_images = model.predict(images)
    return pred_images


if __name__ == '__main__':
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Binarize the input image using a UNet architecture.')
    parser.add_argument('image', metavar='image', type=str, help='Path to input image.')
    parser.add_argument('model', metavar='model', type=str,
                        help='Path to model (ex: model.h5) to be used for binarization.')
    parser.add_argument('save_path', metavar='save_path', type=str, help='Path to save the binary image.')
    args = parser.parse_args()

    # Load input image
    image_path = args.image
    image = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)

    # Split image to fit model's input requirements
    images, shape, padding = split_image(image)

    # Load the model
    model_path = args.model
    UNet = load_model(model_path, custom_objects={'jaccard_loss': jaccard_loss,
                                                  'jaccard_accuracy': jaccard_accuracy,
                                                  'dice_loss': dice_loss,
                                                  'dice_accuracy': dice_accuracy})

    # Get binary images
    pred_images = np.squeeze(predict_images(images, UNet))

    # Merge binary images
    binary_image = merge_predictions(pred_images, shape)
    binary_image = binary_image[padding['upper']:-padding['lower'], padding['left']:-padding['right']]
    plt.imshow(binary_image, cmap='gray')
    plt.show()
    binary_image = Image.fromarray(binary_image)
    binary_image.save(args.save_path)


