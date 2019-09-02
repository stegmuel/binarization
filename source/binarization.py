from keras.models import load_model
import os
from utils import *
import matplotlib.pyplot as plt
from PIL import Image


def split_image(image, crop_size=128):
    shape = image.shape
    H = image.shape[0]
    W = image.shape[1]
    new_H = 128 * (H // crop_size + 1)
    new_W = 128 * (W // crop_size + 1)
    upper_padding = (new_H - H) // 2
    lower_padding = new_H - H - upper_padding
    left_padding = (new_W - W) // 2
    right_padding = new_W - W - left_padding
    image = np.pad(image, ((upper_padding, lower_padding), (left_padding, right_padding)), 'constant',
                   constant_values=np.median(image))
    images = []
    for row in range(0, new_H, crop_size):
        for col in range(0, new_W, crop_size):
            image_cropped = image[row:row+crop_size, col:col+crop_size]
            image_cropped = normalize_image(image_cropped)
            images.append(image_cropped)
    images = np.asarray(images)

    return images, (new_H // crop_size, new_W // crop_size)


def predict_images(images, model):
    images = np.expand_dims(images, axis=3)
    pred_images = model.predict(images)
    return pred_images


def merge_predictions(pred_images, shape, crop_size=128):
    full_image = np.zeros((crop_size * shape[0], crop_size * shape[1]))
    for row in range(shape[0]):
        for col in range(shape[1]):
            pred_image = pred_images[row * shape[1] + col]
            full_image[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size] = pred_image
    return full_image


if __name__ == '__main__':
    # Load input image
    image_path = '../data/DIPCO2016_dataset/8.bmp'
    image = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)

    # Split image to fit model's input requirements
    images, shape = split_image(image)

    # Load the model
    model_path = '../models/'
    mode_name = 'UNet.h5'
    UNet = load_model(os.path.join(model_path, 'UNet_2.h5'), custom_objects={'jaccard_loss': jaccard_loss,
                                                                             'jaccard_accuracy': jaccard_accuracy,
                                                                             'dice_loss': dice_loss,
                                                                             'dice_accuracy': dice_accuracy})

    # Get binary images
    pred_images = np.squeeze(predict_images(images, UNet))

    # Merge binary images
    binary_image = merge_predictions(pred_images, shape)
    plt.imshow(binary_image, cmap='gray')
    plt.show()

