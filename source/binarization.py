from keras.models import load_model
import os
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


def split_image(image):
    shape = image.shape
    H = image.shape[0]
    W = image.shape[1]
    new_H = 128 * (H // 128 + 1)
    new_W = 128 * (W // 128 + 1)
    upper_padding = (new_H - H) // 2
    lower_padding = new_H - H - upper_padding
    left_padding = (new_W - W) // 2
    right_padding = new_W - W - left_padding
    image = np.pad(image, ((upper_padding, lower_padding), (left_padding, right_padding)), 'constant',
                   constant_values=np.median(image))
    images = []
    for row in range(0, new_H, 128):
        for col in range(0, new_W, 128):
            image_cropped = image[row:row+128, col:col+128]
            images.append(image_cropped)
    images = np.asarray(images)
    return images


if __name__ == '__main__':
    image_path = '../data/DIPCO2016_dataset/1.bmp'
    image = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)
    images = split_image(image)
    print(images.shape)

    plt.imshow(images[50])
    plt.show()

    # model_path = '../models/'
    # mode_name = 'UNet.h5'
    # UNet = load_model(os.path.join(model_path, 'UNet_2.h5'), custom_objects={'jaccard_loss': jaccard_loss,
    #                                                                        'jaccard_accuracy': jaccard_accuracy,
    #                                                                        'dice_loss': dice_loss,
    #                                                                        'dice_accuracy': dice_accuracy})
    #
    # images_list_path = '../training/validation.lst'
    # data_dir = '../training/validation/'
    # images_names, images_gt_names = get_images_names(images_list_path)
    # index = 0
    # image = np.load(os.path.join(data_dir, images_names[index]))
    # image = normalize_image(image)
    # image_gt = np.load(os.path.join(data_dir, images_gt_names[index]))
    # image_gt = normalize_image(image_gt, gt_bool=True)
    # input_image = np.expand_dims(np.expand_dims(image, axis=2), axis=0)
    #
    # # Get model's prediction
    # pred_image = UNet.predict(input_image)
    # accuracy = jaccard_accuracy(pred_image, np.expand_dims(np.expand_dims(image_gt, axis=2), axis=0))
    # with tf.Session() as sess:
    #     print(accuracy.eval())
    #
    # pred_image = np.squeeze(pred_image)
    # pred_image[pred_image > 0.5] = 1.0
    # pred_image[pred_image < 0.5] = 0.0
    #
    # # Plots
    # fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    # axes[0].imshow(image, cmap='gray')
    # axes[1].imshow(image_gt, cmap='gray')
    # axes[2].imshow(pred_image, cmap='gray')
    # plt.show()

