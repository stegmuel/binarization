from keras.models import load_model
import os
from utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path = '../models/'
    mode_name = 'UNet.h5'
    # UNet = load_model(os.path.join(model_path, 'UNet.h5'), custom_objects={'jaccard_loss': jaccard_loss,
    #                                                                        'jaccard_accuracy': jaccard_accuracy,
    #                                                                        'dice_loss': dice_loss,
    #                                                                        'dice_accuracy': dice_accuracy})

    images_list_path = '../data/full_validation.lst'
    data_dir = '../data/train_data/'
    images_names, images_gt_names = get_images_names(images_list_path)
    index = 0
    image = np.load(os.path.join(data_dir, images_names[index]))
    image_gt = np.load(os.path.join(data_dir, images_gt_names[index]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    axes[0].imshow(image, cmap='gray')
    axes[1].imshow(image_gt, cmap='gray')
    plt.show()

