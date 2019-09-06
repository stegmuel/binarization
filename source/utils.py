import numpy as np
from random import shuffle
import os
import shutil
import keras.backend as K
import matplotlib.pyplot as plt


def jaccard_accuracy(y_true, y_pred):
    eps = 1.0
    num = K.sum(y_true * y_pred) + 1.0
    den = K.sum(y_true + y_pred) - num + 1.0
    return num / den


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard_accuracy(y_true, y_pred)


def dice_accuracy(y_true, y_pred):
    eps = 1.0
    num = 2 * K.sum(y_true * y_pred) + 1.0
    den = K.sum(y_true + y_pred) + 1.0
    return num / den


def dice_loss(y_true, y_pred):
    return 1 - dice_accuracy(y_true, y_pred)


def normalize_image(image, gt_bool=False):
    image = image.astype(np.float32)
    image /= 255.0
    if not gt_bool:
        image -= 0.5
    return image


def get_images_names(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    shuffle(lines)
    images_names = [line.strip('\n').split(' ')[0] for line in lines]
    images_gt_names = [line.strip('\n').split(' ')[1] for line in lines]
    return images_names, images_gt_names


def split_data_list(data_path, original_list_path, ratio=0.9):
    with open(os.path.join(data_path, original_list_path)) as f:
        lines = f.readlines()
    shuffle(lines)
    train_stop = int(ratio*len(lines))
    with open(os.path.join(data_path, 'full_train.lst'), 'w') as f:
        [f.write(line) for line in lines[:train_stop]]
    with open(os.path.join(data_path, 'full_validation.lst'), 'w') as f:
        [f.write(line) for line in lines[train_stop:]]


def prepare_training(data_dir, train_list_path, validation_list_path, pairs_num, ratio=0.9):
    train_dir = '../training/train/'
    validation_dir = '../training/validation/'

    if os.path.exists('../training.zip'):
        os.remove('../training.zip')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)

    train_images_names, train_images_gt_names = get_images_names(train_list_path)
    validation_images_names, validation_images_gt_names = get_images_names(validation_list_path)
    train_stop = int(ratio * pairs_num)
    validation_stop = int((1-ratio) * pairs_num)
    train_pairs = zip(train_images_names[:train_stop], train_images_gt_names[:train_stop])
    validation_pairs = zip(validation_images_names[:validation_stop], validation_images_gt_names[:validation_stop])

    train_list = open('../training/train.lst', 'w')
    for image_name, image_gt_name in train_pairs:
        train_list.write(image_name + ' ' + image_gt_name + '\n')
        image_path = os.path.join(data_dir, image_name)
        image_gt_path = os.path.join(data_dir, image_gt_name)
        shutil.copy(image_path, os.path.join(train_dir, image_name))
        shutil.copy(image_gt_path, os.path.join(train_dir, image_gt_name))
    train_list.close()

    validation_list = open('../training/validation.lst', 'w')
    for image_name, image_gt_name in validation_pairs:
        validation_list.write(image_name + ' ' + image_gt_name + '\n')
        image_path = os.path.join(data_dir, image_name)
        image_gt_path = os.path.join(data_dir, image_gt_name)
        shutil.copy(image_path, os.path.join(validation_dir, image_name))
        shutil.copy(image_gt_path, os.path.join(validation_dir, image_gt_name))
    validation_list.close()

    # zip training directory
    shutil.make_archive('../training', 'zip', '../training')


def prepare_training_images(data_dir, train_list_path, validation_list_path, pairs_num, ratio=0.9):
    train_images_dir = '../training/train_images/'
    train_images_gt_dir = '../training/train_images_gt/'
    validation_images_dir = '../training/validation_images/'
    validation_images_gt_dir = '../training/validation_images_gt/'

    # Prepare directories
    if os.path.exists(train_images_dir):
        shutil.rmtree(train_images_dir)
    if os.path.exists(train_images_gt_dir):
        shutil.rmtree(train_images_gt_dir)
    if os.path.exists(validation_images_dir):
        shutil.rmtree(validation_images_dir)
    if os.path.exists(validation_images_gt_dir):
        shutil.rmtree(validation_images_gt_dir)

    os.mkdir(train_images_dir)
    os.mkdir(train_images_gt_dir)
    os.mkdir(validation_images_dir)
    os.mkdir(validation_images_gt_dir)

    # Get images' names
    train_images_names, train_images_gt_names = get_images_names(train_list_path)
    validation_images_names, validation_images_gt_names = get_images_names(validation_list_path)
    train_stop = int(ratio * pairs_num)
    validation_stop = int((1-ratio) * pairs_num)
    train_pairs = zip(train_images_names[:train_stop], train_images_gt_names[:train_stop])
    validation_pairs = zip(validation_images_names[:validation_stop], validation_images_gt_names[:validation_stop])

    # Create training list and move data
    train_list = open('../training/train_images.lst', 'w')
    for image_name, image_gt_name in train_pairs:
        train_list.write(image_name.split('.')[0] + '.jpg ' + image_gt_name.split('.')[0] + '.jpg\n')
        image_path = os.path.join(data_dir, image_name)
        image_gt_path = os.path.join(data_dir, image_gt_name)
        image = np.load(image_path)
        image_gt = np.load(image_gt_path)
        plt.imsave(os.path.join(train_images_dir, image_name.split('.')[0] + '.jpg'), image)
        plt.imsave(os.path.join(train_images_gt_dir, image_name.split('.')[0] + '.jpg'), image_gt)
    train_list.close()

    validation_list = open('../training/validation_images.lst', 'w')
    for image_name, image_gt_name in validation_pairs:
        validation_list.write(image_name.split('.')[0] + '.jpg ' + image_gt_name.split('.')[0] + '.jpg\n')
        image_path = os.path.join(data_dir, image_name)
        image_gt_path = os.path.join(data_dir, image_gt_name)
        image = np.load(image_path)
        image_gt = np.load(image_gt_path)
        plt.imsave(os.path.join(validation_images_dir, image_name.split('.')[0] + '.jpg'), image)
        plt.imsave(os.path.join(validation_images_gt_dir, image_name.split('.')[0] + '.jpg'), image_gt)
    validation_list.close()



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


def split_image(image, crop_size=128):
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
    padding = {'upper': upper_padding,
               'lower': lower_padding,
               'left': left_padding,
               'right': right_padding}
    images = []
    for row in range(0, new_H, crop_size):
        for col in range(0, new_W, crop_size):
            image_cropped = image[row:row+crop_size, col:col+crop_size]
            image_cropped = normalize_image(image_cropped)
            images.append(image_cropped)
    images = np.asarray(images)

    return images, (new_H // crop_size, new_W // crop_size), padding


def merge_predictions(pred_images, shape, crop_size=128):
    full_image = np.zeros((crop_size * shape[0], crop_size * shape[1]))
    for row in range(shape[0]):
        for col in range(shape[1]):
            pred_image = pred_images[row * shape[1] + col]
            full_image[row * crop_size: (row + 1) * crop_size, col * crop_size: (col + 1) * crop_size] = pred_image
    full_image[full_image > 0.5] = 255
    full_image[full_image <= 0.5] = 0
    return full_image.astype(np.uint8)
