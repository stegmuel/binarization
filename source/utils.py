import numpy as np
from random import shuffle
import os
import shutil
import keras.backend as K


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


def prepare_training(data_dir, train_list_path, validation_list_path, num_pairs, ratio=0.9):
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
    train_stop = int(ratio * num_pairs)
    validation_stop = int((1-ratio) * num_pairs)
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


