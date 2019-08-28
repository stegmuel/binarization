import numpy as np
from random import shuffle
import os
import shutil


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


def prepare_training(data_dir, list_path, num_pairs, ratio=0.9):
    train_dir = '../training/train/'
    validation_dir = '../training/validation/'
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)

    lists_path = '../training/'

    images_names, images_gt_names = get_images_names(list_path)
    train_stop = int(ratio * num_pairs)
    train_pairs = zip(images_names[0:train_stop], images_gt_names[0:train_stop])
    validation_pairs = zip(images_names[train_stop:num_pairs], images_gt_names[train_stop:num_pairs])

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


