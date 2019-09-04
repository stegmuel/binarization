from keras.utils import Sequence
from copy import deepcopy
import os
from utils import *


class DataGenerator(Sequence):
    """Generate images for training/validation/testing (parallel version)."""
    def __init__(self, images_names, images_gt_names, batch_size: int, train_data_dir):
        self.images_names = deepcopy(images_names)
        self.images_gt_names = deepcopy(images_gt_names)
        self.batch_size = batch_size
        self.indexes = np.array([i for i in range(len(self.images_names))])
        self.train_data_dir = train_data_dir

    def __len__(self):
        return int(np.ceil(float(self.indexes.shape[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Creating numpy arrays with images.
        start = index * self.batch_size
        stop = start + self.batch_size
        if stop >= self.indexes.shape[0]:
            stop = self.indexes.shape[0]

        images = []
        images_gt = []
        for i in range(start, stop):
            image = np.load(os.path.join(self.train_data_dir, self.images_names[self.indexes[i]]))
            images.append(image)
            image_gt = np.load(os.path.join(self.train_data_dir, self.images_gt_names[self.indexes[i]]))
            images_gt.append(image_gt)

        images = np.array([normalize_image(image) for image in images])
        images = np.expand_dims(images, axis=3)
        images_gt = np.array([normalize_image(image_gt, gt_bool=True) for image_gt in images_gt])
        images_gt = np.expand_dims(images_gt, axis=3)

        return images, images_gt
