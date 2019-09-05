from keras.utils import Sequence
from copy import deepcopy
import os
from utils import *
import Augmentor
from PIL import Image


class ImageGenerator(Sequence):
    def __init__(self, image_name_lst, batch_size, images_dir, images_gt_dir, image_dim):
        self.image_name_lst = image_name_lst
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.images_gt_dir = images_gt_dir
        self.indexes = np.array([i for i in range(len(self.image_name_lst))])
        self.image_dim = image_dim

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

        images_pairs = []
        for i in range(start, stop):
            image_path = os.path.join(self.images_dir, self.image_name_lst[self.indexes[i]])
            image_gt_path = os.path.join(self.images_gt_dir, self.image_name_lst[self.indexes[i]])

            image = np.asarray(Image.open(image_path).convert('L'), dtype=np.uint8)
            image_gt = np.asarray(Image.open(image_gt_path).convert('L'), dtype=np.uint8)
            images_pairs.append([image, image_gt])

        p = Augmentor.DataPipeline(images_pairs)
        p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_top_bottom(probability=0.5)
        p.crop_by_size(probability=1, width=self.image_dim, height=self.image_dim, centre=False)
        p.resize(probability=1, width=self.image_dim, height=self.image_dim)
        augmented_images = p.sample(self.batch_size)
        images = np.asarray([np.expand_dims(normalize_image(augmented_image[0]), axis=2)
                             for augmented_image in augmented_images])
        images_gt = np.asarray([np.expand_dims(normalize_image(augmented_image[1], True), axis=2)
                                for augmented_image in augmented_images])
        return images, images_gt


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
