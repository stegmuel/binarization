from models import *
from classes import *
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model


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


if __name__ == '__main__':
    train_data_dir = '../data/train_data/'
    file_path = '../data/train_data.lst'
    models_path = '../models/'
    images_names, images_gt_names = get_images_names(file_path)

    train_images_names = images_names[0:100]
    train_images_gt_names = images_gt_names[0:100]
    validation_images_names = images_names[100:110]
    validation_images_gt_names = images_gt_names[100:110]

    train_generator = DataGenerator(train_images_names, train_images_gt_names, 10, train_data_dir)
    validation_generator = DataGenerator(validation_images_names, validation_images_gt_names, 10, train_data_dir)

    prepare_training(train_data_dir, file_path, 1000, ratio=0.9)
    # if os.path.exists(os.path.join(models_path, 'UNet.h5')):
    #     UNet = load_model(os.path.join(models_path, 'UNet.h5'),custom_objects={'jaccard_loss': jaccard_loss,
    #                                                                            'jaccard_accuracy': jaccard_accuracy,
    #                                                                            'dice_loss': dice_loss,
    #                                                                            'dice_accuracy': dice_accuracy})
    # else:
    #     UNet = unet()
    #     UNet.compile(optimizer=Adam(), loss=jaccard_loss, metrics=['accuracy', jaccard_accuracy, dice_accuracy])

    #
    # UNet.fit_generator(
    #     generator=train_generator,
    #     steps_per_epoch=train_generator.__len__(),  # Compatibility with old Keras versions.
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.__len__(),  # Compatibility with old Keras versions.
    #     epochs=1,
    #     shuffle=True,
    #     use_multiprocessing=True,
    #     workers=4,
    # )
    # UNet.save(os.path.join(models_path, 'UNet.h5'))