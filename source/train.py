from models import *
from classes import *
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
import zipfile


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
    data_path = '../'
    models_path = '../models/'
    training_path = os.path.join(data_path, 'training/')

    # Unzip training data
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    os.mkdir(training_path)

    with zipfile.ZipFile(os.path.join(data_path, 'training.zip'), 'r') as zip_ref:
        zip_ref.extractall(training_path)

    train_images_names, train_images_gt_names = get_images_names(os.path.join(training_path, 'train.lst'))
    validation_images_names, validation_images_gt_names = \
        get_images_names(os.path.join(training_path, 'validation.lst'))

    train_generator = DataGenerator(train_images_names, train_images_gt_names, 10,
                                    os.path.join(data_path, 'training/train/'))
    validation_generator = DataGenerator(validation_images_names, validation_images_gt_names, 10,
                                         os.path.join(data_path, 'training/validation'))

    if os.path.exists(os.path.join(models_path, 'UNet.h5')):
        UNet = load_model(os.path.join(models_path, 'UNet.h5'),custom_objects={'jaccard_loss': jaccard_loss,
                                                                               'jaccard_accuracy': jaccard_accuracy,
                                                                               'dice_loss': dice_loss,
                                                                               'dice_accuracy': dice_accuracy})
    else:
        UNet = unet()
        UNet.compile(optimizer=Adam(), loss=jaccard_loss, metrics=['accuracy', jaccard_accuracy, dice_accuracy])


    UNet.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),  # Compatibility with old Keras versions.
        validation_data=validation_generator,
        validation_steps=validation_generator.__len__(),  # Compatibility with old Keras versions.
        epochs=1,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
    )
    UNet.save(os.path.join(models_path, 'UNet.h5'))