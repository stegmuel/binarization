from models import *
from classes import *
from keras.optimizers import Adam
from keras.models import load_model
import zipfile


if __name__ == '__main__':
    data_dir = '../data/train_data'
    train_list_path = '../data/full_train.lst'
    validation_list_path = '../data/full_validation.lst'
    prepare_training(data_dir, train_list_path, validation_list_path, 1000, ratio=0.9)

    data_path = '../'
    models_path = '../models/'
    training_path = os.path.join(data_path, 'training/')

    train_images_names, train_images_gt_names = get_images_names(os.path.join(training_path, 'train.lst'))
    validation_images_names, validation_images_gt_names = \
        get_images_names(os.path.join(training_path, 'validation.lst'))

    train_generator = DataGenerator(train_images_names, train_images_gt_names, 10,
                                    os.path.join(data_path, 'training/train/'))
    validation_generator = DataGenerator(validation_images_names, validation_images_gt_names, 10,
                                         os.path.join(data_path, 'training/validation'))

    if os.path.exists(os.path.join(models_path, 'UNet_2.h5')):
        UNet = load_model(os.path.join(models_path, 'UNet_2.h5'), custom_objects={'jaccard_loss': jaccard_loss,
                                                                                  'jaccard_accuracy': jaccard_accuracy,
                                                                                  'dice_loss': dice_loss,
                                                                                  'dice_accuracy': dice_accuracy})
    else:
        UNet = unet_learned_up()
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
    UNet.save(os.path.join(models_path, 'UNet_2.h5'))
