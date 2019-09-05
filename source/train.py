from models import *
from classes import *
from keras.optimizers import Adam
from keras.models import load_model
import zipfile


if __name__ == '__main__':
    images_dir = '../data/new_data/image'
    images_gt_dir = '../data/new_data/image_gt'
    models_path = '../models/'
    model_name = 'UNet_3.h5'

    image_name_lst = [image_name for image_name in os.listdir(images_dir)]
    shuffle(image_name_lst)
    train_generator = ImageGenerator(image_name_lst[:1000], 10, images_dir, images_gt_dir, 128)
    validation_generator = ImageGenerator(image_name_lst[1000:1100], 10, images_dir, images_gt_dir, 128)
    images, images_gt = validation_generator.__getitem__(0)

    if os.path.exists(os.path.join(models_path, model_name)):
        UNet = load_model(os.path.join(models_path, model_name), custom_objects={'jaccard_accuracy': jaccard_accuracy,
                                                                                 'jaccard_loss': jaccard_loss,
                                                                                 'dice_accuracy': dice_accuracy,
                                                                                 'dice_loss': dice_loss})
    else:
        UNet = unet_learned_up()
        UNet.compile(optimizer=Adam(), loss=jaccard_loss, metrics=['accuracy', jaccard_accuracy, dice_accuracy])

    UNet.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        validation_data=validation_generator,
        validation_steps=validation_generator.__len__(),
        epochs=1,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
    )
    UNet.save(os.path.join(models_path, 'UNet_3.h5'))