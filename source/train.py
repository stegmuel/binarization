from models import *
from classes import *
from keras.optimizers import Adam
import argparse
from keras.models import load_model


if __name__ == '__main__':
    # Parse arguments from user
    parser = argparse.ArgumentParser(description='Train a UNet model to binarize images.')
    parser.add_argument('images_dir', metavar='images_dir', type=str, help='Path to train images.')
    parser.add_argument('ground_truth_dir', metavar='ground_truth_dir', type=str,
                        help='Path to ground truth images having the same name as the train images.')
    parser.add_argument('models_path', metavar='models_path', type=str, help='Path to the models directory.')
    parser.add_argument('train_size', metavar='train_size', type=int,
                        help='Number of images to be used during a training session')
    parser.add_argument('epochs', metavar='epochs', type=int,
                        help='Number of times the model will train over all train images')
    parser.add_argument('batch_size', metavar='batch_size', type=int,
                        help='number of images that passes through the network before one back-propagation. '
                             'Larger batch size leads to more reliable gradient steps but one batch must fit in memory.'
                             'Choose batch_size such that it divides the number of images.')
    parser.add_argument('output_model', metavar='output_model', type=str,
                        help='Name under which to save the newly trained model.')
    parser.add_argument('input_model', metavar='input_model', nargs='?', type=str,
                        help='Name of a pre trained model to resume the training from.')
    args = parser.parse_args()

    # Define data directories
    images_dir = args.images_dir
    images_gt_dir = args.ground_truth_dir
    models_path = args.models_path
    image_dim = 128

    # Read images' name from directory and split between training and validation
    image_name_lst = [image_name for image_name in os.listdir(images_dir)]
    train_limit = int(0.9 * len(image_name_lst))
    train_lst = image_name_lst[:train_limit]
    validation_lst = image_name_lst[train_limit:]
    shuffle(train_lst)
    current_train_lst = train_lst[:int(0.9 * args.train_size)]
    current_validation_lst = validation_lst[:int(0.1 * args.train_size)]

    train_generator = ImageGenerator(current_train_lst, args.batch_size, images_dir, images_gt_dir, image_dim)
    validation_generator = ImageGenerator(current_validation_lst, args.batch_size, images_dir, images_gt_dir, image_dim)

    if args.input_model:
        UNet = load_model(os.path.join(models_path, args.input_model),
                          custom_objects={'jaccard_accuracy': jaccard_accuracy, 'jaccard_loss': jaccard_loss,
                                          'dice_accuracy': dice_accuracy, 'dice_loss': dice_loss})
    else:
        UNet = unet_learned_up()
        UNet.compile(optimizer=Adam(), loss=jaccard_loss, metrics=['accuracy', jaccard_accuracy, dice_accuracy])

    UNet.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.__len__(),
        validation_data=validation_generator,
        validation_steps=validation_generator.__len__(),
        epochs=args.epochs,
        shuffle=True,
        use_multiprocessing=True,
        workers=4,
    )
    UNet.save(os.path.join(models_path, args.output_model))