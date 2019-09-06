from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def conv_block(X, num_output_channels):
    X = Conv2D(num_output_channels, (3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = Conv2D(num_output_channels, (3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = SpatialDropout2D(0.1)(X)
    return X


def down_block(X, num_output_channels):
    X = conv_block(X, num_output_channels)
    pool = MaxPooling2D(pool_size=(2, 2))(X)
    return X, pool


def up_block(X, X_shortcut, num_output_channels):
    X = UpSampling2D(size=(2, 2))(X)
    X = concatenate([X, X_shortcut], axis=3)
    X = conv_block(X, num_output_channels)
    return X


def up_block_learned(X, X_shortcut, num_filters, num_output_channels):
    X = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='valid')(X)
    X = concatenate([X, X_shortcut], axis=3)
    X = conv_block(X, num_output_channels)
    return X


def unet():
    X_input = Input((128, 128, 1))

    # Downsampling.
    down1, X = down_block(X_input, 32)
    down2, X = down_block(X, 64)
    down3, X = down_block(X, 128)
    down4, X = down_block(X, 256)
    down5, X = down_block(X, 512)

    # Bottleneck.
    bottleneck = conv_block(X, 1024)

    # Upsampling.
    X = up_block(bottleneck, down5, 512)
    X = up_block(X, down4, 256)
    X = up_block(X, down3, 128)
    X = up_block(X, down2, 64)
    X = up_block(X, down1, 32)

    X = Conv2D(1, (1, 1))(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)
    return model


def unet_learned_up():
    X_input = Input((128, 128, 1))

    # Downsampling.
    down1, X = down_block(X_input, 32)
    down2, X = down_block(X, 64)
    down3, X = down_block(X, 128)
    down4, X = down_block(X, 256)
    down5, X = down_block(X, 512)

    # Bottleneck.
    bottleneck = conv_block(X, 1024)

    # Upsampling.
    X = up_block_learned(bottleneck, down5, 256, 512)
    X = up_block_learned(X, down4, 128, 256)
    X = up_block_learned(X, down3, 64, 128)
    X = up_block_learned(X, down2, 32, 64)
    X = up_block_learned(X, down1, 16, 32)

    X = Conv2D(1, (1, 1))(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)
    return model