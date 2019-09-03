import tensorflow as tf


class UNet:
    def __init__(self, input):
        self.input = input
        self.X = None
        self.output = None
        self.create()

    def create(self):
        # Down-sampling.
        down1, self.X = down_block(self.input, kernel_size=(3, 3), num_filters=32)
        down2, self.X = down_block(self.X, kernel_size=(3, 3), num_filters=64)
        down3, self.X = down_block(self.X, kernel_size=(3, 3), num_filters=128)
        down4, self.X = down_block(self.X, kernel_size=(3, 3), num_filters=256)
        down5, self.X = down_block(self.X, kernel_size=(3, 3), num_filters=512)

        # Bottleneck.
        bottleneck = conv_block(self.X, kernel_size=(3, 3), num_filters=1024)

        # Up-sampling.
        self.X = up_block(bottleneck, down5, filters_size=(3, 3), num_filters=512)
        self.X = up_block(self.X, down4, filters_size=(3, 3), num_filters=256)
        self.X = up_block(self.X, down3, filters_size=(3, 3), num_filters=128)
        self.X = up_block(self.X, down2, filters_size=(3, 3), num_filters=64)
        self.X = up_block(self.X, down1, filters_size=(3, 3), num_filters=32)

        self.X = tf.layers.conv2d(inputs=self.X, filters=1, kernel_size=(1, 1), padding='SAME', activation='sigmoid')
        self.output = self.X


def conv_block(X, kernel_size, num_filters):
    X = tf.layers.conv2d(inputs=X, filters=num_filters, kernel_size=kernel_size,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), padding='SAME',
                         activation='relu')
    X = tf.layers.batch_normalization(inputs=X, axis=3)
    X = tf.layers.conv2d(inputs=X, filters=num_filters, kernel_size=kernel_size,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), padding='SAME',
                         activation='relu')
    X = tf.layers.batch_normalization(inputs=X, axis=3)
    return X


def down_block(X, kernel_size, num_filters):
    X = tf.layers.conv2d(inputs=X, filters=num_filters, kernel_size=kernel_size,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), padding='SAME',
                         activation='relu')
    pool = tf.layers.max_pooling2d(inputs=X, pool_size=(2, 2), strides=(2, 2), padding='SAME')
    return X, pool


def up_conv(X, num_filters):
    X = tf.layers.conv2d_transpose(inputs=X, filters=num_filters, kernel_size=(2, 2), strides=(2, 2), padding='VALID',
                                   activation='relu')
    return X


def up_block(X, X_shortcut, filters_size, num_filters):
    X = up_conv(X, num_filters)
    X = tf.concat([X, X_shortcut], axis=3)
    X = conv_block(X, filters_size, num_filters)
    return X