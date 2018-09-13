import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def batch_norm(x, name='batch_norm'):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(x, epsilon=1e-5)


def conv2d(inputs, out_dim, fs=4, s=2, padding='same', name='conv2d'):
    with tf.variable_scope(name):
        return tf.layers.conv2d(inputs=inputs, filters=out_dim, 
                                kernel_size=fs, strides=(s,s), padding=padding)


def deconv2d(inputs, out_dim, fs=4, s=2, padding='same', name='deconv2d'):
    with tf.variable_scope(name):
        return tf.layers.conv2d_transpose(inputs=inputs, filters=out_dim, 
                                          kernel_size=fs, strides=(s,s),
                                          padding=padding)


def resnet_block(x, num_features):
    """A layer that consists of 2 convolutional layers where a residue of input 
    is added to the output. CycleGAN code has two differences from typical 
    residual blocks:
    1) They use instance normalization instead of batch normalization
    2) They are missing the final ReLU nonlinearity."""
    res1 = tf.layers.conv2d(x, num_features, [3, 3], [1, 1], padding='same')
    res2 = tf.layers.conv2d(res1, num_features, [3, 3], [1, 1], padding='same')

    return res2 + x


def generator(x, name='generator'):
    """Builds the generator that consists of an encoder, 
    a transformer and a decoder."""
    with tf.variable_scope(name):
        with tf.variable_scope('encoder'):
            #TODO activation functions
            enc1 = tf.layers.conv2d(x, 32, [3, 3], [1, 1])
            enc_out = tf.layers.conv2d(enc1, 256, [2, 2], [4, 4])
        with tf.variable_scope('transformer'):
            res1 = resnet_block(enc_out, 256)
            res2 = resnet_block(res1, 256)
            res3 = resnet_block(res2, 256)
            res4 = resnet_block(res3, 256)
            res5 = resnet_block(res4, 256)
            res_out = resnet_block(res5, 256)
        with tf.variable_scope('decoder'):
            dec1 = tf.layers.conv2d_transpose(res_out, 64, [2, 2], [2, 2])
            dec2 = tf.layers.conv2d_transpose(dec1, 32, [1, 1], [1, 1])
            gen_out = tf.layers.conv2d_transpose(dec2, 3, [2, 2], [2, 2])
    
    return gen_out


def discriminator(x, name='discriminator'):
    """Builds the discriminator which is a fully-convolutional 
    network."""
    with tf.variable_scope(name):
        #TODO change activation functions
        h1 = lrelu(conv2d(x, 64, [4, 4], [2, 2]))
        h2 = lrelu(tf.layers.conv2d(h1, 128, [4, 4], [2, 2]))
        h3 = lrelu(tf.layers.conv2d(h2, 256, [4, 4], [2, 2]))
        h4 = lrelu(tf.layers.conv2d(h3, 512, [4, 4], [2, 2]))
        out = tf.layers.conv2d(h4, 1, [4, 4], [1, 1])

    return out
