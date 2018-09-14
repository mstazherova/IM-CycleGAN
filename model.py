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


def resnet_block(x, out_dim, fs=3, s=1, p=1, mode='REFLECT', name='res'):
    """A layer that consists of 2 convolutional layers where a residue of input 
    is added to the output. CycleGAN code has two differences from typical 
    residual blocks:
    1) They use instance normalization instead of batch normalization
    2) They are missing the final ReLU nonlinearity."""
    # TODO add instance norm
    paddings = [[0, 0], [p, p], [p, p], [0, 0]]
    y = tf.pad(x, paddings, mode=mode)
    y = conv2d(y, out_dim, fs, s, padding='valid', name=name + '_c1')
    y = tf.pad(tf.nn.relu(y), paddings, mode=mode)
    y = conv2d(y, out_dim, fs, s, padding='valid', name=name + '_c2')

    return y + x


def generator(x, name='generator'):
    """Builds the generator that consists of an encoder, 
    a transformer and a decoder."""
    with tf.variable_scope(name):
        with tf.variable_scope('encoder'):
            #TODO change activation functions, add batch norm
            enc = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            enc1 = tf.nn.relu(conv2d(enc, 32, 7, 1, padding='valid'))
            enc2 = tf.nn.relu(conv2d(enc1, 64, 3, 2))
            enc_out = tf.nn.relu(conv2d(enc2, 128, 3, 2))
        with tf.variable_scope('transformer'):
            res1 = resnet_block(enc_out, 128)
            res2 = resnet_block(res1, 128)
            res3 = resnet_block(res2, 128)
            res4 = resnet_block(res3, 128)
            res5 = resnet_block(res4, 128)
            res_out = resnet_block(res5, 128)
        with tf.variable_scope('decoder'):
            dec1 = tf.nn.relu(deconv2d(res_out, 64, 3, 2))
            dec2 = tf.nn.relu(deconv2d(dec1, 32, 3, 2))
            dec2 = tf.pad(dec2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            gen_out = tf.nn.tanh(conv2d(dec2, 3, 7, 1, padding='valid'))
    
    return gen_out


def discriminator(x, name='discriminator'):
    """Builds the discriminator which is a fully-convolutional 
    network."""
    with tf.variable_scope(name):
        #TODO add batch normalization
        h1 = lrelu(conv2d(x, 64, [4, 4], [2, 2]))
        h2 = lrelu(tf.layers.conv2d(h1, 128, [4, 4], [2, 2]))
        h3 = lrelu(tf.layers.conv2d(h2, 256, [4, 4], [2, 2]))
        h4 = lrelu(tf.layers.conv2d(h3, 512, [4, 4], [2, 2]))
        out = tf.layers.conv2d(h4, 1, [4, 4], [1, 1])

    return out
