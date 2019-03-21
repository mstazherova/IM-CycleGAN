"""Defines layers and models."""

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, ZeroPadding2D, Cropping2D
from keras.layers import Input, Activation, Add, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


# ngf = 32  # Number of filters in first layer of generator
# ndf = 64  # Number of filters in first layer of discriminator
batch_size = 1
img_width = 256
img_height = 256
img_depth = 3

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

def batchnorm():
    """Batch normalization layer."""
    return BatchNormalization(momentum=0.9, axis=3, epsilon=1e-5,
                              gamma_initializer = gamma_init)


def conv2d(*a, **k):
    """2D convolutional layer."""
    return Conv2D(kernel_initializer = conv_init, *a, **k)


def conv_block(x, filters, size, stride=(2, 2),
               use_leaky_relu=False, padding='same'):
    # TODO try out instance normalization
    x = conv2d(x, filters, (size, size), strides=stride, padding=padding)
    x = batchnorm()(x)
    if not use_leaky_relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)

    return x


def resnet_block(x, filters=256, padding='same'):
    y = conv2d(x, filters, kernel_size=3, strides=1, padding=padding)
    y = LeakyReLU(alpha=0.2)(x)
    y = conv2d(x, filters, kernel_size=3, strides=1, padding=padding)

    return Add()([y, x])


def up_block(x, filters, size):
    x = Conv2DTranspose(filters, kernel_size=size, strides=2, padding='same',
                        use_bias=False,
                        kernel_initializer=RandomNormal(0, 0.02))(x)
    x = batchnorm()(x)
    x = Activation('relu')(x)

    return x


def discriminator(channels=img_depth, ndf=64, hidden_layers=3):
    """Builds a fully-convolutional discriminator.

    ndf: filters of the first layer."""
    inputs = Input(shape=(None, None, channels))
    x = inputs
    x = conv2d(x, ndf, kernel_size=4, strides=2, padding="same")
    x = LeakyReLU(alpha=0.2)(x)

    for layer in range(1, hidden_layers):
        out_feat = 2 ** layer * ndf
        x = conv2d(x, out_feat, kernel_size=4, strides=2, padding="same", use_bias=False)
        x = batchnorm()(x, training=1) # training parameter?
        x = LeakyReLU(alpha=0.2)(x)

    out_feat = ndf * 2 ** hidden_layers
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv2d(x, out_feat, kernel_size=4,  use_bias=False)
    x = batchnorm()(x, training=1)
    x = LeakyReLU(alpha=0.2)(x)

    # final layer
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = conv2d(x, 1, kernel_size=4, activation = "sigmoid")

    return Model(inputs=[inputs], outputs=x)


def patch_discriminator(w=img_width, h=img_height, ndf=64):
    """Simple convolutional discriminator.

    Is implementing the 70x70 PatchGAN."""

    n_conv = 3  # hidden layers

    inp = Input(shape=(h, w, 3))
    x = inp

    for depth in range(n_conv):
        x = Conv2D(ndf*(2**depth), kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(x)
        if depth != 0:
            x = batchnorm()(x, training=1)
        x = LeakyReLU(0.2)(x)

    # Last Conv
    x = ZeroPadding2D(1)(x)
    x = Conv2D(ndf * (2 ** n_conv), kernel_size=4, kernel_initializer=conv_init)(x)
    x = batchnorm()(x, training=1)
    x = LeakyReLU(0.2)(x)

    # Decision layer
    x = ZeroPadding2D(1)(x)
    out = Conv2D(1, kernel_size=4, kernel_initializer=conv_init, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)

    return model


def generator(image_size=img_width, channels=img_depth, res_blocks=6):
    """Builds a generator.

    Cnsists of an encoder, a transformer
    and a decoder."""
    inputs = Input(shape=(image_size, image_size, channels))
    x = inputs

    # Encoder
    x = conv_block(x, 64, 7, (1, 1))
    x = conv_block(x, 128, 3, (2, 2))
    x = conv_block(x, 256, 3, (2, 2))

    # Transformer
    for _ in range(res_blocks):
        x = resnet_block(x)

    # Decoder
    x = up_block(x, 128, 3)
    x = up_block(x, 64, 3)

    x = conv2d(x, 3, (7, 7), activation='tanh', strides=(1, 1) ,padding='same')
    outputs = x

    return Model(inputs=inputs, outputs=[outputs])


def unet_generator(isize=img_width, nc_in=img_depth, nc_out=img_depth, ngf=64):
    """Builds a generator.

    Implements a U-Net."""
    max_nf = 8*ngf
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s))(x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=-1)([x, x2])
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = RandomNormal(0, 0.02),
                            name = 'convt.{0}'.format(s))(x)
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)  # , training=1
        if s <=8:
            x = Dropout(0.5)(x, training=1)  # , training=1
        return x

    s = isize

    y = inputs = Input(shape=(s, s, nc_in))
    y = block(y, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    y = Activation('tanh')(y)
    return Model(inputs=inputs, outputs=[y])
