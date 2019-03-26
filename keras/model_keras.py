"""Defines layers and models."""

from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, ZeroPadding2D, Cropping2D
from keras.layers import Input, Activation, Add, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


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
    """Convolutional block."""
    x = conv2d(x, filters, (size, size), strides=stride, padding=padding)
    x = batchnorm()(x)
    if not use_leaky_relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)

    return x


def resnet_block(x, filters=256, padding='same'):
    """Residual block."""
    y = conv2d(x, filters, kernel_size=3, strides=1, padding=padding)
    y = LeakyReLU(alpha=0.2)(x)
    y = conv2d(x, filters, kernel_size=3, strides=1, padding=padding)

    return Add()([y, x])


def up_block(x, filters, size):
    """Deconvolution layer."""
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


def unet_generator(size=img_width, in_ch=img_depth, out_ch=img_depth, nf=64):
    """Builds a pseudo unet generator."""
    max_nf = 8 * nf
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same")(x)
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

    s = size

    y = inputs = Input(shape=(s, s, in_ch))
    y = block(y, size, in_ch, True, nf_out=out_ch, nf_next=nf)
    y = Activation('tanh')(y)

    return Model(inputs=inputs, outputs=[y])


def unet(img_rows = img_height, img_cols=img_width, channels=img_depth):
    """mplements a U-Net. The number of filters for the convolutional layers
    starts from 64 and is doubled every next layer up to a maximum of 512."""
    inputs = Input(shape=(img_rows, img_cols, channels))
    enc1 = conv2d(64, kernel_size=4, strides=2, padding ="same")(inputs)
    enc1 = batchnorm()(enc1, training=1)
    enc1 = LeakyReLU(alpha=0.2)(enc1)

    enc2 = conv2d(128, kernel_size=4, strides=2, padding = 'same')(enc1)
    enc2 = batchnorm()(enc2, training=1)
    enc2 = LeakyReLU(alpha=0.2)(enc2)

    enc3 = conv2d(256, kernel_size=4, strides=2, padding = 'same')(enc2)
    enc3 = batchnorm()(enc3, training=1)
    enc3 = LeakyReLU(alpha=0.2)(enc3)

    enc4 = conv2d(512, kernel_size=4, strides=2, padding = 'same')(enc3)
    enc4 = batchnorm()(enc4, training=1)
    enc4 = LeakyReLU(alpha=0.2)(enc4)
    drop4 = Dropout(0.5)(enc4)

    bottle = conv2d(1024, kernel_size=4, strides=2, padding = 'same')(drop4)
    bottle = batchnorm()(bottle, training=1)
    bottle = LeakyReLU(alpha=0.2)(bottle)
    drop5 = Dropout(0.5)(bottle)

    up5 = Conv2DTranspose(512, kernel_size=4, strides=2, kernel_initializer = conv_init)(drop5)
    up5 = Activation("relu")(up5)
    up5 = Cropping2D(1)(up5)
    merge5 = Concatenate(axis=-1)([enc4, up5])

    up6 = Conv2DTranspose(256, kernel_size=4, strides=2, kernel_initializer = conv_init)(merge5)
    up6 = Activation("relu")(up6)
    up6 = Cropping2D(1)(up6)
    merge6 = Concatenate(axis=-1)([enc3, up6])

    up7 = Conv2DTranspose(128, kernel_size=4, strides=2, kernel_initializer = conv_init)(merge6)
    up7 = Activation("relu")(up7)
    up7 = Cropping2D(1)(up7)
    merge7 = Concatenate(axis=-1)([enc2, up7])

    up8 = Conv2DTranspose(64, kernel_size=4, strides=2, kernel_initializer = conv_init)(merge7)
    up8 = Activation("relu")(up8)
    up8 = Cropping2D(1)(up8)
    merge8 = Concatenate(axis=-1)([enc1, up8])

    up9 = Conv2DTranspose(3, kernel_size=4, strides=2, kernel_initializer = conv_init)(merge8)
    up9 = Cropping2D(1)(up9)

    out = Activation('tanh')(up9)

    model = Model(inputs=inputs, outputs=[out])

    return model
