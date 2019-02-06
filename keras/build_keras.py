import os
import argparse
import time
import glob
import numpy as np

import tensorflow as tf

from model_keras import discriminator, generator, unet_generator
from utils_keras import disc_loss, gen_loss, cycle_loss, minibatchAB
from utils_keras import save_generator, save_plots

from keras import backend as K
from keras import optimizers


def build_model():
    d_a = discriminator()
    d_b = discriminator()
    g_a = unet_generator()
    g_b = unet_generator()
    real_a = g_b.inputs[0]
    fake_b = g_b.outputs[0]
    rec_a = g_a([fake_b])
    real_b = g_a.inputs[0]
    fake_a = g_a.outputs[0]
    rec_b = g_b([fake_a])

    d_a_loss = disc_loss(d_a, real_a, fake_a)
    d_b_loss = disc_loss(d_b, real_b, fake_b)
    g_a_loss = gen_loss(d_a, fake_a)
    g_b_loss = gen_loss(d_b, fake_b)

    cycleA_generate = K.function([real_a], [fake_b, rec_a])
    cycleB_generate = K.function([real_b], [fake_a, rec_b])

    cyc_loss = cycle_loss(rec_a, real_a) + cycle_loss(rec_b, real_b)
    # g_total_a = g_a_loss + 10*cyc_loss
    # g_total_b = g_b_loss + 10*cyc_loss
    g_total = g_a_loss + g_b_loss + 10 * cyc_loss
    d_total = d_a_loss + d_b_loss

    weights_d = d_a.trainable_weights + d_b.trainable_weights
    weights_g = g_a.trainable_weights + g_b.trainable_weights

    training_updates = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(d_total, weights_d)
    d_train_function = K.function([real_a, real_b], [d_a_loss, d_b_loss], training_updates)
    training_updates = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates(g_total, weights_g)
    g_train_function = K.function([real_a, real_b], [g_a_loss, g_b_loss, cyc_loss], training_updates)

    return  d_train_function, g_train_function, cycleA_generate, cycleB_generate


def main(arguments):
    # t0 = time.time()
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number

    SAVE_PATH = '/tmp/stazherova/generated/'

    if GPU == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(GPU_NUMBER)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # pylint: disable=no-member
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.Session(config=config)
        K.set_session(sess)

    parent_dir, _ = os.path.split(os.getcwd())
    trainA = glob.glob(os.path.join(parent_dir, 'data/trainA/*'))
    trainB = glob.glob(os.path.join(parent_dir, 'data/trainB/*'))

    DISPLAY_STEP = 500
    SAVE_STEP = 10
    SUMMARY_STEP = min(len(trainA), len(trainB))
    epoch = 0
    counter = 0

    steps_array = []

    d_a_losses = []
    d_b_losses = []
    g_a_losses = []
    g_b_losses = []

    d_train_function, g_train_function, cycleA_generate, cycleB_generate = build_model()

    train_batch = minibatchAB(trainA, trainB)
    
    while epoch < EPOCHS:
        epoch, A, B = next(train_batch)
        d_a_loss, d_b_loss  = d_train_function([A, B])
        g_a_loss, g_b_loss, _ = g_train_function([A, B])
        counter += 1

        if np.mod(counter, DISPLAY_STEP) == 0:
            print('[Epoch {}/{}][Iteration {}]...'.format(epoch, EPOCHS, counter))
            print('D_a_loss: {:.2f}, D_b_loss: {:.2f}'.format(d_a_loss, d_b_loss))
            print('G_a_loss: {:.2f}, G_b_loss: {:.2f}'.format(g_a_loss, g_b_loss))

        if np.mod(epoch, SAVE_STEP) == 0:
            print('Saving generated images...')
            save_generator(A, B, cycleA_generate, cycleB_generate, SAVE_PATH)

        if np.mod(counter, SUMMARY_STEP) == 0:
            print('Saving data for plots...')
            steps_array.append(counter)
            d_a_losses.append(d_a_loss)
            d_b_losses.append(d_b_loss)
            g_a_losses.append(g_a_loss)
            g_b_losses.append(g_b_loss)

        # TODO image pool
    
    print('Saving plots...')
    save_plots(steps_array, d_a_losses, d_b_losses, g_a_losses, g_b_losses)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=200, 
                        help='Number of epochs. Default:100')
    parser.add_argument('-gpu','--gpu', type=int, default=0,
                        help='If to use GPU. Default: 0')
    parser.add_argument('-number', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    args = parser.parse_args()

    main(args)
