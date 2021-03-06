"""Creating and training the network without cycle."""
import os
import argparse
import time
import glob
import random
import numpy as np

import tensorflow as tf

from model_keras import patch_discriminator, unet
from utils_keras import disc_loss, gen_loss
from utils_keras import save_onedir, save_plots_onedir, minibatchAB

from keras import backend as K
from keras import optimizers

img_width = 256
img_height = 256

def build_model(w=img_width, h=img_height):
    """Builds the model from A to B.

    Creates training functions for discriminator and generator networks."""
    fake_pool_b = K.placeholder(shape=(None, h, w, 3))
    real_b = K.placeholder(shape=(None, h, w, 3))

    d_b = patch_discriminator()
    g_b = unet()  # generator a2b
    real_a = g_b.inputs[0]
    fake_b = g_b.outputs[0]

    d_b_loss = disc_loss(d_b, real_b, fake_b, fake_pool_b)
    g_b_loss = gen_loss(d_b, fake_b)

    weights_d = d_b.trainable_weights
    weights_g = g_b.trainable_weights

    # Define optimizers
    adam_disc = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)
    adam_gen = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

    training_updates_disc = adam_disc.get_updates(weights_d, [], d_b_loss)  #pylint: disable=too-many-function-args
    d_train_function = K.function([real_a, real_b, fake_pool_b], [d_b_loss], training_updates_disc)
    training_updates_gen = adam_gen.get_updates(weights_g, [], g_b_loss)  #pylint: disable=too-many-function-args
    g_train_function = K.function([real_a, real_b], [g_b_loss], training_updates_gen)

    return  d_train_function, g_train_function, g_b, d_b, adam_disc, adam_gen


def main(arguments):
    """Main training loop."""
    t0 = time.time()
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number
    DATASET = arguments.dataset

    if GPU == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU_NUMBER)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # pylint: disable=no-member
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.Session(config=config)
        K.set_session(sess)

    parent_dir, _ = os.path.split(os.getcwd())
    if DATASET == 0:
        trainA = glob.glob(os.path.join(parent_dir, 'data/trainA/*'))
        trainB = glob.glob(os.path.join(parent_dir, 'data/trainB/*'))
    elif DATASET == 1:
        trainB = glob.glob(os.path.join(parent_dir, 'data/mm/no_glasses/*'))
        trainA = glob.glob(os.path.join(parent_dir, 'data/mm/glasses/*'))

    SAVE_PATH_TRAIN = os.path.join(parent_dir, 'results/ds{}-gen-train-onedir{}/'.format(DATASET, time.strftime('%Y%m%d-%H%M%S')))
    # SAVE_PATH_TEST = os.path.join(parent_dir, 'results/dataset{}-generated-test{}/'.format(DATASET, time.strftime('%Y%m%d-%H%M%S')))
    DISPLAY_STEP = 500

    SUMMARY_STEP = min(len(trainA), len(trainB))
    epoch = 0
    counter = 0

    steps_array = []

    d_b_losses = []
    g_b_losses = []

    d_trainer, g_trainer, g_b, d_b, adam_disc, adam_gen = build_model()

    train_batch = minibatchAB(trainA, trainB)

    # Initialize fake images pools
    pool_a2b = []

    while epoch < EPOCHS:
        epoch, A, B = next(train_batch)

        # Learning rate decay
        if epoch < 100:
            lr = 2e-4
        else:
            lr = 2e-4 - (2e-4 * (epoch - 100) / 100)
        adam_disc.lr = lr
        adam_gen.lr = lr

        # Fake images pool
        a2b = g_b.predict(A)

        tmp_a2b = []

        for element in a2b:
            if len(pool_a2b) < 50:
                pool_a2b.append(element)
                tmp_a2b.append(element)
            else:
                p = random.uniform(0, 1)

                if p > 0.5:
                    index = random.randint(0, 49)
                    tmp = np.copy(pool_a2b[index])
                    pool_a2b[index] = element
                    tmp_a2b.append(tmp)
                else:
                    tmp_a2b.append(element)

        pool_b = np.array(tmp_a2b)

        d_b_loss = d_trainer([A, B, pool_b])
        g_b_loss= g_trainer([A, B])

        counter += 1

        if np.mod(counter, DISPLAY_STEP) == 0:
            print('[Epoch {}/{}][Iteration {}]...'.format(epoch, EPOCHS, counter))
            print('D_b_loss: {:.2f}'.format(d_b_loss[0]))
            print('G_b_loss: {:.2f}'.format(g_b_loss[0]))
            print('Saving generated training images...')
            save_onedir(A, g_b, SAVE_PATH_TRAIN, epoch, counter)

        if np.mod(counter, SUMMARY_STEP) == 0:
            print('Saving data for plots...')
            steps_array.append(counter)
            d_b_losses.append(d_b_loss)
            g_b_losses.append(g_b_loss)


    print('Finished training in {:.2f} minutes'.format((time.time()-t0)/60))
    print('Saving training losses plots...')
    save_plots_onedir(steps_array, DATASET, d_b_losses, g_b_losses)
    # print('Saving generated test images...')
    # test_batch = test_batchAB(testA, testB)
    # for _ in range(len(testA)):
    #     test_A, test_B = next(test_batch)
    #     save_test(test_A, test_B, g_a, g_b, SAVE_PATH_TEST)

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=150,
                        help='Number of epochs. Default:150')
    parser.add_argument('-gpu', '--gpu', type=int, default=1,
                        help='If to use GPU. Default: 1')
    parser.add_argument('-n', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    parser.add_argument('-d', '--dataset', type=int, default=0,
                        help='Which dataset to use. Z/H: 0, MM:1. Default:0')
    parser.add_argument('-sw', '--save_weights', type=int, default=0,
                        help='If to save models. Default:0')
    args = parser.parse_args()

    main(args)
