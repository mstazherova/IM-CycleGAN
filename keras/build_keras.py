import os
import argparse
import time
import glob
import numpy as np
import random

import tensorflow as tf

from model_keras import patch_discriminator, unet_generator
from utils_keras import disc_loss, gen_loss, cycle_loss, minibatchAB
from utils_keras import save_generator, save_plots, save_models
from images_keras import ImagePool

from keras import backend as K
from keras import optimizers

# TODO write docstrings


def build_model(h=256, w=256):
    fake_pool_a = K.placeholder(shape=(None, h, w, 3))
    fake_pool_b = K.placeholder(shape=(None, h, w, 3))

    d_a = patch_discriminator()
    d_b = patch_discriminator()
    g_a = unet_generator()  # generator b2a
    g_b = unet_generator()  # generator a2b
    real_a = g_b.inputs[0]
    fake_b = g_b.outputs[0]
    rec_a = g_a([fake_b])
    real_b = g_a.inputs[0]
    fake_a = g_a.outputs[0]
    rec_b = g_b([fake_a])

    d_a_loss = disc_loss(d_a, real_a, fake_a, fake_pool_a)
    d_b_loss = disc_loss(d_b, real_b, fake_b, fake_pool_b)
    g_a_loss = gen_loss(d_a, fake_a)
    g_b_loss = gen_loss(d_b, fake_b)
    cycle_a_loss = cycle_loss(rec_a, real_a)
    cycle_b_loss = cycle_loss(rec_b, real_b)

    # cycleA_generate = K.function([real_a], [fake_b, rec_a])
    # cycleB_generate = K.function([real_b], [fake_a, rec_b])

    cyc_loss = cycle_a_loss + cycle_b_loss

    g_total = g_a_loss + g_b_loss + 10 * cyc_loss
    d_total = (d_a_loss + d_b_loss) * 0.5

    weights_d = d_a.trainable_weights + d_b.trainable_weights
    weights_g = g_a.trainable_weights + g_b.trainable_weights

    # Define optimizers
    adam_disc = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)
    adam_gen = optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999)

    training_updates_disc = adam_disc.get_updates(weights_d, [], d_total)  #pylint: disable=too-many-function-args
    d_train_function = K.function([real_a, real_b, fake_pool_a, fake_pool_b], [d_a_loss, d_b_loss], training_updates_disc) 
    training_updates_gen = adam_gen.get_updates(weights_g, [], g_total)  #pylint: disable=too-many-function-args
    g_train_function = K.function([real_a, real_b], [g_a_loss, g_b_loss, cyc_loss], training_updates_gen)

    return  d_train_function, g_train_function, g_a, g_b, d_a, d_b, adam_disc, adam_gen


def main(arguments):
    t0 = time.time()
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number
    DATASET = arguments.dataset
    SAVE = arguments.save_weights

    if GPU == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(GPU_NUMBER)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

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
        trainA = glob.glob(os.path.join(parent_dir, 'data/mm/no_glasses/*'))
        trainB = glob.glob(os.path.join(parent_dir, 'data/mm/glasses/*'))
        testA = glob.glob(os.path.join(parent_dir, 'data/mm/no_glasses_test/*'))
        testB = glob.glob(os.path.join(parent_dir, 'data/mm/glasses_test/*'))

    SAVE_PATH_TRAIN = os.path.join(parent_dir, 'results/dataset{}-generated-train{}/'.format(DATASET, time.strftime('%Y%m%d-%H%M%S')))
    SAVE_PATH_TEST = os.path.join(parent_dir, 'results/dataset{}-generated-test{}/'.format(DATASET, time.strftime('%Y%m%d-%H%M%S')))
    DISPLAY_STEP = 500

    SUMMARY_STEP = min(len(trainA), len(trainB))
    epoch = 0
    counter = 0

    steps_array = []

    d_a_losses = []
    d_b_losses = []
    g_a_losses = []
    g_b_losses = []

    d_trainer, g_trainer, g_a, g_b, d_a, d_b, adam_disc, adam_gen = build_model()

    train_batch = minibatchAB(trainA, trainB)
    test_batch = minibatchAB(testA, testB)

    # Initialize fake images pools
    pool_a2b = []
    pool_b2a = []

    while epoch < EPOCHS:
        epoch, A, B = next(train_batch)
        _, testA, testB = next(test_batch)

        # Learning rate decay
        if epoch < 100:
            lr = 2e-4
        else:
            lr = 2e-4 - (2e-4 * (epoch - 100) / 100)
        adam_disc.lr = lr
        adam_gen.lr = lr

        # Fake images pool 
        a2b = g_b.predict(A)
        b2a = g_a.predict(B)

        tmp_b2a = []
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

        for element in b2a:
            if len(pool_b2a) < 50:
                pool_b2a.append(element)
                tmp_b2a.append(element)
            else:
                p = random.uniform(0, 1)

                if p >0.5:
                    index = random.randint(0, 49)
                    tmp = np.copy(pool_b2a[index])
                    pool_b2a[index] = element
                    tmp_b2a.append(tmp)
                else:
                    tmp_b2a.append(element)

        pool_a = np.array(tmp_b2a)
        pool_b = np.array(tmp_a2b)

        d_a_loss, d_b_loss  = d_trainer([A, B, pool_a, pool_b])
        g_a_loss, g_b_loss, _ = g_trainer([A, B])

        counter += 1

        if np.mod(counter, DISPLAY_STEP) == 0:
            print('[Epoch {}/{}][Iteration {}]...'.format(epoch, EPOCHS, counter))
            print('D_a_loss: {:.2f}, D_b_loss: {:.2f}'.format(d_a_loss, d_b_loss))
            print('G_a_loss: {:.2f}, G_b_loss: {:.2f}'.format(g_a_loss, g_b_loss))
            print('Saving generated training images...')
            save_generator(A, B, g_a, g_b, SAVE_PATH_TRAIN, epoch)
            print('Saving generated test images...')
            save_generator(testA, testB, g_a, g_b, SAVE_PATH_TEST, epoch)

        if np.mod(counter, SUMMARY_STEP) == 0:
            print('Saving data for plots...')
            steps_array.append(counter)
            d_a_losses.append(d_a_loss)
            d_b_losses.append(d_b_loss)
            g_a_losses.append(g_a_loss)
            g_b_losses.append(g_b_loss)

    
    print('Finished training in {:.2f} minutes'.format((time.time()-t0)/60))
    print('Saving plots...')
    save_plots(steps_array, DATASET, d_a_losses, d_b_losses, g_a_losses, g_b_losses) 

    if SAVE == 1:
        print('Saving models...')
        save_models(epoch, g_b, g_a, d_a, d_b)

    print('Done!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=150, 
                        help='Number of epochs. Default:150')
    parser.add_argument('-gpu','--gpu', type=int, default=1,
                        help='If to use GPU. Default: 1')
    parser.add_argument('-n', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    parser.add_argument('-d', '--dataset', type=int, default=0,
                        help='Which dataset to use. Z/H: 0, MM:1. Default:0')
    parser.add_argument('-sw', '--save_weights', type=int, default=0,
                        help='If to save models. Default:0')
    args = parser.parse_args()

    main(args)
