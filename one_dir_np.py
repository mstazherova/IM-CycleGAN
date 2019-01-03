import os
import time
from glob import glob
import numpy as np
import argparse
from tqdm import tqdm
tqdm.monitor_interval = 0  # issue 481

import tensorflow as tf

from model import discriminator, generator
from utils import load
from image_cache import ImageCache

LOG_DIR = './logs/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

WIDTH = 256
HEIGHT = 256
CHANNEL = 3
BATCH_SIZE = 1


def build_model(input_a, input_b, gen_b_sample, lr):
    start = time.time()
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = generator(input_a, name='g_a2b')               # input A -> generated sample B. Fake B images.
        d_b = discriminator(input_b, name='d_b')            # input B -> [0, 1].  Prob that real B is real.
        d_b_gen = discriminator(g1, 'd_b')                  # generated sample B -> [0, 1]. Prob that fake B is real.
        
        cycle_a = generator(g1, name='g_b2a')               # generated B -> reconstructed A. Cycle A images.

        d_b_sample = discriminator(gen_b_sample, "d_b")     # Prob that fake B pool is real. 

        
    # Discriminator loss 
    # mean squared error
    d_loss_real = tf.reduce_mean(tf.squared_difference(d_b, 1))
    d_loss_fake = tf.reduce_mean(tf.square(d_b_sample))
    d_b_loss = (d_loss_real + d_loss_fake)/2
    tf.summary.scalar('Discriminator_B_Loss', d_b_loss)
    
    # Generator loss
    # mean squared error
    g_b_loss = tf.reduce_mean(tf.squared_difference(d_b_gen, 1))
    
    # Reconstruction loss
    cycle_loss = tf.reduce_mean(tf.abs(input_a - cycle_a))
    
    g_total = g_b_loss + 10 * cycle_loss
    tf.summary.scalar('Generator_Loss', g_total)
    
    # Optimizers
    trainable_vars = tf.trainable_variables()
    d_b_vars = [var for var in trainable_vars if 'd_b' in var.name]
    g_vars = [var for var in trainable_vars if 'g_' in var.name]

    d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_b_loss, 
                                                         var_list=d_b_vars)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_total, 
                                                         var_list=g_vars)

    tf.summary.image('Input_A', input_a, max_outputs=10)
    tf.summary.image('Generated_B', g1, max_outputs=10) 
       
    print('Built the model in {:.2f} seconds'.format(time.time() - start))

    return d_b_train_op, g_train_op, g1  


def main(arguments):
    """Main loop."""
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number

    # DATA_PATH = 'data/horse2zebra/'
    HORSE_TRAIN_PATH = 'data/horse2zebra/trainA/'
    ZEBRA_TRAIN_PATH = 'data/horse2zebra/trainB/'
    # A_DIR = './data/trainA/*.jpg'
    # B_DIR = './data/trainB/*.jpg'

    tf.reset_default_graph() 

    if GPU == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(GPU_NUMBER)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # pylint: disable=no-member
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    
    # data_A = glob(DATA_PATH + 'trainA/*.*')
    # data_B = glob(DATA_PATH + 'trainB/*.*')
    data_A = load(HORSE_TRAIN_PATH)
    data_B = load(ZEBRA_TRAIN_PATH)
    np.random.shuffle(data_A)
    np.random.shuffle(data_B)
    batch_idxs = min(len(data_A), len(data_B))

    # imgs_a = [load_data(img_file) for img_file in data_A]
    # imgs_b = [load_data(img_file) for img_file in data_B]
    # imgs_a = np.array(imgs_a)
    # imgs_b = np.array(imgs_b)
    
    input_a = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="input_a")
    input_b = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="input_a")
    gen_b_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_b_sample")
    learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

    d_train_op, g_train_op, g1 = build_model(input_a, input_b, gen_b_sample, learning_rate)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with sess:
        sess.run(init)
        writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
    
        cache_b = ImageCache(50)

        print('Beginning training...')
        start = time.perf_counter()  
        for epoch in range(EPOCHS):
            if epoch < 100:
                lr = 2e-4
            else:
                lr = 2e-4 - (2e-4 * (epoch - 100) / 100)
            
            for step in tqdm(range(batch_idxs)):
                a_input = data_A[step].reshape(BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)
                b_input = data_B[step].reshape(BATCH_SIZE, HEIGHT, WIDTH, CHANNEL)

                gen_b = sess.run(g1, feed_dict={input_a:a_input})
                _, _, summaries = sess.run([d_train_op, g_train_op, merged],
                                            feed_dict={input_a: a_input, input_b: b_input,
                                                       gen_b_sample: cache_b.fetch(gen_b),
                                                       learning_rate: lr})
                if step % 100 == 0:
                    writer.add_summary(summaries, epoch * batch_idxs + step)
           
            print("Epoch {}/{} done, running for {:.2f} minutes.".format(epoch+1, EPOCHS, (time.perf_counter() - start)/60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='Number of epochs. Default:100')
    parser.add_argument('-gpu','--gpu', type=int, default=0,
                        help='If to use GPU. Default: 0')
    parser.add_argument('-number', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    args = parser.parse_args()

    main(args)
