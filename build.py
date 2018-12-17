import os
import argparse
import time 
import numpy as np
from tqdm import tqdm
tqdm.monitor_interval = 0  # issue 481

import tensorflow as tf

from model import discriminator, generator
from utils import sample, save_model
from images import Images
from image_cache import ImageCache

np.random.seed(1234)

LOG_DIR = './logs/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

WIDTH = 256
HEIGHT = 256
CHANNEL = 3
BATCH_SIZE = 1
SOFT = 0.05

SAMPLE_STEP = 10
SAVE_STEP = 50

    
def build_model(input_a, input_b, gen_a_sample, gen_b_sample, lr):
    start = time.time()
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = generator(input_a, name='g_a2b')     # input A -> generated sample B 
        d_b = discriminator(input_b, name='d_b')  # input B -> [0, 1].  Prob that real B is real.
        d_b_gen = discriminator(g1, name='d_b')   # generated sample B -> [0, 1]. Prob that fake B is real.
        cycle_a = generator(g1, name='g_b2a')     # generated B -> reconstructed A

        g2 = generator(input_b, name='g_b2a')     # input B -> generated sample A 
        d_a = discriminator(input_a, name='d_b')  # input A -> [0, 1].  Prob that real A is real.
        d_a_gen = discriminator(g2, name='d_b')   # generated sample A -> [0, 1]. Prob that fake A is real.
        cycle_b = generator(g2, name='g_b2a')     # generated A -> reconstructed B
    
        d_a_sample = discriminator(gen_a_sample, name='d_a')  # Prob that fake A pool is real.
        d_b_sample = discriminator(gen_b_sample, name='d_b')  # Prob that fake B pool is real.

       
    # Discriminator loss 
    # mean squared error
    d_b_loss_real = tf.reduce_mean(tf.squared_difference(d_b, 1))
    d_b_loss_fake = tf.reduce_mean(tf.square(d_b_sample))
    d_b_loss = (d_b_loss_real + d_b_loss_fake)/2
    tf.summary.scalar('Discriminator_B_Loss', d_b_loss)
   
    d_a_loss_real = tf.reduce_mean(tf.squared_difference(d_a, 1))
    d_a_loss_fake = tf.reduce_mean(tf.square(d_a_sample))
    d_a_loss = (d_a_loss_real + d_a_loss_fake)/2
    tf.summary.scalar('Discriminator_A_Loss', d_a_loss)


    # Generator loss
    # g_a_loss = tf.reduce_mean((d_gen_b - tf.ones_like(d_gen_b) * np.abs(np.random.normal(1.0, SOFT))) ** 2)
    # g_b_loss = tf.reduce_mean((d_gen_a - tf.ones_like(d_gen_a) * np.abs(np.random.normal(1.0, SOFT))) ** 2)
    # mean squared error
    g_b_loss = tf.reduce_mean(tf.squared_difference(d_b_gen, 1))
    g_a_loss = tf.reduce_mean(tf.squared_difference(d_a_gen, 1))

    # Reconstruction loss
    cycle_loss = tf.reduce_mean(tf.abs(input_a - cycle_a)) + \
                 tf.reduce_mean(tf.abs(input_b - cycle_b))

    g_total_a = g_a_loss + 10 * cycle_loss
    g_total_b = g_b_loss + 10 * cycle_loss
    tf.summary.scalar('G_a Loss', g_total_a)
    tf.summary.scalar('G_b Loss', g_total_b)

    # Optimizers
    trainable_vars = tf.trainable_variables()
    d_a_vars = [var for var in trainable_vars if 'd_a' in var.name]
    d_b_vars = [var for var in trainable_vars if 'd_b' in var.name]
    g_a_vars = [var for var in trainable_vars if 'g_a' in var.name]
    g_b_vars = [var for var in trainable_vars if 'g_b' in var.name]

    d_a_train_op = tf.train.AdamOptimizer(2e-4).minimize(d_a_loss, 
                                                         var_list=d_a_vars)
    d_b_train_op = tf.train.AdamOptimizer(2e-4).minimize(d_b_loss, 
                                                         var_list=d_b_vars)
    g_a_train_op = tf.train.AdamOptimizer(2e-4).minimize(g_total_a, 
                                                         var_list=g_a_vars)
    g_b_train_op = tf.train.AdamOptimizer(2e-4).minimize(g_total_b, 
                                                         var_list=g_b_vars)

    tf.summary.image('Input A', input_a, max_outputs=10)
    tf.summary.image('Generated B', g1, max_outputs=10)
    tf.summary.image('Input B', input_b, max_outputs=10)
    tf.summary.image('Generated A', g2, max_outputs=10)

    # d_sum = tf.summary.merge([d_a_sum, d_b_sum])
    # g_sum = tf.summary.merge([g_a_sum, g_b_sum, a_sum, g1_sum, b_sum, g2_sum])
    
    print('Built the model in {0:.2f} seconds'.format(time.time() - start))

    return d_a_train_op, d_b_train_op, g_a_train_op, g_b_train_op, g1, g2
    

def main(arguments):
    """Main loop."""
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number
    TO_SAMPLE = arguments.sample

    DATA_PATH = 'data/horse2zebra/'

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
    
    it_a, train_A = Images(DATA_PATH + '_trainA.tfrecords', name='trainA').feed()
    it_b, train_B = Images(DATA_PATH + '_trainB.tfrecords', name='trainB').feed()
    # it_at, test_A = Images(DATA_PATH + '_testA.tfrecords', name='test_a').feed()
    # it_bt, test_B = Images(DATA_PATH + '_testB.tfrecords', name='test_b').feed()
    
    gen_a_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_a_sample")
    gen_b_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_b_sample")
    learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

    d_a_train_op, d_b_train_op, g_a_train_op, g_b_train_op, g1, g2 = \
    build_model(train_A, train_B, gen_a_sample, gen_b_sample, learning_rate)

    # testG1 = generator(test_A, name='g_a2b')
    # testG2 = generator(test_B,  name='g_b2a')
    # testCycleA = generator(testG1,  name='d_a')
    # testCycleB = generator(testG2, name='d_b')

    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with sess:
        sess.run(init)
        writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
        
        cache_a = ImageCache(50)
        cache_b = ImageCache(50)

        print('Beginning training...')
        start = time.perf_counter()
        for epoch in range(EPOCHS):
            sess.run(it_a)
            sess.run(it_b)
            if epoch < 100:
                lr = 2e-4
            else:
                lr = 2e-4 - (2e-4 * (epoch - 100) / 100)
            try:
                for step in tqdm(range(533)):  # TODO change number of steps
                    gen_a, gen_b, = sess.run([g1, g2])

                    _, _, _, _, summaries = sess.run([d_b_train_op, d_a_train_op, 
                                                      g_a_train_op, g_b_train_op, merged],
                                                     feed_dict={gen_b_sample: cache_b.fetch(gen_b),
                                                                gen_a_sample: cache_a.fetch(gen_a),
                                                                learning_rate: lr})
                    if step % 100 == 0:
                        writer.add_summary(summaries, epoch * 533 + step)

            except tf.errors.OutOfRangeError as e:
                print(e)
                print("Out of range: {}".format(step))
                pass  
           
            print("Epoch {}/{} done.".format(epoch+1, EPOCHS))

            counter = epoch + 1
                
            if np.mod(counter, SAVE_STEP) == 0:
                save_path = save_model(saver, sess, counter)
                print('Running for {:.2f} seconds, saving to {}'.format(time.perf_counter() - start, save_path))

            # if TO_SAMPLE == 1 and np.mod(counter, SAMPLE_STEP) == 0:
            #     sample(it_at, it_bt, sess, counter, test_A, test_B, testG1, testG2, testCycleA, testCycleB)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='Number of epochs. Default:100')
    parser.add_argument('-gpu','--gpu', type=int, default=0,
                        help='If to use GPU. Default: 0')
    parser.add_argument('-number', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    parser.add_argument('-s', '--sample', type=int, default=0,
                        help='If to save sampled imgs. Default: 0')
    args = parser.parse_args()

    main(args)
