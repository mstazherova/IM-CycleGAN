import os
import argparse
import time 
from tqdm import tqdm

import tensorflow as tf

from model import *
from utils import *
from images import Images
from image_cache import ImageCache

np.random.seed(1234)

start = time.time()

LOG_DIR = './logs/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

WIDTH = 256
HEIGHT = 256
CHANNEL = 3
BATCH_SIZE = 1
SOFT = 0.05

SAMPLE_STEP = 10
SAVE_STEP = 50

    
def build_model(input_a, input_b, gen_a_sample, gen_b_sample):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = generator(input_a, name='g_a2b')     # input A -> generated sample B 
        g2 = generator(input_b, name='g_b2a')     # input B -> generated sample A
        d_a = discriminator(input_a, name='d_a')  # input A -> [0, 1]
        d_b = discriminator(input_b, name='d_b')  # input B -> [0, 1]

        d_gen_b = discriminator(g1, name='d_b')   # generated sample B -> [0, 1]
        d_gen_a = discriminator(g2, name='d_a')   # generated sample A -> [0, 1]
        cycle_a = generator(g1, name='g_b2a')     # generated B -> reconstructed A
        cycle_b = generator(g2, name='g_a2b')     # generated A -> reconstructed B

        d_a_sample = discriminator(gen_a_sample, name='d_a')
        d_b_sample = discriminator(gen_b_sample, name='d_b')
       
    # Discriminator loss 
    d_a_loss_real = tf.reduce_mean((d_a - tf.ones_like(d_a) * np.abs(np.random.normal(1.0, SOFT))) ** 2)
    d_a_loss_fake = tf.reduce_mean((d_a_sample - tf.zeros_like(d_a_sample)) ** 2)

    d_b_loss_real = tf.reduce_mean((d_b - tf.ones_like(d_b) * np.abs(np.random.normal(1.0, SOFT))) ** 2)
    d_b_loss_fake = tf.reduce_mean((d_b_sample - tf.zeros_like(d_b_sample)) ** 2)

    d_a_loss = (d_a_loss_real + d_a_loss_fake) / 2
    d_b_loss = (d_b_loss_real + d_b_loss_fake) / 2
    tf.summary.scalar('D_a Loss', d_a_loss)
    tf.summary.scalar('D_b loss', d_b_loss)

    # Generator loss
    g_a_loss = tf.reduce_mean((d_gen_b - tf.ones_like(d_gen_b) * np.abs(np.random.normal(1.0, SOFT))) ** 2)
    g_b_loss = tf.reduce_mean((d_gen_a - tf.ones_like(d_gen_a) * np.abs(np.random.normal(1.0, SOFT))) ** 2)

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

    tf.summary.image('Input A', input_a, max_outputs=5)
    tf.summary.image('Generated B', g1, max_outputs=5)
    tf.summary.image('Input B', input_b, max_outputs=5)
    tf.summary.image('Generated A', g2, max_outputs=5)
    
    print('Built the model in {0:.2f} seconds'.format(time.time() - start))

    return d_a_loss, d_b_loss, g_total_a, g_total_b, d_a_train_op, \
    d_b_train_op, g_a_train_op, g_b_train_op, g1, g2
    

def main(arguments):
    """Main loop."""
    epochs = arguments.epochs
    gpu = arguments.gpu
    to_sample = arguments.sample
    gpu_number = arguments.gpu_number
    DATA_PATH = 'data/horse2zebra/'
    HORSE_TRAIN_PATH = 'data/horse2zebra/trainA/'

    tf.reset_default_graph() 

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu_number)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # iterations = 1067 * epochs
    
    it_a, train_A = Images(DATA_PATH + '_trainA.tfrecords', name='trainA').feed()
    it_b, train_B = Images(DATA_PATH + '_trainB.tfrecords', name='trainB').feed()
    it_at, test_A = Images(DATA_PATH + '_testA.tfrecords', name='test_a').feed()
    it_bt, test_B = Images(DATA_PATH + '_testB.tfrecords', name='test_b').feed()
    
    gen_a_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_a_sample")
    gen_b_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_b_sample")

    d_a_loss, d_b_loss, g_a_loss, g_b_loss, d_a_train_op, d_b_train_op, \
    g_a_train_op, g_b_train_op, g1, g2 = build_model(train_A, train_B, gen_a_sample, gen_b_sample)

    testG1 = generator(test_A, name='g_a2b')
    testG2 = generator(test_B,  name='g_b2a')
    testCycleA = generator(testG1,  name='d_a')
    testCycleB = generator(testG2, name='d_b')
    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with sess:
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(init)

        cache_a = ImageCache(50)
        cache_b = ImageCache(50)

        print('Beginning training...')
        for epoch in range(epochs):
            start = time.time()
            sess.run(it_a)
            sess.run(it_b)
            try:
                for step in tqdm(range(1067)):  # TODO change number of steps
                    gen_a, gen_b = sess.run([g1, g2])
                    _, _, _, _, summaries = sess.run([g_a_train_op, d_b_train_op, 
                                                      g_b_train_op, d_a_train_op, merged],
                                                     feed_dict={gen_b_sample: cache_b.fetch(gen_b),
                                                                gen_a_sample: cache_a.fetch(gen_a)})

                    writer.add_summary(summaries, epoch)
            except tf.errors.OutOfRangeError:
                pass  
           
            print("Epoch {}/{} done...".format(epoch+1, epochs))

            counter = epoch + 1
                
            if np.mod(counter, SAVE_STEP) == 0:
                save_path = save_model(saver, sess, counter)
                print('Running for {0:.2} mins, saving to {}'.format((time.time() - start) / 60, save_path))

            if to_sample and (np.mod(counter, SAMPLE_STEP) == 0):
                sample(it_at, it_bt, sess, counter, test_A, test_B, testG1, testG2, testCycleA, testCycleB)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='Number of epochs. Default:100')
    parser.add_argument('-gpu','--gpu', type=bool, default=False,
                        help='If to use GPU. Default: False')
    parser.add_argument('-number', '--gpu_number', type=int, default=0,
                        help='Which GPU to use. Default:0')
    parser.add_argument('-s', '--sample', type=bool, default=True,
                        help='If to save sampled imgs. Default: True')
    args = parser.parse_args()

    main(args)
