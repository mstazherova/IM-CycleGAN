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

SAMPLE_STEP = 10
SAVE_STEP = 50

    
def build_model(input_a, gen_b_sample):
    start = time.time()
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        g1 = generator(input_a, name='g_a2b')     # input A -> generated sample B 
        d_b_sample = discriminator(gen_b_sample, name='d_b') # generated sample B -> [0, 1]
        cycle_a = generator(g1, name='g_b2a')     # generated B -> reconstructed A

        d_a = discriminator(input_a, name='d_a')  # input A -> [0, 1]
             
    # Discriminator loss 
    # mean squared error
    d_a_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_a, labels=tf.ones_like(d_a)))

    d_b_loss = tf.reduce_mean(tf.square(d_b_sample))

    tf.summary.scalar('D_a Loss', d_a_loss)
    tf.summary.scalar('D_b loss', d_b_loss)

    # Generator loss
    # mean squared error
    g_b_loss = tf.reduce_mean(tf.squared_difference(d_b_sample, 0.9))

    # Reconstruction loss
    cycle_loss = tf.reduce_mean(tf.abs(input_a - cycle_a))

    g_total_b = g_b_loss + 10 * cycle_loss
    tf.summary.scalar('G_b Loss', g_total_b)

    # Optimizers
    trainable_vars = tf.trainable_variables()
    d_a_vars = [var for var in trainable_vars if 'd_a' in var.name]
    d_b_vars = [var for var in trainable_vars if 'd_b' in var.name]
    g_b_vars = [var for var in trainable_vars if 'g_b' in var.name]

    d_a_train_op = tf.train.AdamOptimizer(2e-4).minimize(d_a_loss, 
                                                         var_list=d_a_vars)
    d_b_train_op = tf.train.AdamOptimizer(2e-4).minimize(d_b_loss, 
                                                         var_list=d_b_vars)
    g_b_train_op = tf.train.AdamOptimizer(2e-4).minimize(g_total_b, 
                                                         var_list=g_b_vars)

    tf.summary.image('Input A', input_a, max_outputs=2)
    tf.summary.image('Generated B', g1, max_outputs=2) 
    tf.summary.image('Generated_B_sample', gen_b_sample, max_outputs=2)
       
    print('Built the model in {:.2f} seconds'.format(time.time() - start))

    return d_a_train_op, d_b_train_op, g_b_train_op, g1
    

def main(arguments):
    """Main loop."""
    EPOCHS = arguments.epochs
    GPU = arguments.gpu
    GPU_NUMBER = arguments.gpu_number

    DATA_PATH = 'data/horse2zebra/'

    tf.reset_default_graph() 

    if GPU == 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(GPU_NUMBER)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5 
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()
    
    it_a, train_A = Images(DATA_PATH + '_trainA.tfrecords', name='trainA').feed()
    
    gen_b_sample = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNEL], name="fake_b_sample")

    d_a_train_op, d_b_train_op, g_b_train_op, g1 = build_model(train_A, gen_b_sample)

    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()

    with sess:
        sess.run(init)
        writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
    
        cache_b = ImageCache(50)

        print('Beginning training...')
        start = time.perf_counter()
        sess.run(it_a)
        for epoch in range(EPOCHS):
            try:
                for step in tqdm(range(1067)):  # TODO change number of steps
                    gen_b = sess.run(g1)

                    _, _, _, summaries = sess.run([d_a_train_op, d_b_train_op, 
                                                      g_b_train_op, merged],
                                                     feed_dict={gen_b_sample: cache_b.fetch(gen_b)})
                    if step % 100 == 0:
                        writer.add_summary(summaries, epoch * 1067 + step)

            except tf.errors.OutOfRangeError as e:
                print(e)
                print("Out of range: {}".format(step))
                pass  
           
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
