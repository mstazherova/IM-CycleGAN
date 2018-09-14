import os
import argparse
import time 
import numpy as np

import tensorflow as tf

from input import A, B
from model import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# ensure reproducability
np.random.seed(1234)

start = time.time()

CHECKPOINT_FILE = 'cyclegan.ckpt'
CHECKPOINT_DIR = './checkpoint/'
LOG_DIR = './log/'

WIDTH = 256
HEIGHT = 256
CHANNEL = 3
BATCH_SIZE = 1


def save_model(saver, sess, counter):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    saver.save(sess, path, global_step=counter)
    return path


def build_model(input_a, input_b):
    #TODO create a class (maybe)

    with tf.variable_scope('model') as scope:
        #TODO rename a and b into source and target domains
        g1 = generator(input_a, 'g_a2b')     # input A -> generated sample B 
        g2 = generator(input_b, 'g_b2a')     # input B -> generated sample A
        d_a = discriminator(input_a, 'd_a')  # input A -> [0, 1]
        d_b = discriminator(input_b, 'd_b')  # input B -> [0, 1]

        scope.reuse_variables()

        d_gen_b = discriminator(g1, 'd_b')  # generated sample B -> [0, 1]
        d_gen_a = discriminator(g2, 'd_a')  # generated sample A -> [0, 1]
        cycle_a = generator(g1, 'g_b2a')    # generated B -> reconstructed A
        cycle_b = generator(g2, 'g_a2b')    # generated A -> reconstructed B

    # Discriminator loss
    # recommendation for images from A must be close to 1
    d_a_loss_real = tf.reduce_mean(tf.squared_difference(d_a, 1))  
    d_b_loss_real = tf.reduce_mean(tf.squared_difference(d_b, 1))

    # predicting 0 for images produced by the generator
    d_a_loss_fake = tf.reduce_mean(tf.square(d_gen_a))  
    d_b_loss_fake = tf.reduce_mean(tf.square(d_gen_b))

    d_a_loss = tf.reduce_mean(d_a_loss_real + d_a_loss_fake)
    d_b_loss = tf.reduce_mean(d_b_loss_real + d_b_loss_fake)

    # Generator loss
    g_a_loss = tf.reduce_mean(tf.squared_difference(d_gen_a, 1))
    g_b_loss = tf.reduce_mean(tf.squared_difference(d_gen_b, 1))

    # Reconstruction loss
    cycle_loss = tf.reduce_mean(tf.abs(input_a - cycle_a)) + \
                 tf.reduce_mean(tf.abs(input_b - cycle_b))

    g_total_a = g_a_loss + 10 * cycle_loss
    g_total_b = g_b_loss + 10 * cycle_loss

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

    print('The model is built! It took {} seconds'.format(time.time() - start))

    return d_a_loss, d_b_loss, g_total_a, g_total_b, d_a_train_op, \
           d_b_train_op, g_a_train_op, g_b_train_op, g1, g2
    

def main(epochs, num_img):
    """Main loop."""

    tf.reset_default_graph() 

    A_input, B_input = A, B

    input_a = tf.placeholder(dtype=tf.float32, 
                             shape=[None, HEIGHT, WIDTH, CHANNEL],
                             name='input_a')
    input_b = tf.placeholder(dtype=tf.float32, 
                             shape=[None, HEIGHT, WIDTH, CHANNEL], 
                             name='input_b')

    d_a_loss, d_b_loss, g_a_loss, g_b_loss, d_a_train_op, d_b_train_op, \
    g_a_train_op, g_b_train_op, g1, g2 = build_model(input_a, input_b)

    with sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for step in range(num_img):  # batch size = 1
                a_input = A_input[step].reshape(1, HEIGHT, WIDTH, CHANNEL)
                b_input = B_input[step].reshape(1, HEIGHT, WIDTH, CHANNEL)

                # G a -> b, D_b
                _, gen_b_tmp = sess.run([g_a_train_op, g1], 
                                        feed_dict={input_a:a_input, 
                                                   input_b:b_input})
                _ = sess.run([d_b_train_op], 
                             feed_dict={input_a:a_input, 
                                        input_b:b_input})
                # G b -> a, D_a
                _, gen_a_tmp = sess.run([g_b_train_op, g2], 
                                        feed_dict={input_a:a_input, 
                                                   input_b:b_input})
                _ = sess.run([d_a_train_op], 
                             feed_dict={input_a:a_input, 
                                        input_b:b_input})
            d_loss_a = sess.run(d_a_loss, {input_a: a_input, input_b: b_input})
            g_loss_a = g_a_loss.eval({input_a: a_input, input_b: b_input})
            d_loss_b = sess.run(d_b_loss, {input_a: a_input, input_b: b_input})
            g_loss_b = g_b_loss.eval({input_a: a_input, input_b: b_input})
            print("Epoch {}/{}...".format(epoch+1, epochs))
            print("Discriminator A Loss: {:.4f}...".format(d_loss_a))
            print("Generator A Loss: {:.4f}".format(g_loss_a))
            print("Discriminator B Loss: {:.4f}...".format(d_loss_b))
            print("Generator B Loss: {:.4f}".format(g_loss_b))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50, 
                        help='Number of epochs. Default:50')
    parser.add_argument('-i', '--images', type=int, default=1000,
                        help='Number of training images/batches. Default:1000')
    args = parser.parse_args()

    main(args.epochs, args.images)
