import os
import numpy as np
import scipy.misc
import time
import glob


SAMPLE_DIR = '/tmp/stazherova/samples/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

CHECKPOINT_DIR = '/tmp/stazherova/checkpoint/'
CHECKPOINT_FILE = 'cyclegan.ckpt'


def sample(it_a, it_b, sess, idx, testX, testY, testG1, testG2, testCx, testCy):
    """Samples generated images from the test set."""
    sess.run(it_a)
    sess.run(it_b)
    x_val, y_val, y_samp, x_samp, x_cycle_samp, y_cycle_samp = sess.run(
        [testX, testY, testG1, testG2, testCx, testCy]) 

    x = merge(inverse_transform(x_samp), [1, 1])
    y = merge(inverse_transform(y_samp), [1, 1])
    cycle_x = merge(inverse_transform(x_cycle_samp), [1, 1])
    cycle_y = merge(inverse_transform(y_cycle_samp), [1, 1])

    if not os.path.isdir(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
    
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_A.jpg'.format(idx)), x_val[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_B.jpg'.format(idx)), y_val[0])

    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_A_2B.jpg'.format(idx)), y_samp[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_B_2A.jpg'.format(idx)), x_samp[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_Cycle_A.jpg'.format(idx)), x_cycle_samp[0])
    scipy.misc.imsave(os.path.join(SAMPLE_DIR,'{}_Cycle_B.jpg'.format(idx)), y_cycle_samp[0])


def save_model(saver, sess, counter):
    """Saves model checkpoint."""
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
    saver.save(sess, path, global_step=counter)

    return path


def merge(imgs, size):
    h, w = imgs.shape[1], imgs.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, img in enumerate(imgs):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = img

    return img


def inverse_transform(images):
    return (images + 1.) / 2.
