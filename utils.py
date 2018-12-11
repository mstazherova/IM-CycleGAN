import os
import numpy as np
from skimage import transform
import time
import glob
from scipy import misc
import imageio


SAMPLE_DIR = '/tmp/stazherova/samples/{}'.format(time.strftime('%Y%m%d-%H%M%S'))

CHECKPOINT_DIR = '/tmp/stazherova/checkpoint/'
CHECKPOINT_FILE = 'cyclegan.ckpt'


def load_data(path, img_size=256, channels=3):
    image = imageio.imread(path)
    image = transform.resize(image, [img_size, img_size, channels])
    image = (image / 127.5) - 1.

    return image


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
    
    misc.imsave(os.path.join(SAMPLE_DIR,'{}_A.jpg'.format(idx)), x_val[0])
    misc.imsave(os.path.join(SAMPLE_DIR,'{}_B.jpg'.format(idx)), y_val[0])

    misc.imsave(os.path.join(SAMPLE_DIR,'{}_A_2B.jpg'.format(idx)), y)
    misc.imsave(os.path.join(SAMPLE_DIR,'{}_B_2A.jpg'.format(idx)), x)
    misc.imsave(os.path.join(SAMPLE_DIR,'{}_Cycle_A.jpg'.format(idx)), cycle_x)
    misc.imsave(os.path.join(SAMPLE_DIR,'{}_Cycle_B.jpg'.format(idx)), cycle_y)


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
