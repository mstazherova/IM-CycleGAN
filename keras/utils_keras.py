"""Utility functions."""

import os
import time

from random import shuffle
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from keras import backend as K  # pylint: disable=wrong-import-position


parent_dir, _ = os.path.split(os.getcwd())

def mae(target, output):
    """Mean absolute error."""
    return K.mean(K.abs(output - target), axis=-1)


def mse(target, output):
    """Mean squared error."""
    return K.mean(K.square(output - target))  # ? axis=-1 doesn't work


def disc_loss(disc, real, fake, pool):
    """Full discriminator loss."""
    d_real = disc([real])  # input  -> [0, 1].  Prob that real input is real.
    d_fake = disc([fake]) # generated sample -> [0, 1]. Prob that generated output is real.
    d_fake_pool = disc([pool])
    d_loss_real = mse(K.ones_like(d_real) * 0.9, d_real)
    d_loss_fake = mse(K.zeros_like(d_fake), d_fake_pool)
    d_loss = (d_loss_real + d_loss_fake)/2

    return d_loss


def cycle_loss(reconstructed, real):
    """Cyclic loss."""
    return mae(real, reconstructed)


def gen_loss(disc, fake):
    """Generator loss."""
    d_gen = disc([fake])
    return mse(K.ones_like(d_gen), d_gen)


def read_image(img, imagesize=256):
    """Reads in and normalizes an image."""
    img = Image.open(img).convert('RGB')
    img = img.resize((imagesize, imagesize), Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5

    return img


def minibatch(data, batchsize=1):
    """Creates minibatches of the dataset."""
    length = len(data)
    shuffle(data)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1
        rtn = [read_image(data[j]) for j in range(i, i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, dataB, batchsize=1):
    """Yields epoch, batch pairs for both datasets."""
    batchA = minibatch(dataA, batchsize)
    batchB = minibatch(dataB, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B


def test_batch(data, batchsize=1):
    """Creates test batches."""
    i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        rtn = [read_image(data[j]) for j in range(i,i+size)]
        i+=size
        tmpsize = yield np.float32(rtn)


def test_batchAB(dataA, dataB, batchsize=1):
    """Yields test pairs for both datasets."""
    batchA = test_batch(dataA, batchsize)
    batchB = test_batch(dataB, batchsize)
    tmpsize = None
    while True:
        A = batchA.send(tmpsize)
        B = batchB.send(tmpsize)
        tmpsize = yield A, B


def save_plots(steps, dataset, d_a, d_b, g_a, g_b):
    """Plot losses.

    Plots losses for all discriminator and generator networks."""
    sns.set()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,10))
    fig.subplots_adjust(hspace=0.3)
    plt.xlabel("Iterations")

    ax1.plot(steps, d_a, label="D_A")
    ax1.plot(steps, d_b, label="D_B")
    ax1.legend()
    ax1.set_title("Discriminator loss")
    ax1.set_ylabel("Loss")

    ax2.plot(steps, g_a, label="G_A")
    ax2.plot(steps, g_b, label="G_B")
    ax2.legend()
    ax2.set_title("Generator loss")
    ax2.set_ylabel("Loss")

    fig.savefig(os.path.join(parent_dir, 'logs/dataset{}-losses{}.png'.format(dataset, time.strftime('%Y%m%d-%H%M%S'))))


def save_image(X, path, epoch=None, it=None, rows=1, image_size=256):
    """Saves image on disk."""
    assert X.shape[0]%rows == 0
    int_X = ((X*127.5+127.5).clip(0, 255).astype('uint8'))
    int_X = int_X.reshape(-1, image_size, image_size, 3)
    int_X = int_X.reshape(rows, -1, image_size, image_size, 3).swapaxes(1,2).reshape(rows*image_size, -1, 3)
    pil_X = Image.fromarray(int_X)
    if epoch:
        pil_X.save('{}epoch{}-it{}.jpg'.format(path, epoch, it), 'JPEG')
    else:
        pil_X.save('{}{}.jpg'.format(path, time.strftime('%Y%m%d-%H%M%S')), 'JPEG')


def save_generator(A, B, g_a, g_b, path, epoch, it):
    """Saves images produced by generator, with original and reconstructed."""
    if not os.path.isdir(path):
        os.makedirs(path)
    generated_b = g_b.predict(A)
    rec_a = g_a.predict(generated_b)
    generated_a = g_a.predict(B)
    rec_b = g_b.predict(generated_a)

    arr = np.concatenate([A, B, generated_b, generated_a, rec_a, rec_b])
    save_image(arr, path, epoch=epoch, it=it, rows=3)


def save_test(A, B, g_a, g_b, path):
    """Saves images produced by generator, with original input"""
    if not os.path.isdir(path):
        os.makedirs(path)
    generated_b = g_b.predict(A)
    generated_a = g_a.predict(B)

    arr = np.concatenate([A, B, generated_b, generated_a])
    save_image(arr, path, rows=2)


def get_metrics_disc(disc, X):
    """Returns F1 score, false and true positive rate and area under curve."""
    y_pred = disc.predict(X)  # softmax probabilities
    y_prob = y_pred[:,1]       # probabilities of positive class

    y = K.ones_like(X) * 0.9
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    f1 = f1_score(y, np.argmax(y_pred, axis=1), average='micro')

    return f1, fpr, tpr, auc


def plot_roc_curve(disc, X, dataset):
    """Plots ROC curve.

    Given the false and true positive rate, as well as the area under curve,
    this function plots the ROC curve for a given dataset."""
    sns.set()

    f1, fpr, tpr, auc = get_metrics_disc(disc, X)

    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr, tpr, lw=lw, label='ROC curve (area = {:.2}, f1 = {:.3})'.format(auc, f1))
    ax.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    ax.xlim([0.0, 1.0])
    ax.ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic on {} set'.format(dataset))
    ax.legend(loc="lower right")

    fig.savefig(os.path.join(parent_dir, 'logs/dataset{}-ROC{}.png'.format(dataset, time.strftime('%Y%m%d-%H%M%S'))))



def save_models(epoch, genA2B, genB2A, discA, discB):
    """Saves model weights on disk."""
    genA2B.save(os.path.join(parent_dir, 'models/generatorA2B_epoch_{}.h5'.format(epoch)))
    genB2A.save(os.path.join(parent_dir, 'models/generatorB2A_epoch_{}.h5'.format(epoch)))
    discA.save(os.path.join(parent_dir, 'models/discriminatorA_epoch_{}.h5'.format(epoch)))
    discB.save(os.path.join(parent_dir, 'models/discriminatorB_epoch_{}.h5'.format(epoch)))
