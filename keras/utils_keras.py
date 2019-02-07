import os
from keras import backend as K
from PIL import Image
import numpy as np
from random import shuffle
import time

from matplotlib import pyplot as plt
plt.switch_backend('agg')


parent_dir, _ = os.path.split(os.getcwd())

def mse(output, target):
    return K.mean(K.abs(K.square(output-target)))


def disc_loss(disc, real, fake):
    d_real = disc([real])  # input  -> [0, 1].  Prob that real input is real.
    d_fake = disc([fake])  # generated sample -> [0, 1]. Prob that generated output is real.
    d_loss_real = mse(d_real, K.ones_like(d_real))
    d_loss_fake = mse(d_fake, K.zeros_like(d_fake))
    d_loss = (d_loss_real + d_loss_fake)/2
    
    return d_loss


def cycle_loss(reconstructed, real):
    return K.mean(K.abs(reconstructed - real))


def gen_loss(disc, fake):
    d_gen = disc([fake])
    return mse(d_gen, K.ones_like(d_gen))


def read_image(img, imagesize=256):
    img = Image.open(img).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    img = np.array(img)
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5

    return img


def save_image(X, path, epoch, rows=1, image_size=256):
    assert X.shape[0]%rows == 0
    int_X = ((X*127.5+127.5).clip(0,255).astype('uint8'))
    int_X = int_X.reshape(-1,image_size,image_size, 3)
    int_X = int_X.reshape(rows, -1, image_size, image_size,3).swapaxes(1,2).reshape(rows*image_size,-1, 3)
    pil_X = Image.fromarray(int_X)
    pil_X.save('{}epoch{}.jpg'.format(path, epoch), 'JPEG')


# def save_image(X, path, rows=1, image_size=256):
#     assert X.shape[0] % rows == 0
#     int_X = ((X * 255).clip(0, 255).astype('uint8'))
#     int_X = int_X.reshape(-1, image_size, image_size, 3)
#     int_X = int_X.reshape(rows, -1, image_size, image_size, 3).swapaxes(1, 2).reshape(rows * image_size, -1, 3)
#     pil_X = Image.fromarray(int_X)
#     t = str(time.time())
#     pil_X.save(path + t, 'JPEG')


def save_generator(A, B, rec_a, rec_b, path, epoch):
    if not os.path.isdir(path):
        os.makedirs(path)

    def G(generated, X):
        r = np.array([generated([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]        
    rA = G(rec_a, A)
    rB = G(rec_b, B)
    arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])
    save_image(arr, path, epoch, rows=3)


def minibatch(data, batchsize=1):
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
        rtn = [read_image(data[j]) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)       


def minibatchAB(dataA, dataB, batchsize=1):
    batchA = minibatch(dataA, batchsize)
    batchB = minibatch(dataB, batchsize)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B


def save_plots(steps, d_a, d_b, g_a, g_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5), sharex=True)
    
    ax1.plot(steps, d_a, label="D_A loss")
    ax1.plot(steps, d_b, label="D_B loss")
    ax1.legend()
    
    ax2.plot(steps, g_a, label="G_A loss")
    ax2.plot(steps, g_b, label="G_B loss")
    ax2.legend()

    fig.savefig(os.path.join(parent_dir, '/logs/losses{}.png'.format(time.strftime('%Y%m%d-%H%M%S'))))
