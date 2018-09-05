import numpy as np
import glob
import cv2
import time 

start = time.time()

IR_DATA_PATH = 'data/ir/'
RGB_DATA_PATH = 'data/rgb/'

HORSE_TRAIN_PATH = 'data/horse2zebra/trainA/'
ZEBRA_TRAIN_PATH = 'data/horse2zebra/trainB/'

def load_data(a_path, b_path):
        """Loads data. Returns 2 numpy arrays."""
        a_images = sorted(glob.glob(a_path + '*.jpg')) 
        b_images = sorted(glob.glob(b_path + '*.jpg'))  

        A = []
        B = []

        for img_name in a_images:
                image = cv2.imread(img_name)
                A.append(image)

        for img_name in b_images:
                image = cv2.imread(img_name)
                B.append(image)

        A = np.array(A)
        B = np.array(B)

        return A, B


A, B = load_data(HORSE_TRAIN_PATH, ZEBRA_TRAIN_PATH)

# print('Shape of horses train array: {}'.format(A.shape))
# print('Shape of zebras train array: {}'.format(B.shape))
# print('Took {} seconds'.format(time.time() - start))
