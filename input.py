import numpy as np
import glob
import cv2
import time 

start = time.time()

IR_DATA_PATH = 'data/ir/'
RGB_DATA_PATH = 'data/rgb/'

def load_data():
        """Loads data. Returns 2 numpy arrays 
        (RGB pictures and IR pictures)."""
        rgb_images = sorted(glob.glob(RGB_DATA_PATH + '**/*.png', 
                                      recursive=True)) 
        ir_images = sorted(glob.glob(IR_DATA_PATH + '**/*.png', 
                                     recursive=True))  

        RGB = []
        IR = []

        for img_name in rgb_images:
                image = cv2.imread(img_name)
                RGB.append(image)

        for img_name in ir_images:
                image = cv2.imread(img_name)
                IR.append(image)

        RGB = np.array(RGB)
        IR = np.array(IR)

        return RGB, IR

# RGB, IR = load_data()

# print('Shape of IR array: {}'.format(IR.shape))
# print('Shape of RGB array: {}'.format(RGB.shape))
# print('Took {} seconds'.format(time.time() - start))
