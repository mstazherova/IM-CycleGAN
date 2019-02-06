import os
import cv2
from PIL import ImageOps

folder = 'data/mm/no_glasses'
capture = cv2.VideoCapture('data/mm/IMG_0001.mov')
# width  = int(capture.get(3))
# height = int(capture.get(4))
# fps = int(capture.get(5))


def getFrame(sec):
     capture.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
     hasFrames, image = capture.read()
     if hasFrames:
          # image = image[0:1920, 300:1600]
          image = image[100:1000, 400:1400]
          image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
          image = cv2.flip(image, flipCode=-1)
          cv2.imwrite(os.path.join(folder, "01-{}sec.jpg".format(sec)), image)     
     
     return hasFrames


sec = 0
count = 0
frameRate = 0.06  
success = getFrame(sec)

while success:
     sec = sec + frameRate
     sec = round(sec, 2)
     success = getFrame(sec)
     count += 1 
     
print("{} images are extracted in {}.".format(count, folder))
