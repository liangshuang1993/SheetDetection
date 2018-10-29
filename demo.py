# coding:utf-8
import time
from glob import glob
import os

import numpy as np
from PIL import Image

import model
import find_bbox
# ces


if __name__ == '__main__':
    # images = glob('/opt/pics/*.*')
    images = glob('/home/ls/project201803/test/6.JPG')
    # paths = glob('./test/*.*')
    for image in images:
        #if '11' not in image:
        #    continue
        im = Image.open(image)
        # find bounding box of font aera
        # im = find_bbox.process_image(image, '', save_flag=False)
    
        img = np.array(im.convert('RGB'))
        model.model(0, os.path.basename(image), img ,model='keras',recheck_flag=False) ##if model == keras ,you should install keras
        print "---------------------------------------"
