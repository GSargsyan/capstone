import cv2
import numpy as np
import os
from imutils import resize


""" MAIN """
dataset_path = '/home/grigor/Homeworks/imgprocessing/project3/dataset1/'
celebs_path = '/home/grigor/Homeworks/imgprocessing/project3/celebs/'

layer_no = 1
for img_name in os.listdir(dataset_path ):
    print(dataset_path + img_name)
    img = cv2.imread(dataset_path  + img_name)
    img = resize(img, width=500)
    cv2.imshow('original', img)
    layer(img, layer_no)
