import os

import setup
from core import angles, lands_from_img, feed_angle
from config import IMG_PATH, DATASET_PATH

""" --- MAIN --- """

if __name__ == '__main__':
    for im_no, img_name in enumerate(os.listdir(DATASET_PATH)):
        print(im_no + 1)
        img = cv2.imread(DATASET_PATH + img_name)
        ls = lands_from_img(img)
        angs = angles(ls)
        feed_angle(angs)
