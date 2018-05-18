import os

import setup
from core import angles, lands_from_img, feed_angle
from config import ANGLES_PATH

""" --- MAIN --- """

if __name__ == '__main__':
    for img_name in os.listdir(DATASET_PATH):
        img = cv2.imread(DATASET_PATH + img_name)
        ls = lands_from_img(img)
        angs = angles(ls)

        for ang in angs:
            feed_angle(ang)
