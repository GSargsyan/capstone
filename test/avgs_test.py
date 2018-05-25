import setup
import cv2

from core import compare_with_avg, lands_from_img
from utils import show_img_with_lands
from config import DATASET_PATH

from helpers import pp

if __name__ == '__main__':
    scores = {}
    for i in range(1, 200):
        img_path = DATASET_PATH + str(i) + '-11.jpg'
        img = cv2.imread(img_path)
        scores[i] = compare_with_avg(img)
        print(i)
    pp(scores)
    """
    img_path = DATASET_PATH + '131-11.jpg'
    #img_path = '/var/www/Capstone/var/img/ugly2.jpg'
    img = cv2.imread(img_path)
    compare_with_avg(img)

    lands = lands_from_img(img)
    show_img_with_lands(img, lands)
    """
