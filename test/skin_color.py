import setup
import cv2

from config import IMG_PATH, DATASET_PATH
from core import to_max_sat, layer

img_name = DATASET_PATH + '1-11.jpg'

if __name__ == '__main__':
    img = cv2.imread(img_name)
    maxed = to_max_sat(img)
    for l in range(4):
        layer(maxed, l)
    cv2.waitKey(0)
