import cv2
import numpy as np


def show_img_with_lands(img, lands):
    for (x, y) in lands:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Image with landmarks", img)
    cv2.waitKey(0)


def visible_only(lands):
    """ Returns filtered lands containing only the ones
    that are visible in the image
    """
    np.array([(x, y) for (x, y) in lands if x >= 0 and y >= 0])
