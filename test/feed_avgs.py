import cv2
import os

import setup
from config import DATASET_PATH

from core import faces, landmarks, all_distances, ratios, feed_ratio


def _feed_ratio(img_name):
    """ Assumes one face per image """
    img = cv2.imread(DATASET_PATH + img_name)
    # img = resize(img, width=1000)

    dlib_face = faces(img)[0]
    # Open this to convert all faces to same width
    """
    face_only = extract_rect(img, face)
    face_resized = resize(face_only, width=500)

    dlib_face = faces(face_resized)[0]
    """
    lands = landmarks(img, dlib_face)

    all_dists = all_distances(lands, allow_repeats=True)
    rs = ratios(all_dists)
    feed_ratio(rs)


def _feed_ratios(img_names):
    for img_name in img_names:
        _feed_ratio(img_name)


""" --- MAIN --- """

if __name__ == '__main__':
    """
    img_name = IMG_PATH + 'me-frontal.jpg'
    img = cv2.imread(img_name)
    compare_with_avg(img)
    """

    # Use this to feed ratios.json
    img_names = os.listdir(DATASET_PATH)
    _feed_ratios(img_names)
