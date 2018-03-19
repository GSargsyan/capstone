import os
import json
import statistics

from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2

from config import IMG_PATH, LANDS_PATH, AVERAGES_PATH
from helpers import pp, throw


def faces(img):
    """ Returns dlib.rectangles containing rectangles of faces """
    if not isinstance(img, np.ndarray):
        throw("Expected numpy.array got " + type(img))
    if not os.path.isfile(LANDS_PATH):
        throw(LANDS_PATH + " doesn't exist")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(gray, 1)

    return faces

def landmarks(img, face):
    if not isinstance(face, dlib.rectangle):
        throw("Expected dlib.rectangle got " + type(face))
    predictor = dlib.shape_predictor(LANDS_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dlib_shape = predictor(gray, face)
    return face_utils.shape_to_np(dlib_shape)

def all_distances(lands, as_dict=True):
    """ Returns distances from each point of landmarks to every other point.
    Returning format:
    if as_dict is True
    {1: {2: 3}} - from 1st landmark to 2nd the distance is 1
    if as_dict is False
    [(1, 2, 3)] 
    """
    if as_dict:
        distances = {}
    else:
        distances = []
    length = len(lands)
    for i in range(length):
        pfrom = lands[i]
        if as_dict:
            distances[i] = {}
        for j in range(i + 1, length):
            pto = lands[j]
            dis = _distance(pfrom, pto)
            if as_dict:
                distances[i][j] = dis
            else:
                distances.append((i, j, dis))
    return distances


def _distance(p1, p2):
    """ Returns distance between p1 point and p2 """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def stdeviation(data):
    """ Returns variance of the data """
    return statistics.stdev(data)
    

def feed_average(val):
    """ Writes into averages the new value - val """
    with open(AVERAGES_PATH, 'r+') as avgs:
        for line in avgs:
            pass
        avgs.write(str(val) + "\n")


def ratios(distances):
    # TODO:
    if not isinstance(distances, dict):
        throw("Expected dict got " + type(distances))
    ratios = []
    for p1_idx, vals1 in distances.items():
        for p2_idx, dis in vals1.items():
            for p3_idx, vals2 in distances.items():
                ratio = 0
                if p3_idx != p2_idx and p3_idx != p1_idx:
                    if p3_idx < p2_idx:
                        ratio = dis / vals2[p2_idx]
                    elif p3_idx == p2_idx:
                        ratios.extend([
                            (p1_idx, p2_idx, i, dis / v)\
                                    for i, v in vals2.items()])
                if ratio != 0:
                    ratios.append((p1_idx, p2_idx, p3_idx, ratio))


def dlib_rect_to_np(rect):
    """ Format: (x, y, width, height) - tuple """
    return face_utils.rect_to_bb(rect)


def lands_with_names(lands):
     return [{
	    "chin": lands[0:17],
	    "left_eyebrow": lands[17:22],
	    "right_eyebrow": lands[22:27],
	    "nose_bridge": lands[27:31],
	    "nose_tip": lands[31:36],
	    "left_eye": lands[36:42],
	    "right_eye": lands[42:48],
	    "top_lip": lands[48:55] + [lands[64]] + [lands[63]] +\
                    [lands[62]] + [lands[61]] + [lands[60]],
	    "bottom_lip": lands[54:60] + [lands[48]] + [lands[60]] +\
                    [lands[67]] + [lands[66]] + [lands[65]] + [lands[64]]
            }]
