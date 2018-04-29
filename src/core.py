import os
import json
import statistics
from imutils import face_utils

import numpy as np
import dlib
import cv2

from config import LANDS_PATH, RATIOS_PATH, RATIOS_AVG_PATH, RATIOS_STDS_PATH
from helpers import pp, throw


NUM_OF_RATIOS = 100232


def faces(img):
    """ Returns dlib.rectangles containing rectangles of faces """
    if not isinstance(img, np.ndarray):
        throw("Expected numpy.array got " + type(img).__name__)
    if not os.path.isfile(LANDS_PATH):
        throw(LANDS_PATH + " doesn't exist")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(gray, 1)

    return faces


def landmarks(img, face):
    if not isinstance(face, dlib.rectangle):
        throw("Expected dlib.rectangle got " + type(face).__name__)
    predictor = dlib.shape_predictor(LANDS_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dlib_shape = predictor(gray, face)
    return face_utils.shape_to_np(dlib_shape)


def all_distances(lands, as_dict=True, allow_repeats=False):
    """ Returns distances from each point of landmarks to every other point.
    Returning format:
    if as_dict is True
    {1: {2: 3}} - from 1st landmark to 2nd the distance is 3
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
        from_index = 0 if allow_repeats else i + 1
        for j in range(from_index, length):
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


def extract_rect(img, face):
    npf = _dlib_rect_to_np(face)
    return img[npf[1]:npf[1] + npf[3], npf[0]:npf[0] + npf[2]]


def stdeviation(data):
    """ Returns standard deviation of the data """
    return statistics.stdev(data)


def feed_ratio(val):
    """ Writes into ratio the new value - val
    The file is a line by line json objects
    """
    with open(RATIOS_PATH, 'a') as ratios:
        json.dump(val, ratios)
        ratios.write("\n")


def ratios(all_dists):
    if not isinstance(all_dists, dict):
        throw("Expected dict got " + type(all_dists).__name__)

    length = len(all_dists)
    ratios = []
    for i in range(length):
        for j in range(i + 1, length):
            dist_i_j = all_dists[i][j]
            if i == j:
                continue
            for k in range(j + 1, length):
                if i == k or j == k:
                    continue
                dist_j_k = all_dists[j][k]
                dist_i_k = all_dists[i][k]

                by_j = dist_i_j / dist_j_k if dist_j_k != 0 else 0
                by_k = dist_j_k / dist_i_k if dist_i_k != 0 else 0

                ratios.append((i, k, j, by_k))
                ratios.append((i, j, k, by_j))
    return ratios


def calc_avg_ratios():
    sums = np.zeros(NUM_OF_RATIOS)
    with open(RATIOS_PATH, 'r') as ratios:
        for line_count, line in enumerate(ratios):
            curr_ratios = json.loads(line)
            for n, ratio in enumerate(curr_ratios):
                sums[n] += ratio[3]

    avgs = [0 for i in range(NUM_OF_RATIOS)]
    for i, j in enumerate(sums):
        avgs[i] = j / (line_count + 1)
    with open(RATIOS_AVG_PATH, 'w') as avgs_file:
        json.dump(avgs, avgs_file)


def calc_ratio_stds():
    """ Calculate standard deviations of ratios and write them into file.
    Will be total (global NUM_OF_RATIOS)
    """
    data = {i: [] for i in range(NUM_OF_RATIOS)}
    with open(RATIOS_PATH, 'r') as ratios:
        for line_count, line in enumerate(ratios):
            curr_data = [x[3] for x in json.loads(line)]
            for n, ratio in enumerate(curr_data):
                data[n].append(ratio)
            pp("Num of lines processed: " + str(line_count))
    pp("Calculating standard deviations...")
    stds = [stdeviation(data[i]) for i in range(NUM_OF_RATIOS)]
    with open(RATIOS_STDS_PATH, 'w') as stds_file:
        json.dump(stds, stds_file)


def _dlib_rect_to_np(rect):
    """ Format: (x, y, width, height) - tuple """
    return face_utils.rect_to_bb(rect)


def show_img(img, name='default'):
    cv2.imshow(name, img)
    cv2.wait()


def compare_with_avg(img):
    fs = faces(img)
    lands = landmarks(img, fs[0])
    all_dists = all_distances(lands, allow_repeats=True)
    rs = [x[3] for x in ratios(all_dists)]

    with open(RATIOS_AVG_PATH, 'r') as avgs_file:
        means = json.load(avgs_file)
    with open(RATIOS_STDS_PATH, 'r') as stds_file:
        stds = json.load(stds_file)
    z_scores = [z_score(rs[i], means[i], stds[i])
                for i in range(NUM_OF_RATIOS)]
    pp(z_scores)


def z_score(x, u, o):
    return (x - u) / o


def lands_with_names(lands):
    return [{
        "chin": lands[0:17],
        "left_eyebrow": lands[17:22],
        "right_eyebrow": lands[22:27],
        "nose_bridge": lands[27:31],
        "nose_tip": lands[31:36],
        "left_eye": lands[36:42],
        "right_eye": lands[42:48],
        "top_lip": lands[48:55] + [lands[64]] + [lands[63]] +
        [lands[62]] + [lands[61]] + [lands[60]],
        "bottom_lip": lands[54:60] + [lands[48]] + [lands[60]] +
        [lands[67]] + [lands[66]] + [lands[65]] + [lands[64]]
     }]
