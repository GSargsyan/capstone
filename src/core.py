import os
import math
import json
import statistics
from imutils import face_utils
from copy import deepcopy

import numpy as np
import dlib
import cv2

from config import LANDS_PATH, RATIOS_PATH,\
        RATIOS_AVG_PATH, RATIOS_STDS_PATH, ANGLES_PATH
from helpers import pp, throw
from utils import adjust, cumul, match_hists


NUM_OF_RATIOS = 100232
NUM_OF_ANGLES = 100232


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


def feed_angle(ang):
    """ Writes into angles the new angle - ang
    The file is a line by line json objects
    """
    with open(ANGLES_PATH, 'a') as angles:
        json.dump(ang, angles)
        angles.write("\n")


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


def calc_avg_angles():
    # TODO: To Be Continued...
    sums = np.zeros(NUM_OF_ANGLES)
    with open(ANGLES_PATH, 'r') as angles:
        for line_count, line in enumerate(angles):
            curr_angles = json.loads(line)
            for n, angle in enumerate(curr_angles):
                if is_number(angle[3]):
                    sums[n] += angle[3]

    for i, j in enumerate(sums):
        avgs.append(j / (line_count + 1))
    with open(ANGLES_AVG_PATH, 'w') as avgs_file:
        json.dump(avgs, avgs_file)


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
            print("Num of lines processed: " + str(line_count))
    print("Calculating standard deviations...")
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
    score = sum(z_scores)
    mean_score = score / len(z_scores)
    pp(mean_score)


def z_score(x, u, o):
    return abs((x - u) / o)


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


def lands_from_img(img):
    dlib_face = faces(img)[0]
    return landmarks(img, dlib_face)


def angles(lands):
    length = len(lands) # 68
    angles = []
    for i in range(length):
        for j in range(i + 1, length):
            for k in range(j + 1, length):
                ang = angle(lands[i], lands[j], lands[k])
                angles.append((i, j, k, ang))
    return angles


def angle(a, b, c):
    """ Returns angle in degrees between 
    line 1 and line 2 created by joining p1 to p2 and p2 to p3
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def equalize_red(img1, img2):
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    if height1 != height2 or width1 != width2:
        throw('img1 and img2 should have the same size')

    red_hist1 = [0 for _ in range(255)]
    red_hist2 = [0 for _ in range(255)]

    # compute red color histograms
    for i in range(height1):
        for j in range(width1):
            b1, g1, r1 = img1[i, j]
            b2, g2, r2 = img2[i, j]
            red_hist1[r1] += 1
            red_hist2[r2] += 1
            #hsv1 = to_hsv(img1)
            #h, s, v = hsv[i, j]
            #hsv[i, j] = np.array([h, s, v], dtype=np.uint8)
    cumul1 = cumul(red_hist1)
    cumul2 = cumul(red_hist2)

    adjusted = match_hists(cumul1, cumul2)

    return repaint_red(img1, adjusted)


def repaint_red(img, mapping):
    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            r = mapping[r]
            img[i, j] = b, g, r
    return img


def extract_face(img):
    # TODO: Continue
    lands = lands_from_img(img)


def bot_0(x):
    return 0.9848 * x - 6.7474
def top_0(x):
    return - 0.0009 * x * x + 1.1917 * x - 4.0146

def bot_1(x):
    return - 0.0009 * x * x + 1.1917 * x - 4.0146
def top_1(x):
    return - 0.0011 * x * x + 1.2262 * x + 4.0264

def bot_2(x):
    return - 0.0011 * x * x + 1.2262 * x + 4.0264
def top_2(x):
    return - 0.0013 * x * x + 1.2608 * x + 12.067

def bot_3(x):
    return - 0.0013 * x * x + 1.2608 * x + 12.067
def top_3(x):
    return - 0.0026 * x * x + 1.5713 * x + 14.8


def layer(img, layer_no):
    tmp = deepcopy(img)
    top_func = eval('top_' + str(layer_no))
    bot_func = eval('bot_' + str(layer_no))
    h, w, c = tmp.shape
    for i in range(h):
        for j in range(w):
            px = tmp[i, j]
            b, g, r = px
            rb = (int(r) + int(b)) / 2

            if b < g and g < r and rb >= bot_func(g) and rb <= top_func(g):
                tmp[i, j] = 0, 0, 0
            else:
                tmp[i, j] = 255, 255, 255
    return tmp
