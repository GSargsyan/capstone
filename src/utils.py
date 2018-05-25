from random import randrange
import numpy as np
import cv2

from helpers import pp


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


def cumul(d):
    """ Cumulates the values in list """
    c = 0
    for val, count in enumerate(d):
        d[val] = count + c
        c += count
    return d


def adjust(hist1, hist2):
    """ Adjusts cumul histogram 1 to hitogram 2 """
    adjusted = [0 for _ in range(255)]
    for val, count in enumerate(hist1):
        closest, closest_idx = _closest(count, hist2)
        adjusted[closest_idx] = count
        #del hist2[closest_idx]
    return adjusted


def _closest(e, l):
    """ Finds the element and its index, closest to e, in l """
    closest = None
    closest_idx = None
    min_diff = float('inf')
    for i, j in enumerate(l):
        diff = abs(j - e)
        if diff < min_diff:
            min_diff = abs(j - e)
            closest = j
            closest_idx = i
    return closest, closest_idx


def match_hists(hist1, hist2):
    k = len(hist1)
    mapping = [0 for _ in range(k)]

    for a in range(k):
        j = k - 1
        while True:
            mapping[a] = j
            j -= 1
            if j < 0 or hist1[a] > hist2[j]:
                break
    return mapping


def draw_lines(img, p1, p2, p3):
    """ Draws a line on img in random color made by
    joinging p1 to p2 and p2 to p3
    """
    r, g, b = randrange(0, 255, 1), randrange(0, 255, 1), randrange(0, 255, 1)
    pp(p1)
    print(b, g, r)
    cv2.line(img, p1, p2, (b, g, r,))
    cv2.line(img, p2, p3, (b, g, r,))
