import cv2
import setup
from utils import show_img_with_lands, visible_only
from core import faces, landmarks, all_distances, ratios, feed_ratio,\
    show_img, extract_rect, calc_avg_ratios, calc_ratio_stds,\
    compare_with_avg
from config import IMG_PATH, PHI
from helpers import pp

def golden_test(img_name):
    img = cv2.imread(IMG_PATH + img_name)
    dlib_face = faces(img)[0]
    lands = landmarks(img, dlib_face)
    all_dists = all_distances(lands, allow_repeats=True)
    rs = ratios(all_dists)
    golden_scores = [abs(PHI - a[3]) for a in rs]
    for i, j in enumerate(golden_scores):
        if j < 0.001:
            pp(rs[i])


if __name__ == '__main__':
    img_name = 'mona-lisa.jpg'
    golden_test(img_name)
