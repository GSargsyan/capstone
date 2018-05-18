import cv2
import setup
from utils import show_img_with_lands, visible_only
from core import faces, landmarks, all_distances, ratios, feed_ratio,\
    show_img, extract_rect, calc_avg_ratios, calc_ratio_stds,\
    compare_with_avg, lands_from_img, angles
from config import IMG_PATH, PHI
from helpers import pp

def golden_test(img_name):
    img = cv2.imread(IMG_PATH + img_name)
    lands = lands_from_img(img)

    all_dists = all_distances(lands, allow_repeats=True)
    rs = ratios(all_dists)
    angs = angles(lands)

    #golden_scores = [abs(PHI - a[3]) for a in rs]

    pp(len(angs))
    pp(len(rs))

    full_rats = [(rs[i][0], rs[i][1], rs[i][2],\
            rs[i][3], angs[i][3]) for i in range(len(rs))]

    pp(full_rats)
    for i, j in enumerate(golden_scores):
        if j < 0.001:
            pp(rs[i])


if __name__ == '__main__':
    img_name = 'mona-lisa.jpg'
    golden_test(img_name)
