import cv2
import setup
from utils import show_img_with_lands, visible_only, draw_lines
from core import faces, landmarks, all_distances, ratios, feed_ratio,\
    show_img, extract_rect, calc_avg_ratios, calc_ratio_stds,\
    compare_with_avg, lands_from_img, angles
from config import IMG_PATH, PHI
from helpers import pp, is_number, cvwait


def golden_test(img_name):
    img = cv2.imread(IMG_PATH + img_name)
    lands = lands_from_img(img)

    all_dists = all_distances(lands, allow_repeats=True)
    rs = ratios(all_dists)
    angs = angles(lands)

    for i, j in enumerate(rs):
        if abs(j[3] - PHI) < 0.005:
            ang = angs[tuple(sorted((j[0], j[1], j[2])))]
            if not is_number(ang) or ang < 90:
                continue
            p1 = tuple(lands[rs[i][0]])
            p2 = tuple(lands[rs[i][1]])
            p3 = tuple(lands[rs[i][2]])
            draw_lines(img, p1, p2, p3)
    cv2.imshow('golden-test', img)
    cv2.imwrite('golden-test-angles.jpg', img)
    cvwait()


if __name__ == '__main__':
    img_name = 'mona-lisa.jpg'
    golden_test(img_name)
