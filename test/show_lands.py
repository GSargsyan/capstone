import cv2
import setup
from utils import show_img_with_lands, visible_only
from core import faces, landmarks, all_distances, ratios, feed_ratio,\
    show_img, extract_rect, calc_avg_ratios, calc_ratio_stds,\
    compare_with_avg
from config import IMG_PATH

img_name = 'me-frontal.jpg'
img = cv2.imread(IMG_PATH + img_name)

dlib_face = faces(img)[0]
lands = landmarks(img, dlib_face)
show_img_with_lands(img, lands)
