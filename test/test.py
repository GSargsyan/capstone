import cv2
from imutils import resize

import setup
from core import faces, landmarks, all_distances, stdeviation, feed_average
from utils import show_img_with_lands, visible_only
from config import IMG_PATH, AVERAGES_PATH
from helpers import pp

""" --- FUNCTIONS --- """

def feed_avg_std(img_name):
    img = cv2.imread(IMG_PATH + img_name)
    img = resize(img, width=1000)

    fs = faces(img)
    for face in fs:
        lands = landmarks(img, face)
        dists = all_distances(lands, as_dict=False)
        stdev = stdeviation([dist[2] for dist in dists])
        feed_average(stdev)

img_names = ('hd-girl.jpg', 'hd-boy.jpg', 'me.jpg', 'us.jpg', 'mona-lisa.jpg')

""" --- MAIN --- """

# TODO: Add distances between all points to the file
if __name__ == '__main__':
    for img_name in img_names:
        feed_avg_std(img_name)
