import cv2
from imutils import resize

import setup
from core import faces, landmarks, all_distances
from utils import show_img_with_lands, visible_only
from config import IMG_PATH
from helpers import pp

img = cv2.imread(IMG_PATH + 'hd-girl.jpg')
img = resize(img, width=600)

faces = faces(img)
for face in faces:
    lands = landmarks(img, face)
    pp(all_distances(lands))
