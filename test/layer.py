import setup
import cv2
from core import layer
from config import DATASET_PATH

""" MAIN """
orig = cv2.imread('original.jpg')
equa = cv2.imread('equalized.jpg')
refe = cv2.imread('reference.jpg')

layer_no = 0
tmp = layer(refe, layer_no)

cv2.imwrite('reference_layer_0.jpg', tmp)
