import pprint
from time import sleep

import cv2


def pp(obj, indent=4):
    """ Pretty printer for debugging purposes """
    p = pprint.PrettyPrinter(indent=indent)
    p.pprint(obj)


def throw(msg):
    raise Exception(msg)


def cvwait(time=0):
    cv2.waitKey(time)


def sl(seconds):
    sleep(seconds)
