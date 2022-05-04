import cv2 as cv
import os

def open_image(path):
    x = cv.imread(path,0)
    if (x is None):
        raise Exception("File not found")
    return x

    