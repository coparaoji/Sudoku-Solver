import cv2
import numpy as np

def show_image(img : np.ndarray):
    cv2.imshow('here', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()