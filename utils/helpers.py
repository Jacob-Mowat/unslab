from numpy import ndarray
from typing import List
import cv2 as cv

def show(image: ndarray, title: str = 'image'):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
