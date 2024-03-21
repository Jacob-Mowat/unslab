import imageio
import numpy
import cv2 as cv

class EdgeDetector:
    image: numpy.ndarray

    def __init__(self, image_rgb: numpy.ndarray):
        self.image = image_rgb

    def detect_edges_contour(self):
        cv.imshow('image',self.image)
        cv.waitKey(0)
        fixed = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)
        cv.imshow('image', fixed)
        cv.waitKey(0)
        imgray = cv.cvtColor(fixed, cv.COLOR_BGR2GRAY)
        print(imgray.shape)
        cv.imshow('image',imgray)
        cv.waitKey(0)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )

        cv.drawContours(self.image, contours, -1, (0,255,0), 6)
        cv.imshow('image', self.image)

        return contours
