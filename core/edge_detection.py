import imageio
import numpy
import cv2 as cv
from numpy.lib.function_base import cov

class EdgeDetector:
    image: numpy.ndarray
    drawing: bool
    mode: bool
    ix: int
    iy: int
    img2: numpy.ndarray

    threshold: int

    def __init__(self, image_rgb: numpy.ndarray):
        self.image = image_rgb
        self.mode = True
        self.ix = -1
        self.iy = -1
        self.drawing = False

        # Tunable parameters
        self.threshold = 170

    def draw_rectangle(self, event, x, y, flags, param):
        overlay = self.image.copy()
        output = self.image.copy()
        alpha = 0.5

        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.__setattr__ == True:
                    cv.rectangle(overlay, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
                    cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, self.img2)
                    cv.imshow('image', self.img2)

        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True:
                cv.rectangle(overlay, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
                cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, self.image)
    def run_draw_rectangle(self):
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.draw_rectangle)

        self.img2 = self.image.copy()

        while(1):
            cv.imshow('image', self.image)
            k = cv.waitKey(1) & 0xFF
            if k == ord('m'):
                self.mode = not self.mode
            elif k == 27:
                break

        cv.destroyAllWindows()
    def convert_to_gray(self):
        fixed = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)
        return cv.cvtColor(fixed, cv.COLOR_BGR2GRAY)

    def detect_edges_contour(self):
        print("Detecting edges...")
        imgray = self.convert_to_gray()
        ret, thresh = cv.threshold(imgray, self.threshold, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        show(thresh)
        # contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS )
        # CHAIN_APPROX_NONE
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )


        return contours
