from typing import Sequence
from cv2.typing import MatLike
import imageio
import numpy
import cv2 as cv
from numpy.lib.function_base import cov
from unslab.utils.helpers import show
import math
from enum import Enum

class AVAILIBLE_ALGORITHMS(Enum):
    SUSUKI="suzuki"
    CANNY="canny"
    # TODO: (@Jacob-Mowat) - Implement these algorithms
    # SOBEL="sobel"
    # LAPLACIAN="laplacian"
    # PREWITT="prewitt"
    # ROBERTS="roberts"
    # SCHARR="scharr"

class EdgeDetector:
    image: numpy.ndarray
    drawing: bool
    mode: bool
    ix: int
    iy: int
    img2: numpy.ndarray

    threshold: int

    def __init__(
        self,
        image_rgb: numpy.ndarray,
        threshold: int = 170,
        algorithm: AVAILIBLE_ALGORITHMS = AVAILIBLE_ALGORITHMS.CANNY
    ):
        self.image = image_rgb
        self.mode = True
        self.ix = -1
        self.iy = -1
        self.drawing = False

        # Tunable parameters
        self.threshold = 170
        self.algorithm = algorithm

    def run(self):
        if self.algorithm == AVAILIBLE_ALGORITHMS.CANNY:
            self.use_canny()
        elif self.algorithm == AVAILIBLE_ALGORITHMS.SUSUKI:
            self.use_susuki()

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


        self.img2 = self.image.copy()
        cv.drawContours(self.img2, contours, -1, (0,255,0), 3)

        print(f"Using threshold: {ret}/{self.threshold}")
        show(self.img2)

        newContours: Sequence[MatLike] = [numpy.array(0) for _ in range(len(contours))]

        # Aproximate contours
        for i, contour in enumerate(contours):
            newContours[i] = cv.approxPolyDP(contour, 0.0001*cv.arcLength(contour, True), True)

        self.img2 = self.image.copy()
        cv.drawContours(self.img2, newContours, -1, (0,255,0), 3)

        print(f"Using threshold: {ret}/{self.threshold}")
        show(self.img2)

        return contours

    def use_canny(self):
        print("Using Canny edge detection...")

        imgray = self.convert_to_gray()

        sigma = 0.36

        median_v = numpy.median(list(imgray))
        lower = int(max(0, (1.0 - sigma) * median_v))
        upper = int(min(255, (1.0 + sigma) * median_v))

        edges = cv.Canny(imgray, threshold1=lower, threshold2=upper, L2gradient=True)
        show(edges, "Canny Edge Detection")
