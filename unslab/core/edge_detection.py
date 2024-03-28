from typing import Sequence
from cv2.typing import MatLike
import imageio
import numpy
import cv2 as cv
from numpy.core.multiarray import ndarray
from numpy.lib.function_base import cov
from unslab.utils.helpers import show
import math
from enum import Enum
from rembg import remove as remove_bg

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
    image: numpy.ndarray | None
    drawing: bool
    mode: bool
    ix: int
    iy: int
    img2: numpy.ndarray | None

    threshold: int
    sigma: float

    def __init__(
        self,
        image_rgb: numpy.ndarray | None = None,
        threshold: int = 170,
        sigma: float = 0.33,
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
        self.sigma = sigma

    def run(self):
        if self.image is None:
            print("No image to process")
            return

        if self.algorithm == AVAILIBLE_ALGORITHMS.CANNY:
            self.use_canny()
        elif self.algorithm == AVAILIBLE_ALGORITHMS.SUSUKI:
            self.use_susuki()

    def reset(self):
        self.image = None
        self.drawing = False
        self.mode = True
        self.ix = -1
        self.iy = -1
        self.img2 = None

    def set_image(self, image: numpy.ndarray):
        self.image = image

    def draw_rectangle(self, event, x, y, flags, param):
        if self.image is None or self.img2 is None:
            return

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
        if self.image is None:
            return

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


    def convert_rgb_to_bgr(self, image: numpy.ndarray):
        return cv.cvtColor(image, cv.COLOR_RGB2BGR)

    def convert_to_gray(self, image: numpy.ndarray):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def use_susuki(self):
        if self.image is None:
            return

        print("Detecting edges using Susuki...")

        # Remove background
        # new_img = remove(imgray)
        #

        # Convert to BGR from RGB
        bgr_image: MatLike = self.convert_rgb_to_bgr(self.image)
        show(bgr_image, "BGR")

        def custom_remove_bg():
            global bgr_image
            # Remove background using HSV Mask
            # Convert to HSV
            hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)

            # show(hsv, "HSV")

            # Define range of white color in HSV
            lower_white = numpy.array([0, 0, 0], dtype=numpy.uint8)
            upper_white = numpy.array([255, 255, 255], dtype=numpy.uint8)

            # Threshold the HSV image to get only white colors
            mask = cv.inRange(hsv, lower_white, upper_white)

            # Bitwise-AND mask and original image
            return cv.bitwise_and(hsv, bgr_image, mask=mask)

        # nobg_image = remove_bg(bgr_image)

        # Convert to grayscale
        gray = self.convert_to_gray(bgr_image)

        show(gray, "Gray")

        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        show(blurred, "Blurred")

        ret, thresh = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE )

        self.img2 = bgr_image.copy()
        cv.drawContours(self.img2, contours, -1, (0,255,0), 3)

        print(f"Using threshold: {ret}/{self.threshold}")
        show(self.img2)

        newContours: Sequence[MatLike] = [numpy.array(0) for _ in range(len(contours))]

        # Aproximate contours using Douglas-Peucker algorithm
        for i, contour in enumerate(contours):
            newContours[i] = cv.approxPolyDP(contour, 0.00027*cv.arcLength(contour, True), True)

        max_contour = max(contours, key=cv.contourArea)
        print(f"Max contour area: {cv.contourArea(max_contour)}")

        # Create a mask of the slab contour
        mask = numpy.zeros_like(gray)
        cv.drawContours(mask, [max_contour], -1, (0, 255, 0), thickness=cv.FILLED)

        # Invert the mask
        mask = cv.bitwise_not(mask)

        # Apply the mask to the original image to remove the slab
        original_image = self.image.copy()
        result = cv.bitwise_and(original_image, original_image, mask=mask)

        show(result, "Result")

        self.img2 = self.image.copy()
        cv.drawContours(self.img2, newContours, -1, (0,255,0), 3)
        show(self.img2)

        # self.erode_and_dilate()

        # def erode_and_dilate(self):
        #     # Run a minimum area filter:
        #     minArea = 50
        #     mask = areaFilter(minArea, self.img2)

        #     # Pre-process mask:
        #     kernelSize = 5

        #     structuringElement: MatLike = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))
        #     iterations = 2

        #     mask = cv.morphologyEx(mask, cv.MORPH_DILATE, structuringElement, None, None, iterations, cv.BORDER_REFLECT101)
        #     mask = cv.morphologyEx(mask, cv.MORPH_ERODE, structuringElement, None, None, iterations, cv.BORDER_REFLECT101)

        #     erosion = cv.erode(thresh, kernel, iterations=2)
        #     dilation = cv.dilate(erosion, kernel, iterations=2)
        #     return dilation

        return newContours

    def use_canny(self):
        print("Detecting edges using Canny...")

        imgray = self.convert_to_gray()

        if imgray is None:
            return

        # TODO: (@Jacob-Mowat) - Possibly add  HSV Mask

        # Remove background
        new_img = remove(imgray)

        # median_v = numpy.median(list(new_img))
        # lower = int(max(0, (1.0 - self.sigma) * median_v + 100))
        # upper = int(min(255, (1.0 + self.sigma) * median_v + 20))

        edges = cv.Canny(new_img, threshold1=5.0, threshold2=25.0, L2gradient=True)
        show(edges, "Canny Edge Detection")
