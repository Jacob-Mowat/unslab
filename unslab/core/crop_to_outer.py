
import cv2

def show(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class CropToOuter:
    def __init__(self, show_images=False):
        threshold_upper = 255
        threshold_lower = 110
        area_threshold = 5000

        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.area_threshold = area_threshold
        self.show_images = show_images

    def find_contours(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.show_images: show(gray, "Gray")

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        if self.show_images: show(blurred, "Blurred")

        # Thresholding to separate card and slab borders
        _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)
        if self.show_images: show(thresh, "Thresholded Image")

        # Find contours in the grayscale image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours, hierarchy

    def filter_contours(self, contours, hierarchy):
        # Filter contours based on area, aspect ratio, etc.
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > self.area_threshold]

         # Filter contours based on hierarchy
        valid_contours = []
        for i, contour in enumerate(filtered_contours):
            parent_contour = hierarchy[0][i][3]  # Index of parent contour
            if parent_contour != -1:  # Contour has parent (external contour)
                # has no children (internal contours)
                if hierarchy[0][parent_contour][2] != -1:
                    valid_contours.append(contour)

        return valid_contours

    def find_outer_contour(self, valid_contours):
        # Find outermost contour
        return max(valid_contours, key=cv2.contourArea)

    def transform(self, outer_contour, image):
        # Get bounding box of the outermost contour
        x, y, w, h = cv2.boundingRect(outer_contour)

        # Crop the image to fit the bounding box
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image

    def run(self, image):
        contours, hierarchy = self.find_contours(image)
        valid_contours = self.filter_contours(contours, hierarchy)
        outer_contour = self.find_outer_contour(valid_contours)
        cropped_image = self.transform(outer_contour, image)

        if self.show_images: show(cropped_image, "Cropped Image")

        return cropped_image
