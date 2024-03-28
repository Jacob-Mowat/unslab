from os import listdir
from typing import List
import rawpy
import imageio
import numpy

class RAWLoader:
    path: str
    found_images: List[str]

    raw_images: List[rawpy.RawPy] | None
    rgb_images: List[numpy.ndarray] | None

    def __init__(self, path):
        self.path = path
        self.images = None
        self.raw_images = None
        self.rgb_images = None

    def load_raw_images(self) -> None:
        # If images are not found, find them
        if self.images is None:
            self.findRAWImagesInDir()

            if self.images is None:
                print("No images found in directory")
                return

        # Create a list of paths to the images
        paths: List[str] = [self.path + img for img in self.images]

        # Load the raw images
        self.raw_images = [rawpy.imread(path) for path in paths]

    def post_process_raw_images(self) -> None:
        # If raw images are not loaded, load them
        if self.raw_images is None:
            self.load_raw_images()

            if self.raw_images is None:
                print("No raw images loaded")
                return

        # Post process the raw images
        self.rgb_images = [img.postprocess(no_auto_bright=False,use_auto_wb=False,gamma=None) for img in self.raw_images]

    def findRAWImagesInDir(self) -> None:
        # Find all the files in the directory -> assumes all files are images
        self.images = listdir(self.path)

    def get_images(self) -> List[numpy.ndarray]:
        return self.rgb_images

class PNGLoader:
    path: str
    found_images: List[str]

    rgb_images: List[numpy.ndarray] | None

    def __init__(self, path):
        self.path = path
        self.images = None
        self.rgb_images = None

        self.load_png_images()

    def load_png_images(self) -> None:
        # If images are not found, find them
        if self.images is None:
            self.findPNGImagesInDir()

            if self.images is None:
                print("No images found in directory")
                return

        # Create a list of paths to the images
        paths: List[str] = [self.path + img for img in self.images]

        # Load the raw images
        self.rgb_images = [imageio.imread(path) for path in paths]

    def findPNGImagesInDir(self) -> None:
        # Find all the files in the directory -> assumes all files are images
        files = listdir(self.path)
        self.images = [file for file in files if file.endswith(".png")]

    def get_images(self) -> List[numpy.ndarray]:
        return self.rgb_images
