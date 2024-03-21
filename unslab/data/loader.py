from os import listdir
from typing import List
import rawpy
import imageio
import numpy

class RAWLoader:
    path: str
    found_images: List[str]

    raw_images: List[rawpy.RawPy]
    rgb_images: List[numpy.ndarray]

    def __init__(self, path):
        self.path = path

    def load_raw_images(self) -> None:
        # If images are not found, find them
        self.findRAWImagesInDir() if self.images is None else None

        # Create a list of paths to the images
        paths: List[str] = [self.path + img for img in self.images]

        # Load the raw images
        self.raw_images = [rawpy.imread(path) for path in paths]

    def post_process_raw_images(self) -> None:
        # If raw images are not loaded, load them
        self.load_raw_images() if self.raw_images is None else None

        # Post process the raw images
        self.rgb_images = [img.postprocess(no_auto_bright=True,use_auto_wb=True,gamma=None) for img in self.raw_images]

    def findRAWImagesInDir(self) -> None:
        # Find all the files in the directory -> assumes all files are images
        self.images = listdir(self.path)

    def get_images(self) -> List[numpy.ndarray]:
        return self.rgb_images
