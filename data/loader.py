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
        self.findRAWImagesInDir()

        paths: List[str] = [self.path + img for img in self.images]
        self.raw_images = [rawpy.imread(path) for path in paths]

    def post_process_raw_images(self) -> None:
        self.load_raw_images() if self.raw_images is None else None
        self.rgb_images = [img.postprocess(no_auto_bright=True,use_auto_wb=False,gamma=None) for img in self.raw_images]

    def findRAWImagesInDir(self) -> None:
        self.images = listdir(self.path)

    def get_images(self) -> List[numpy.ndarray]:
        return self.rgb_images
