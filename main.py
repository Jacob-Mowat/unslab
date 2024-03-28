from unslab.data.loader import PNGLoader
from unslab.core.edge_detection import EdgeDetector, AVAILIBLE_ALGORITHMS
from typing import List
import numpy

def main():
    fronts_loader = PNGLoader("/Users/jacob/documents/unprocessed_slabs/")

    images: List[numpy.ndarray] = fronts_loader.get_images()

    edge_detector = EdgeDetector(algorithm=AVAILIBLE_ALGORITHMS.SUSUKI, threshold=170, sigma=0.36)
    for image in images:
        edge_detector.set_image(image)
        edge_detector.run()
        edge_detector.reset()


if __name__ == "__main__":
    main()
