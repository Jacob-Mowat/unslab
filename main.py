from data.loader import RAWLoader
from core.edge_detection import EdgeDetector
from typing import List
import numpy

def main():
    fronts_loader = RAWLoader("/Users/jacob/documents/unprocessed_slabs/CharizardEX/front/")
    fronts_loader.load_raw_images()
    fronts_loader.post_process_raw_images()

    images: List[numpy.ndarray] = fronts_loader.get_images()

    for image in images:
        edge_detector = EdgeDetector(image)
        edge_detector.detect_edges_contour()
        # edge_detector.use_canny()


if __name__ == "__main__":
    main()
