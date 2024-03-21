from data.loader import RAWLoader
from unslab.core.edge_detection import EdgeDetector, AVAILIBLE_ALGORITHMS
from typing import List
import numpy

def main():
    fronts_loader = RAWLoader("/Users/jacob/documents/unprocessed_slabs/CharizardEX/front/")
    fronts_loader.post_process_raw_images()

    images: List[numpy.ndarray] = fronts_loader.get_images()

    edge_detector = EdgeDetector(algorithm=AVAILIBLE_ALGORITHMS.CANNY, threshold=170, sigma=0.33)
    for image in images:
        edge_detector.set_image(image)
        edge_detector.run()
        edge_detector.reset()


if __name__ == "__main__":
    main()
