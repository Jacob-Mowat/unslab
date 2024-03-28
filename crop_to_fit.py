import cv2
import unslab.core.crop_to_outer as crop_to_outer

# Load the Pokémon card image
image = cv2.imread('/Users/jacob/Documents/unprocessed_slabs/c1warped.png')
original_image = image.copy()

# Crop the image to fit the Pokémon card
crop_to_outer = crop_to_outer.CropToOuter(show_images=True)
cropped_image = crop_to_outer.run(image)
