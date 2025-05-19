import numpy as np
import os
from PIL import Image

def color_to_category(np_img):
    """
    Map specific RGB color values to categories (0, 1, 2, 3, 4).

    Args:
        np_img (numpy.ndarray): NumPy array of the image with shape (H, W, 3).

    Returns:
        numpy.ndarray: A 2D NumPy array with categorized values.
    """
    # Define the color to category mapping
    colors = np.array([
        [0, 0, 0],    # Black -> 0
        [255, 0, 0],  # Red -> 1
        [0, 255, 0],  # Green -> 2
        [0, 0, 255],  # Blue -> 3
        [0, 255, 255]  # Cyan # Yellow -> 4 (assuming you meant yellow for the 4th class)
    ], dtype=np.uint8)

    # Initialize an empty array for the categorized image
    categorized_img = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.uint8)

    # Iterate through each color and assign the corresponding category
    for idx, color in enumerate(colors):
        # Find pixels matching the current color
        matches = np.all(np_img == color, axis=-1)

        # Assign the category to matching pixels
        categorized_img[matches] = idx

    return categorized_img

def convert_pngs_to_npys(input_dir, output_dir):
    """
    Convert PNG images in the input directory (including subdirectories) to categorized NPY files,
    maintaining the same directory structure in the output directory.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                image_array = np.array(image)

                # Convert image colors to categories
                categorized_array = color_to_category(image_array)

                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                npy_file_path = os.path.join(output_path, file.replace(".png", ".npy"))
                np.save(npy_file_path, categorized_array)
                print(f"Saved {npy_file_path}")

# Update these paths to match your directory structure
input_dir = path/to/mask_conditioned/labels_pure
output_dir = path/tp/mask_conditioned/labels_npy
convert_pngs_to_npys(input_dir, output_dir)
