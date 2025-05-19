import numpy as np
from PIL import Image
import os

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
        [0, 255, 255] # Cyan -> 4
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

def png_to_npy(png_path, npy_path):
    """
    Convert PNG image to a categorized NumPy array based on specific color values and save as .npy file.
    
    Args:
        png_path (str): Path to the input PNG image.
        npy_path (str): Path to save the output .npy file.
    """
    # Load PNG image using PIL
    img = Image.open(png_path)
    
    # Convert image to NumPy array
    np_img = np.array(img)
    
    # Map colors to categories
    categorized_img = color_to_category(np_img)
    
    # Save categorized NumPy array as .npy file
    np.save(npy_path, categorized_img)

def batch_png_to_npy(png_dir, npy_dir):
    """
    Batch conversion of PNG images to categorized NumPy arrays based on specific color values and save as .npy files.
    
    Args:
        png_dir (str): Directory containing input PNG images.
        npy_dir (str): Directory to save output .npy files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    
    # Iterate through PNG files in the directory
    for filename in os.listdir(png_dir):
        if filename.endswith(".png"):
            png_path = os.path.join(png_dir, filename)
            npy_filename = os.path.splitext(filename)[0] + ".npy"
            npy_path = os.path.join(npy_dir, npy_filename)
            png_to_npy(png_path, npy_path)
            print(f"Converted {filename} to {npy_filename}")

# Example usage
if __name__ == "__main__":
    # Define input and output directories
    png_directory = "/home/atuin/b143dc/b143dc22/GCD_UNETS/mask_conditioned/labels_pure"
    npy_directory = "/home/atuin/b143dc/b143dc22/GCD_UNETS/mask_conditioned/labels_npy"
   # Convert PNG images to categorized NumPy arrays
    batch_png_to_npy(png_directory, npy_directory)