import numpy as np
import os
from PIL import Image

def convert_npys_to_custom_colored_pngs(input_dir, output_dir):
    # Define the custom color map
    colors = np.array([
        [0, 0, 0],       # Black
        [255, 0, 0],     # Red
        [0, 255, 0],     # Green
        [0, 0, 255],     # Blue
        [0, 255, 255],   # Cyan
        [255, 0, 255],   # Magenta
        [255, 255, 0],   # Yellow
        [139, 69, 19],   # Brown (saddlebrown)
        [128, 0, 128],   # Purple
        [255, 140, 0],   # Orange
        [255, 255, 255]  # White
    ], dtype=np.uint8)
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all .npy files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            # Load the .npy file
            npy_path = os.path.join(input_dir, filename)
            index_data = np.load(npy_path)
            
            # Map indices to colors
            colored_data = colors[index_data]
            
            # Convert numpy array to image
            img = Image.fromarray(colored_data, 'RGB')
            
            # Save the image to the output directory with the same name but with .png extension
            png_path = os.path.join(output_dir, filename.replace(".npy", ".png"))
            img.save(png_path)
            print(f"Saved {png_path}")

# Specify your input and output directories
input_dir = 
output_dir = 

# Call the function
convert_npys_to_custom_colored_pngs(input_dir, output_dir)
