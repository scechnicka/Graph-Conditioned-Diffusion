import numpy as np
import os

# Your specified colors array
colors = np.array([
    [0, 0, 0],    # Black -> 0
    [255, 0, 0],  # Red -> 1
    [0, 255, 0],  # Green -> 2
    [0, 0, 255],  # Blue -> 3
    [0, 255, 255] # Cyan -> 4
], dtype=np.uint8)

def find_color_index(color):
    # Find the index of the color in the colors array
    for i, c in enumerate(colors):
        if np.array_equal(c, color):
            return i
    return -1 # Return -1 if the color is not found

def convert_file(filename):
    # Load the NPY file
    data = np.load(filename)
    
    # Create a new empty array of the same height and width as the original but with only one channel
    new_data = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
    
    # Iterate over each pixel and set the corresponding index value
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            color = data[x, y]
            color_index = find_color_index(color)
            new_data[x, y] = color_index
            
    # Save the new NPY file using the original filename, effectively replacing it
    np.save(filename, new_data)


# Example usage
directory = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/modified/image/changed_labels/total"

for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        filepath = os.path.join(directory, filename)
        convert_file(filepath)
        print(f"Converted {filename}")
