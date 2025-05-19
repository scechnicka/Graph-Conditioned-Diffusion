import os
import numpy as np
import cv2
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import euclidean
import networkx as nx
from matplotlib import pyplot as plt
import glob
from matplotlib import cm
import scipy.ndimage as ndimage
from PIL import Image,ImageDraw
from matplotlib.colors import ListedColormap
from scipy.ndimage import find_objects
import math


colors = ['red','green','blue','cyan']

colors = np.array([[0, 0, 0],  # Black
                       [255, 0, 0],  # Red
                       [0, 255, 0],  # Green
                       [0, 0, 255],  # Blue
                       [0, 255, 255],  # Cyan
                       [255, 0, 255],  # Magenta
                       [255, 255, 0],  # Yellow
                       [139, 69, 19],  # Brown (saddlebrown)
                       [128, 0, 128],  # Purple
                       [255, 140, 0],  # Orange
                       [255, 255, 255]], dtype=np.uint8)  # White
                       
def array_to_colormap(labelmap,colors):
    """
    Method used to generate rgb label maps for tensorboard visualization
    :param labelmap: HxW label map tensor containing values from 0 to n_classes
    :return: 3xHxW RGB label map containing colors in the following order: Black (background), Red, Green, Blue, Cyan, Magenta, Yellow, Brown, Orange, Purple
    source: https://github.com/NBouteldja/KidneySegmentation_Histology
    """
    # Initialize the RGB image
    rgb_image = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
    
    # Apply colors to each label
    for label in np.unique(labelmap):
        if label < len(colors):  # Ensure label is within the bounds of colors array
            label_int = int(label)  # Explicitly convert label to int
            rgb_image[labelmap == label] = colors[label_int]
        else:
            print(f"Warning: Label {label} exceeds color map bounds. Skipping.")
    
    return rgb_image



# Function to calculate the center of mass for each segment
def calculate_objects(label_mask):

    instance_labels = np.zeros_like(label_mask);
    first_label = 0
    com = []
    for k in range(1,5):
        l=(label_mask==k).astype(np.uint8)
        label, nb = ndimage.label(l)
        for i in range(1,np.max(label)+1):
            indices = np.argwhere(label==i)
            # Skip segments with fewer than 5 pixels
            if len(indices) < 5:
                continue
            s = np.sum(indices, axis=0)
            
            inst_id = i + first_label
            com += [(inst_id, k, [int(s[0]/np.shape(indices)[0]), int(s[1]/np.shape(indices)[0])])]
            instance_labels[label == i] = inst_id
        first_label += nb
    return com, instance_labels

# Function to create and save individual segments
def save_individual_segments(image, instance_map, objects_info, output_folder):
    
    for inst_id, k, c in objects_info:  # Skipping background which is assumed to be label 0

        
        # Use the slice to extract the segment
        segment_image = np.zeros_like(image)
        segment_mask = (instance_map == inst_id)
        segment_image[segment_mask] = image[segment_mask]

        # Calculate the center of mass for the current segment
        # Note: You'll need to adjust the center of mass to the full image coordinates if necessary
        #print(centers_of_mass)
        # com = centers_of_mass[label_val - 1]

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Construct the filename based on the center of mass, adjusting indices for the slice
        filename_png = f"{output_folder}/segment_{inst_id}_{int(c[0])}_{int(c[1])}_class_{k}.png"
        filename_npy = f"{output_folder}/segment_{inst_id}_{int(c[0])}_{int(c[1])}_class_{k}.npy"

        
        # Save the segment image
        cv2.imwrite(filename_png, segment_image)
        
        # Save the numpy array representing the segment image as NPY
        np.save(filename_npy, segment_image)

# Main function to process each image
def process_image(npz_file_path, output_path):
    # Load data
    data = np.load(npz_file_path)
    image = data['image']
    label_mask = data['label']
    
    # Convert and save the image and label_mask as PNG files
    image_path = os.path.join(output_path, 'image.png')
    label_mask_path = os.path.join(output_path, 'label_mask.png')
    
    # Paths for saving as NPY files
    image_npy_path = os.path.join(output_path, 'image.npy')
    label_mask_npy_path = os.path.join(output_path, 'label_mask.npy')
    
    
    # Check if image is grayscale or RGB, and save it
    if image.ndim == 2:  # Grayscale
        cv2.imwrite(image_path, image)
    elif image.ndim == 3:  # RGB
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr)
        
    # Save the numpy array of the real image as NPY
    np.save(image_npy_path, image)
    
    # For label mask, apply the custom colormap for visualization and save it
    label_mask_color = array_to_colormap(label_mask, colors)
    # Convert RGB (PIL/OpenCV) to BGR (OpenCV) before saving
    label_mask_color_bgr = cv2.cvtColor(label_mask_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(label_mask_path, label_mask_color_bgr)
    
    # Save the original label mask numpy array as NPY
    np.save(label_mask_npy_path, label_mask)
    
    # Create an RGBA version of the image
    if image.ndim == 2:  # Grayscale
        image_rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    else:  # RGB
        image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)

    # Create an RGBA version of the label_mask
    label_mask_rgba = np.zeros((label_mask.shape[0], label_mask.shape[1], 4), dtype=np.uint8)
    # Use your custom colormap function here to get the RGB values
    label_mask_rgb = array_to_colormap(label_mask, colors)  # Assuming this function returns an RGB image
    label_mask_rgba[..., :3] = label_mask_rgb
    
    # Create alpha mask 40% opaque in mask region
    alpha_mask = np.ones_like(label_mask) * 255  # Fully opaque by default
    alpha_mask[label_mask > 0] = int(0.4 * 255)  # 40% opaque where there is a label
    label_mask_rgba[..., 3] = alpha_mask  # Set the alpha channel
    
    # Create a combined image with correct background and semi-transparent mask
    combined_image = np.where(label_mask_rgba[:, :, 3:4] == int(0.4 * 255),  # Where mask is present
                              cv2.addWeighted(image_rgba, 0.6, label_mask_rgba, 0.4, 0),  # Blend image and mask
                              image_rgba)  # Where mask is not present, use original image
    
    # Save the combined image
    combined_image_path = os.path.join(output_path, 'mask_and_image.png')
    cv2.imwrite(combined_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGBA2BGR))
    # Save the combined image as NPY
    # Since NPY doesn't directly support RGBA, convert to RGB if the original image is RGB
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_RGBA2RGB)
    combined_image_npy_path = os.path.join(output_path, 'mask_and_image.npy')
    np.save(combined_image_npy_path, combined_image_rgb) 
       
    # Calculate the centers of mass
    objects_info, instance_map = calculate_objects(label_mask)
    
    #Create the adjacency matrix
    adjacency_matrix_absolute,adjacency_matrix_inverse = create_adjacency_matrix(objects_info, instance_map)
    
    #  Save the adjacency matrix
    np.save(os.path.join(output_path, 'adjacency_matrix_absolute.npy'), adjacency_matrix_absolute)
    np.save(os.path.join(output_path, 'adjacency_matrix_inverse.npy'), adjacency_matrix_inverse)   
    #Create the overlay image graph
    # Inside the process_image function
    overlay_image = create_overlay_image(image, objects_info, adjacency_matrix_absolute)

    # Save the overlay image and graph
    overlay_image_path = os.path.join(output_path, 'overlay_image_graph.png')
    overlay_image.save(overlay_image_path)
    
    # Save overlay_image as NPY
    overlay_image_npy_path = os.path.join(output_path, 'overlay_image_graph.npy')
    overlay_image_np = np.array(overlay_image.convert('RGB'))  # Convert PIL Image to NumPy array
    np.save(overlay_image_npy_path, overlay_image_np)

    overlay_image_withmask = create_overlay_image(combined_image, objects_info, adjacency_matrix_absolute)

    # Save the overlay image mask and graph
    overlay_image_path_mask = os.path.join(output_path, 'overlay_image_graph_and_mask.png')
    overlay_image_withmask.save(overlay_image_path_mask)
    
    # Save overlay_image_withmask as NPY
    overlay_image_withmask_npy_path = os.path.join(output_path, 'overlay_image_graph_and_mask.npy')
    overlay_image_withmask_np = np.array(overlay_image_withmask.convert('RGB'))  # Convert PIL Image to NumPy array
    np.save(overlay_image_withmask_npy_path, overlay_image_withmask_np)
    
        
    black_image = np.zeros_like(label_mask_rgb)
    graph = create_overlay_image(black_image, objects_info, adjacency_matrix_absolute)

    # Save the overlay image mask and graph
    graph_path = os.path.join(output_path, 'graph.png')
    graph.save(graph_path)   
    
    # For saving the graph (if it's also a PIL Image and created in a similar fashion)
    graph_npy_path = os.path.join(output_path, 'graph.npy')
    graph_np = np.array(graph.convert('RGB'))  # Assuming 'graph' is a PIL Image
    np.save(graph_npy_path, graph_np)
    
    # Write segment details into segments.txt
    segments_file_path = os.path.join(output_path, 'segments.txt')  # Adjust the path as needed
    with open(segments_file_path, 'w') as file:  # Use 'a' to append or 'w' to overwrite
        for inst_id, k, c in objects_info:
            file.write(f"Instance ID: {inst_id}, Class: {k}, Centroid: {c}\n")

    
    # Save individual segments into a folder named after the image
    segment_folder = os.path.join(output_path, os.path.basename(npz_file_path).split('.')[0])
    os.makedirs(segment_folder, exist_ok=True)
    save_individual_segments(image, instance_map, objects_info, segment_folder)

    return adjacency_matrix_absolute, overlay_image, segment_folder

def create_adjacency_matrix(objects_info, segmentation_mask):
    num_segments = len(objects_info)
    adjacency_matrix_absolute = np.full((num_segments, num_segments), np.inf)
    adjacency_matrix_inverse = np.full((num_segments, num_segments), np.inf)
    for i, (i_id, _, com1) in enumerate(objects_info):
        for j, (j_id, _, com2) in enumerate(objects_info):
            if i == j:
                # Distance to itself is 0
                adjacency_matrix_absolute[i, j] = 1
                adjacency_matrix_inverse[i, j] = 1
            else:
                # Draw a line between CoM1 and CoM2
                line_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
                cv2.line(line_mask, (int(com1[1]), int(com1[0])), (int(com2[1]), int(com2[0])), 1, 1)
                
                # Extract the line pixels using the line mask
                line_pixels = segmentation_mask[line_mask == 1]

                # Check if the line crosses any segment other than the start and end segments
                unique_segments = np.unique(line_pixels)
                if any(s not in [i_id, j_id, 0] for s in unique_segments):
                    # If line crosses a segment other than start and end, nodes are not directly accessible
                    adjacency_matrix_absolute[i, j] = 0
                    adjacency_matrix_inverse[i, j] = 0
                else:
                    # If line does not cross any other segments, calculate and save the euclidean distance
                    adjacency_matrix_absolute[i, j] = euclidean(com1, com2)
                    adjacency_matrix_inverse[i, j] = 1/(math.log2(euclidean(com1, com2)))

    return adjacency_matrix_absolute, adjacency_matrix_inverse

def create_overlay_image(image, object_infos, adjacency_matrix):    
    # Convert numpy array to PIL Image
    if image.dtype == np.uint8:
        # Assuming the array is in the correct uint8 format for PIL
        image = Image.fromarray(image)
    else:
        # If the array is not uint8, it needs to be converted
        image = Image.fromarray((image * 255).astype(np.uint8))
    if image.mode != 'RGBA':
      image = image.convert('RGBA')
    # Create a transparent overlay
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw connections based on the adjacency matrix
    for i, (_, _, centroid_i) in enumerate(object_infos):
        for j, (_, _, centroid_j) in enumerate(object_infos):
            if adjacency_matrix[i, j] != 0:  # Assuming non-zero values indicate a connection
                # Ensure the centroids are swapped from [y, x] to [x, y]
                y1, x1 = centroid_i
                y2, x2 = centroid_j
                # Correctly order and flatten the list of coordinates for drawing
                coords = [x1, y1, x2, y2]
                draw.line(coords, fill='white', width=2)
    
    # Draw the centroids as nodes
    for segment_id, segment_class, centroid in object_infos:
        color = tuple(colors[segment_class]) 
        y,x = centroid
        draw.ellipse((x-10, y-10, x+10, y+10), fill=color)


    # Overlay the graph on the original image
    combined = Image.alpha_composite(image, overlay)
    return combined
  

    

def main(input_folder, base_output_folder):
    # Get a list of all .npz files in the input directory
    npz_files = glob.glob(os.path.join(input_folder, '*.npz'))

    # Process each npz file
    for npz_file in npz_files:
        # Extract the base name of the .npz file to use in creating a unique output subdirectory
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        specific_output_folder = os.path.join(base_output_folder, base_name)
        os.makedirs(specific_output_folder, exist_ok=True)  # Create the directory if it doesn't exist

        print(f"Processing {npz_file}...")
        adjacency_matrix, overlay_image, segment_folder = process_image(npz_file, specific_output_folder)
        '''
        try:
            # Process the image and get the outputs, passing the specific output directory for this file
            adjacency_matrix, overlay_image, segment_folder = process_image(npz_file, specific_output_folder)
            print(f"Finished processing {npz_file}. Results saved to {specific_output_folder}")
        except Exception as e:
            print(f"An error occurred while processing {npz_file}: {e}")
            '''

if __name__ == "__main__":
    # Define the input directory where .npz files are located
    input_folder = path/file/to/npz_train_data
    # Define the output directory where you want to save the results
    base_output_folder = output/path/to_preprocessed_data

    # Run the main function
    main(input_folder, base_output_folder)
