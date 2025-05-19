import os
import numpy as np
from PIL import Image, ImageDraw
import re

def read_segments(segments_file):
    segments = []
    with open(segments_file, 'r') as file:
        for line in file:
            # Parse each line using regex
            match = re.match(r"Instance ID: (\d+), Class: (\d+), Centroid: \[(\d+), (\d+)\]", line)
            if match:
                inst_id, class_id, y, x = map(int, match.groups())
                segments.append((inst_id, class_id, [y, x]))
    return segments

def create_graph_image(objects_info, adjacency_matrix, colors):
    # Create a black background image
    image_size = (1024, 1024)  # Adjust size as needed
    image = Image.new('RGBA', image_size, (0, 0, 0, 255))
    overlay = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw connections based on the adjacency matrix
    for i, (_, _, centroid_i) in enumerate(objects_info):
        for j, (_, _, centroid_j) in enumerate(objects_info):  # Changed from object_info to objects_info
            if adjacency_matrix[i, j] != 0:  # Assuming non-zero values indicate a connection
                y1, x1 = centroid_i
                y2, x2 = centroid_j
                coords = [x1, y1, x2, y2]
                draw.line(coords, fill='white', width=2)
    
    # Draw the centroids as nodes
    for segment_id, segment_class, centroid in objects_info:
        color = tuple(colors[segment_class]) + (255,)  # Add alpha channel
        y, x = centroid
        draw.ellipse((x-10, y-10, x+10, y+10), fill=color)

    combined = Image.alpha_composite(image, overlay)
    return combined

def find_removed_node(original_adj, modified_adj):
    # Ensure matrices are 2D
    if original_adj.ndim == 1:
        original_size = int(np.sqrt(original_adj.shape[0]))
        original_adj = original_adj.reshape(original_size, original_size)
    if modified_adj.ndim == 1:
        modified_size = int(np.sqrt(modified_adj.shape[0]))
        modified_adj = modified_adj.reshape(modified_size, modified_size)
    
    # Find the difference in shape between original and modified adjacency matrices
    original_shape = original_adj.shape[0]
    modified_shape = modified_adj.shape[0]
    
    # The removed node would be reflected in the size difference
    if original_shape > modified_shape:
        # Compare each row/column to find which node was removed
        for i in range(original_shape):
            # Create a 2D mask
            mask = np.ones((original_shape, original_shape), dtype=bool)
            mask[i, :] = False
            mask[:, i] = False
            
            # Get indices for the remaining elements
            remaining_indices = np.where(mask.ravel())[0]
            
            # Extract submatrix excluding i-th row and column
            submatrix = original_adj.ravel()[remaining_indices].reshape(modified_shape, modified_shape)
            
            # Compare with modified matrix
            if np.allclose(submatrix, modified_adj):
                return i
    return None

def main():
    base_original = path/to/preprocessed
    base_modified = path/to/generated_graphs/removed_node
    
    colors = np.array([[0, 0, 0],  # Black
                      [255, 0, 0],  # Red
                      [0, 255, 0],  # Green
                      [0, 0, 255],  # Blue
                      [0, 255, 255],  # Cyan
                      [255, 0, 255],  # Magenta
                      [255, 255, 0],  # Yellow
                      [139, 69, 19],  # Brown
                      [128, 0, 128],  # Purple
                      [255, 140, 0],  # Orange
                      [255, 255, 255]], dtype=np.uint8)  # White

    # First, get all unique result numbers
    result_nums = set()
    for result_dir in os.listdir(base_modified):
        if result_dir.startswith('results'):
            result_num = result_dir.split('_')[0]
            result_nums.add(result_num)

    # For each result number, process both modifications
    for result_num in result_nums:
        # Original paths
        original_path = os.path.join(base_original, result_num)
        
        # Process both modifications (1 and 2)
        for mod_num in [1, 2]:
            modified_dir = f"{result_num}_modified_{mod_num}"
            modified_path = os.path.join(base_modified, modified_dir)
            
            if not os.path.exists(modified_path):
                print(f"Skipping {modified_dir} - directory does not exist")
                continue
            
            # Load adjacency matrices
            original_adj = np.load(os.path.join(original_path, 'adjacency_matrix_inverse.npy'))
            modified_adj = np.load(os.path.join(modified_path, 'adjacency_matrix_inverse.npy'))
            
            # Find removed node
            removed_node_idx = find_removed_node(original_adj, modified_adj)
            
            if removed_node_idx is not None:
                # Load segments and adjacency matrix for visualization
                segments = read_segments(os.path.join(original_path, 'segments.txt'))
                abs_adj = np.load(os.path.join(original_path, 'adjacency_matrix_absolute.npy'))
                
                # Remove the identified node from segments and adjacency matrix
                segments = [seg for i, seg in enumerate(segments) if i != removed_node_idx]
                abs_adj = np.delete(np.delete(abs_adj, removed_node_idx, 0), removed_node_idx, 1)
                
                # Create and save the graph image
                graph_image = create_graph_image(segments, abs_adj, colors)
                output_path = os.path.join(modified_path, f'graph_removed_node_{removed_node_idx}.png')
                graph_image.save(output_path)
                print(f"Processed {result_num} modification {mod_num} - Removed node {removed_node_idx}")

if __name__ == "__main__":
    main()
