import os
import numpy as np
from PIL import Image, ImageDraw
import re

def read_segments(segments_file):
    segments = []
    with open(segments_file, 'r') as file:
        for line in file:
            match = re.match(r"Instance ID: (\d+), Class: (\d+), Centroid: \[(\d+), (\d+)\]", line)
            if match:
                inst_id, class_id, y, x = map(int, match.groups())
                segments.append((inst_id, class_id, [y, x]))
    return segments

def create_graph_image(objects_info, adjacency_matrix, colors):
    image_size = (1024, 1024)
    image = Image.new('RGBA', image_size, (0, 0, 0, 255))
    overlay = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw connections
    for i, (_, _, centroid_i) in enumerate(objects_info):
        for j, (_, _, centroid_j) in enumerate(objects_info):
            if adjacency_matrix[i, j] != 0:
                y1, x1 = centroid_i
                y2, x2 = centroid_j
                coords = [x1, y1, x2, y2]
                draw.line(coords, fill='white', width=2)
    
    # Draw nodes
    for segment_id, segment_class, centroid in objects_info:
        color = tuple(colors[segment_class]) + (255,)
        y, x = centroid
        draw.ellipse((x-10, y-10, x+10, y+10), fill=color)

    combined = Image.alpha_composite(image, overlay)
    return combined

def find_changed_node(original_features, modified_features):
    # Compare the feature vectors to find which node changed class
    differences = np.abs(original_features - modified_features).sum(axis=1)
    changed_node_idx = np.argmax(differences)
    
    # Verify this is actually a change
    if differences[changed_node_idx] > 0:
        return changed_node_idx
    return None

def main():
   base_original = path/to/preprocessed
   base_modified = path/to/generated_graphs/change_one_node_graph
   
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

   # Get all directories in modified folder
   for dir_name in os.listdir(base_modified):
       if dir_name.endswith('_modified_class'):
            result_num = dir_name.split('_')[0]  # Extract result number
            
            # Construct paths
            original_path = os.path.join(base_original, result_num)
            modified_path = os.path.join(base_modified, dir_name)
            
            # Load feature vectors with corrected filename
            original_features = np.load(os.path.join(original_path, f'{result_num}_feature_vectors_extracted_1024.npy'))
            modified_features = np.load(os.path.join(modified_path, f'{result_num}_modified_feature_vectors_extracted_1024.npy'))

            # Find changed node
            changed_node_idx = find_changed_node(original_features, modified_features)
            
            if changed_node_idx is not None:
                # Load segments and adjacency matrix
                segments = read_segments(os.path.join(original_path, 'segments.txt'))
                abs_adj = np.load(os.path.join(original_path, 'adjacency_matrix_absolute.npy'))
                
                # Determine new class by analyzing modified features
                old_class = segments[changed_node_idx][1]
                # You might need to adjust this logic depending on how classes are encoded in features
                new_class = old_class + 1 if old_class < 4 else 1  # Example logic
                
                # Update the class in segments
                old_segment = segments[changed_node_idx]
                segments[changed_node_idx] = (old_segment[0], new_class, old_segment[2])
                
                # Create and save the graph image
                graph_image = create_graph_image(segments, abs_adj, colors)
                output_path = os.path.join(modified_path, f'graph_changed_node_{changed_node_idx}_class_{old_class}_to_{new_class}.png')
                graph_image.save(output_path)
                print(f"Processed {dir_name} - Changed node {changed_node_idx} from class {old_class} to {new_class}")

if __name__ == "__main__":
    main()
