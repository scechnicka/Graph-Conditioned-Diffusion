import numpy as np
import os
import random
from PIL import Image, ImageDraw

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
      
def decode_feature_vector(feature_vector):
    """
    Decode the feature vector back into usable features.
    """
    # Assuming the first 4 elements are one-hot encoded class information
    class_id = np.argmax(feature_vector[:4]) + 1  # +1 to shift from 0-based index to 1-based class ID
    # Extracting normalized features
    centroid_x, centroid_y, avg_area, bbox_x, bbox_y, bbox_width, bbox_height = feature_vector[4:]
    # Denormalize features
    centroid = (centroid_x * 1024, centroid_y * 1024)
    bounding_box = (bbox_x * 1024, bbox_y * 1024, bbox_width * 1024, bbox_height * 1024)
    
    return class_id, centroid, avg_area, bounding_box

def create_overlay_image_with_decoded_features(base_size, feature_vectors, adjacency_matrix):
    image = Image.new('RGB', base_size, (0, 0, 0)).convert('RGBA')
    overlay = Image.new("RGBA", base_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    object_infos = [decode_feature_vector(fv) for fv in feature_vectors]

    # Draw connections based on adjacency matrix
    for i, obj_info_i in enumerate(object_infos):
        for j, obj_info_j in enumerate(object_infos):
            if adjacency_matrix[i, j] != 0:  # Draw connection if adjacency != 0
                draw.line([obj_info_i[1], obj_info_j[1]], fill='white', width=2)
                
    # Draw nodes with corresponding colors
    for class_id, centroid, _, _ in object_infos:
        # Map class_id to a color
        fill_color = tuple(colors[class_id])  # Convert NumPy color to a PIL-compatible format
        draw.ellipse((centroid[0]-10, centroid[1]-10, centroid[0]+10, centroid[1]+10), fill=fill_color)

    # Composite overlay onto base image
    image_with_overlay = Image.alpha_composite(image, overlay)
    return image_with_overlay.convert('RGB')

def random_augmentation(adjacency_matrix,features):
    # Randomly select an augmentation operation
    operation = random.choice(['rotation', 'sheer', 'flip', 'crop'])
    
    if operation == 'rotation':
        # Perform random rotation
        angle = random.randint(0, 360)
        # Apply rotation to adjacency matrix or any other necessary operations
        adjacency_matrix = np.rot90(adjacency_matrix, k=angle // 90)
        
    elif operation == 'sheer':
        # Perform random sheer
        sheer_factor = random.uniform(0.1, 0.5)
        adjacency_matrix = adjacency_matrix * sheer_factor
        
    elif operation == 'flip':
        # Perform random flip
        flip_direction = random.choice(['horizontal', 'vertical'])
        if flip_direction == 'horizontal':
            adjacency_matrix = np.fliplr(adjacency_matrix)
        else:
            adjacency_matrix = np.flipud(adjacency_matrix)
        

    elif operation == 'crop':
        # Perform random crop
        crop_size = random.randint(0, 30)
        features,adjacency_matrix = crop(features, adjacency_matrix, crop_size)
    return adjacency_matrix, features

def crop(features, image, size):
    width, height = image.shape
    x = random.randint(0, width - size)
    y = random.randint(0, height - size)
    cropped_image = image[x:x+size, y:y+size]
    cropped_features = features[x:x+size,:]
    return cropped_image, cropped_features

def augment_graphs(adjacency, features):
    augmented_adjacency_matrix,augmented_features = random_augmentation(adjacency,features)
    return augmented_features, augmented_adjacency_matrix
def generate_augmented_graphs(base_dir, output_dir, num_graphs):
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]
    
    for i in range(num_graphs):
        graph_dir = random.sample(folders, 1)
        
        features = np.load(os.path.join(base_dir, graph_dir, f"{graph_dir}_feature_vectors_manual.npy"))
        adjacency = np.load(os.path.join(base_dir, graph_dir, "adjacency_matrix_inverse.npy"))
        print(np.shape(features),graph_dir)
        augmented_features,augmented_adjacency = augment_graphs(adjacency,features)
        base_size = (1024, 1024)
        np.save(os.path.join(output_dir, f"augmented{i+1}_features.npy"), augmented_features)
        np.save(os.path.join(output_dir, f"augmented{i+1}_adjacency.npy"), augmented_adjacency)
        image_with_overlay = create_overlay_image_with_decoded_features(base_size, augmented_features, augmented_adjacency)
        image_with_overlay.save(os.path.join(output_dir, f"augmented{i+1}_graph.png"))

    

# Example usage
base_dir = path/to/preprocessed
output_dir = path/to/generated_graphs/augmented/ 
num_graphs = 5
generate_augmented_graphs(base_dir, output_dir, num_graphs)

