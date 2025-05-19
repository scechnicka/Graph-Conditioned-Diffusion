import numpy as np
import os
import random
from PIL import Image, ImageDraw
import networkx as nx
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

def blend_graphs(features_1,adj_1, features_2,adj_2, blend_ratio, output_nodes):
    features_1 = np.array(features_1)
    features_2 = np.array(features_2)
    print(np.shape(features_1), np.shape(features_2))
    adj_1 = np.array(adj_1)
    adj_2 = np.array(adj_2)

    # Determine the larger graph for additional nodes based on size after initial blend
    common_nodes = min(features_1.shape[0], features_2.shape[0])
    larger_graph_features, larger_graph_adj = (features_2, adj_2) if features_2.shape[0] > features_1.shape[0] else (features_1, adj_1)
    remaining_nodes_indices = np.arange(common_nodes, larger_graph_features.shape[0])  # Indices of remaining nodes in the larger graph

    # Calculate the number of additional nodes to include from the larger graph
    additional_nodes_count = output_nodes - common_nodes

    # Initialize blended feature and adjacency matrices
    blended_features = np.zeros((output_nodes,features_1.shape[1]))
    blended_adjacency = np.zeros((output_nodes, output_nodes))

    # Blend features for common nodes
    for i in range(common_nodes):
        # Randomly choose the class from one of the two nodes
        if random.random() < 0.5:
            blended_features[i, :4] = features_1[i, :4]  # Copy class from node in features_1
        else:
            blended_features[i, :4] = features_2[i, :4]  # Copy class from node in features_2
        
        # Blend the rest of the features
        blended_features[i, 4:] = blend_ratio * features_1[i, 4:] + (1 - blend_ratio) * features_2[i, 4:]
        
        # Blend adjacency for common nodes
        for j in range(common_nodes):
            blended_adjacency[i, j] = blend_ratio * adj_1[i, j] + (1 - blend_ratio) * adj_2[i, j]

    # Select additional nodes from the remaining nodes of the larger graph
    if additional_nodes_count > 0 and len(remaining_nodes_indices) > 0:
        selected_additional_indices = np.random.choice(remaining_nodes_indices, size=additional_nodes_count, replace=False)
        for i, idx in enumerate(selected_additional_indices, start=common_nodes):
            blended_features[i] = larger_graph_features[idx]
            for j in range(output_nodes):
                if j < common_nodes:
                    blended_adjacency[i, j] = blended_adjacency[j, i] = larger_graph_adj[idx, min(j, larger_graph_adj.shape[1]-1)]
                if j >= common_nodes:
                    # For additional nodes, ensure adjacency reflects the larger graph's structure
                    adj_idx = selected_additional_indices[j - common_nodes] if j != i else idx
                    blended_adjacency[i, j] = blended_adjacency[j, i] = larger_graph_adj[idx, adj_idx]

    return blended_features, blended_adjacency
def load_graph_features_and_adjacency(base_dir,graph_dir):
    features = np.load(os.path.join(base_dir,graph_dir, f"{os.path.basename(graph_dir)}_feature_vectors_extracted_1024.npy"))
    adjacency = np.load(os.path.join(base_dir,graph_dir, "adjacency_matrix_inverse.npy"))
    adjacency = np.logical_not(adjacency == 0).astype(int)  # Convert to binary adjacency matrix
    return features, adjacency
    
def average_feature_similarity(features1, features2):
    """
    Computes the average Euclidean distance between feature vectors of two graphs.
    """
    # Ensure both feature sets have the same number of features by padding shorter one
    if features1.shape[0] > features2.shape[0]:
        features2 = np.pad(features2, ((0, features1.shape[0] - features2.shape[0]), (0, 0)), 'constant')
    elif features2.shape[0] > features1.shape[0]:
        features1 = np.pad(features1, ((0, features2.shape[0] - features1.shape[0]), (0, 0)), 'constant')
    
    distances = np.linalg.norm(features1 - features2, axis=1)
    average_distance = np.mean(distances)
    return average_distance
def spectral_distance(G1, G2):
    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()
    eigenvalues_L1 = np.sort(np.linalg.eigvals(L1))
    eigenvalues_L2 = np.sort(np.linalg.eigvals(L2))
    if len(eigenvalues_L1) > len(eigenvalues_L2):
        eigenvalues_L2 = np.pad(eigenvalues_L2, (0, len(eigenvalues_L1) - len(eigenvalues_L2)), 'constant')
    elif len(eigenvalues_L1) < len(eigenvalues_L2):
        eigenvalues_L1 = np.pad(eigenvalues_L1, (0, len(eigenvalues_L2) - len(eigenvalues_L1)), 'constant')
    distance = np.linalg.norm(eigenvalues_L1 - eigenvalues_L2)
    return distance
    
def adjacency_to_nx(adjacency_matrix):
    return nx.from_numpy_array(adjacency_matrix)        
def generate_blended_graphs(base_dir, output_dir, num_graphs):
    print(f"Generating {num_graphs} blended graphs...")
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]
    
    for i in range(num_graphs):
        print(f"Processing graph {i+1}/{num_graphs}...")
        graph1_dir, graph2_dir = random.sample(folders, 2)
        
        features1, adjacency1 = load_graph_features_and_adjacency(base_dir,graph1_dir)
        features2, adjacency2 = load_graph_features_and_adjacency(base_dir,graph2_dir)
        print(np.shape(features1), np.shape(features2),graph1_dir,graph2_dir)
        blend_ratio = random.random()
        output_nodes = random.randint(min(features1.shape[0], features2.shape[0]), max(features1.shape[0], features2.shape[0]))
        
        blended_features, blended_adjacency = blend_graphs(features1, adjacency1, features2, adjacency2, blend_ratio, output_nodes)
        selected_folders = random.sample(folders, 5)  # Select 5 random folders for comparison        
        # Convert the blended adjacency matrix to a NetworkX graph
        G_blended = adjacency_to_nx(blended_adjacency)

        # Compute spectral distances between the blended graph and 5 randomly selected graphs
        spectral_distances = []
        feature_distances = []
        for folder in selected_folders:
            features, adjacency = load_graph_features_and_adjacency(base_dir, folder)
            G = adjacency_to_nx(adjacency)
            spectral_distance_val = spectral_distance(G_blended, G)
            spectral_distances.append(spectral_distance_val)
            
            feature_distance_val = average_feature_similarity(blended_features, features)
            feature_distances.append(feature_distance_val)
        
        average_spectral_distance = np.mean(spectral_distances)
        average_feature_distance = np.mean(feature_distances)
        # Combine the spectral and feature distances into a single score
        # Here, simply averaging the two, but you can choose a different method
        combined_score = (average_spectral_distance + average_feature_distance) / 2
        print(f"Combined score (spectral + feature): {combined_score}, Spectral score : {average_spectral_distance} ")

        #base_size = (1024, 1024)
        np.save(os.path.join(output_dir, f"blended_{i+1}_{combined_score:.2f}_{average_spectral_distance}_features.npy"), blended_features)
        np.save(os.path.join(output_dir, f"blended_{i+1}_{combined_score:.2f}_{average_spectral_distance}_adjacency.npy"), blended_adjacency)
        #image_with_overlay = create_overlay_image_with_decoded_features(base_size, blended_features, blended_adjacency)
        #image_with_overlay.save(os.path.join(output_dir, f"blended_{i+1}_{combined_score:.2f}_{average_spectral_distance}_graph.png"))

# Example usage
base_dir = '/home/atuin/b143dc/b143dc22/GCD/kidney_preprocessed'
output_dir = '/home/atuin/b143dc/b143dc22/GCD/generated_graphs/blended_extracted'  
num_graphs = 9990
generate_blended_graphs(base_dir, output_dir, num_graphs)