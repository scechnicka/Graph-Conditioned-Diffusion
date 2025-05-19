import numpy as np
import os
import random
from PIL import Image, ImageDraw
import numpy as np
import random
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
      
def find_bridges_remove_avoid_lone_nodes(adj_matrix):
    n = adj_matrix.shape[0]  # Number of vertices
    modified_matrix = np.copy(adj_matrix)
    visited = [False] * n
    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    time = [0]  # Mutable object to keep track of discovery time
    
    def bridge_dfs(u):
        visited[u] = True
        disc[u] = time[0]
        low[u] = time[0]
        time[0] += 1
        
        for v in range(n):
            # Added check to ensure u and v are within bounds
            if u < n and v < n:
                if modified_matrix[u, v] == 1:  # If u-v is connected
                    if not visited[v]:  # If v is not visited
                        parent[v] = u
                        bridge_dfs(v)
                        
                        # Check if the subtree rooted at v has a connection to one of the ancestors of u
                        low[u] = min(low[u], low[v])
                        
                        if low[v] > disc[u]:
                            # Temporarily remove the bridge to check for lone nodes
                            modified_matrix[u, v] = 0
                            modified_matrix[v, u] = 0
                            
                            # Additional checks for matrix dimensions before accessing
                            if u < n and v < n:
                                if (np.sum(modified_matrix[u, :]) == 1 and modified_matrix[u, u] == 1) or \
                                   (np.sum(modified_matrix[:, v]) == 1 and modified_matrix[v, v] == 1) or \
                                   (np.sum(modified_matrix[v, :]) == 1 and modified_matrix[v, v] == 1) or \
                                   (np.sum(modified_matrix[:, u]) == 1 and modified_matrix[u, u] == 1):
                                    # Restore the bridge if removal would result in a lone node
                                    modified_matrix[u, v] = 1
                                    modified_matrix[v, u] = 1
                    elif v != parent[u]:
                        low[u] = min(low[u], disc[v])
    
    # Call the recursive helper function to find bridges
    # in DFS tree rooted with vertex 'i'
    for i in range(n):
        if not visited[i]:
            bridge_dfs(i)
    
    return modified_matrix


def find_connected_components(num_vertices, adjacency_matrix):
    print("Finding connected components...")
    """
    Find connected components in the graph using DFS.
    Returns a list of lists, where each sublist represents a connected component
    and contains the indices of nodes in that component.
    """
    visited = [False] * num_vertices
    components = []

    def dfs(v, current_component):
        visited[v] = True
        current_component.append(v)
        for i, is_connected in enumerate(adjacency_matrix[v]):
            if is_connected and not visited[i]:
                dfs(i, current_component)

    for i in range(num_vertices):
        if not visited[i]:
            current_component = []
            dfs(i, current_component)
            components.append(current_component)
    
    return components

def build_subgraph_matrices(components, adjacency_matrix):
    print(f"Building subgraph matrices for {len(components)} components...")
    """
    Build adjacency matrices for each component (subgraph).
    """
    subgraph_matrices = []

    for component in components:
        # Create a subgraph adjacency matrix for the current component
        subgraph_size = len(component)
        subgraph_matrix = np.zeros((subgraph_size, subgraph_size), dtype=int)

        # Map original indices to new, local indices in the subgraph
        index_map = {original: new for new, original in enumerate(component)}

        for i in range(subgraph_size):
            for j in range(subgraph_size):
                if adjacency_matrix[component[i]][component[j]]:
                    subgraph_matrix[index_map[component[i]], index_map[component[j]]] = 1

        subgraph_matrices.append(subgraph_matrix)

    return subgraph_matrices

def find_and_separate_subgraphs_with_features(adjacency_matrix, feature_vectors):
    print("Separating subgraphs with features...")
    num_vertices = len(adjacency_matrix)
    # Find and remove bridges, then identify connected components
    modified_adjacency_matrix = find_bridges_remove_avoid_lone_nodes(adjacency_matrix.copy())
    components = find_connected_components(num_vertices, modified_adjacency_matrix)

    subgraph_matrices = []
    subgraph_feature_vectors = []

    for component in components:
        # Create a subgraph adjacency matrix for the current component
        subgraph_size = len(component)
        subgraph_matrix = np.zeros((subgraph_size, subgraph_size), dtype=int)
        # Allocate space for the subgraph's feature vectors
        subgraph_features = np.zeros((subgraph_size, feature_vectors.shape[1]))

        # Map original indices to new, local indices in the subgraph and separate features
        index_map = {original: new for new, original in enumerate(component)}

        for i in range(subgraph_size):
            for j in range(subgraph_size):
                if adjacency_matrix[component[i]][component[j]]:
                    subgraph_matrix[index_map[component[i]], index_map[component[j]]] = 1
            subgraph_features[index_map[component[i]]] = feature_vectors[component[i]]

        subgraph_matrices.append(subgraph_matrix)
        subgraph_feature_vectors.append(subgraph_features)

    return subgraph_matrices, subgraph_feature_vectors

def decode_feature_vector(feature_vector):
    # Ensure we only consider the first 11 elements of the feature vector
    relevant_vector = feature_vector[:11]
    
    class_id = np.argmax(relevant_vector[:4]) + 1
    centroid_x, centroid_y, avg_area, bbox_x, bbox_y, bbox_width, bbox_height = relevant_vector[4:]
    centroid = (centroid_x * 1024, centroid_y * 1024)
    bounding_box = (bbox_x * 1024, bbox_y * 1024, bbox_width * 1024, bbox_height * 1024)
    
    return class_id, centroid, avg_area, bounding_box
    
def concatenate_ordered_subgraphs(subgraph_matrices1, subgraph_feature_vectors1, subgraph_matrices2, subgraph_feature_vectors2):
    print("Concatenating ordered subgraphs...")
    # Randomly select indices for subgraphs from both sets
    selected_indices1 = random.sample(range(len(subgraph_matrices1)), random.randint(1, len(subgraph_matrices1)))
    selected_indices2 = random.sample(range(len(subgraph_matrices2)), random.randint(1, len(subgraph_matrices2)))
    
    # Sort the selected indices
    selected_indices1.sort()
    selected_indices2.sort()
    
    # Combine and tag the indices, then sort by index
    combined_indices = [(idx, 1) for idx in selected_indices1] + [(idx, 2) for idx in selected_indices2]
    combined_indices.sort(key=lambda x: x[0])
    
    # Initialize the first subgraph and features to start concatenation
    final_adj_matrix, final_features = None, None
    
    for idx, origin in combined_indices:
        if origin == 1:
            matrix, features = subgraph_matrices1[idx], subgraph_feature_vectors1[idx]
        else:  # origin == 2
            matrix, features = subgraph_matrices2[idx], subgraph_feature_vectors2[idx]
        
        if final_adj_matrix is None:
            final_adj_matrix = matrix
            final_features = features
        else:
            # Create zero matrices for the off-diagonal blocks
            zeros_top_right = np.zeros((final_adj_matrix.shape[0], matrix.shape[1]))
            zeros_bottom_left = np.zeros((matrix.shape[0], final_adj_matrix.shape[1]))
            
            # Add connectivity between the last node of the current graph and the first node of the next subgraph
            zeros_top_right[-1, 0] = zeros_bottom_left[0, -1] = 1
            
            # Concatenate matrices and feature vectors
            final_adj_matrix = np.block([
                [final_adj_matrix, zeros_top_right],
                [zeros_bottom_left, matrix]
            ])
            final_features = np.vstack((final_features, features))
    
    return final_adj_matrix, final_features
    
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
       
def create_overlay_image_with_decoded_features(base_size, feature_vectors, adjacency_matrix):
    print("Creating overlay image with decoded features...")
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
    
def partial_graphs(adjacency1, features1, adjacency2, features2):
    print("Generating partial graphs...")
    
    subgraph_matrices1, subgraph_feature_vectors1 = find_and_separate_subgraphs_with_features(adjacency1, features1)
    subgraph_matrices2, subgraph_feature_vectors2 = find_and_separate_subgraphs_with_features(adjacency2, features2)

        # Concatenate a random number of subgraphs from both sets into a final graph
    partial_adjacency_matrix, partial_features = concatenate_ordered_subgraphs(subgraph_matrices1, subgraph_feature_vectors1, subgraph_matrices2, subgraph_feature_vectors2)
    return partial_features, partial_adjacency_matrix

def load_graph_features_and_adjacency(base_dir,graph_dir):
    features = np.load(os.path.join(base_dir,graph_dir, f"{os.path.basename(graph_dir)}_feature_vectors_manual_1024.npy"))
    adjacency = np.load(os.path.join(base_dir,graph_dir, "adjacency_matrix_inverse.npy"))
    adjacency = np.logical_not(adjacency == 0).astype(int)  # Convert to binary adjacency matrix
    return features, adjacency
    
def generate_partial_graphs(base_dir, output_dir, num_graphs):
    print(f"Generating {num_graphs} partial graphs...")
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]
    
    for i in range(num_graphs):
        print(f"Processing graph {i+1}/{num_graphs}...")
        graph1_dir, graph2_dir = random.sample(folders, 2)
        
        features1, adjacency1 = load_graph_features_and_adjacency(base_dir,graph1_dir)
        features2, adjacency2 = load_graph_features_and_adjacency(base_dir,graph2_dir)
        print(np.shape(features1),np.shape(adjacency1), np.shape(features2), np.shape(adjacency2), graph1_dir,graph2_dir)
        partial_features, partial_adjacency_matrix = partial_graphs(adjacency1, features1, adjacency2, features2)
        base_size = (1024, 1024)
        selected_folders = random.sample(folders, 5)  # Select 5 random folders for comparison        
        # Convert the partial adjacency matrix to a NetworkX graph
        G_partial = adjacency_to_nx(partial_adjacency_matrix)

        # Compute spectral distances between the partial graph and 5 randomly selected graphs
        spectral_distances = []
        feature_distances = []
        for folder in selected_folders:
            features, adjacency = load_graph_features_and_adjacency(base_dir, folder)
            G = adjacency_to_nx(adjacency)
            spectral_distance_val = spectral_distance(G_partial, G)
            spectral_distances.append(spectral_distance_val)
            
            feature_distance_val = average_feature_similarity(partial_features, features)
            feature_distances.append(feature_distance_val)
        
        average_spectral_distance = np.mean(spectral_distances)
        average_feature_distance = np.mean(feature_distances)
        # Combine the spectral and feature distances into a single score
        # Here, simply averaging the two, but you can choose a different method
        combined_score = (average_spectral_distance + average_feature_distance) / 2
        print(f"Combined score (spectral + feature): {combined_score}, Spectral score : {average_spectral_distance} ")

        np.save(os.path.join(output_dir, f"partial_{i+1}_{combined_score:.2f}_{average_spectral_distance}_features.npy"), partial_features)
        np.save(os.path.join(output_dir, f"partial_{i+1}_{combined_score:.2f}_{average_spectral_distance}_adjacency.npy"), partial_adjacency_matrix)
        #image_with_overlay = create_overlay_image_with_decoded_features(base_size, partial_features, partial_adjacency_matrix)
        #image_with_overlay.save(os.path.join(output_dir, f"partial_{i+1}_{combined_score:.2f}_{average_spectral_distance}_graph.png"))
        print(f"Graph {i+1} processing completed.")

    

# Example usage
base_dir = path/to/preprocessed
output_dir = path/to/generated_graphs/partial_manual

num_graphs = 9990
generate_partial_graphs(base_dir, output_dir, num_graphs)

