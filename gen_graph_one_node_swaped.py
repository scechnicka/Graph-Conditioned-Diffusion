import os
import numpy as np
import random

def integrate_node_from_graph(features1, adjacency1, features2, adjacency2):
    """
    Integrates a single random node from graph 2 into graph 1, modifying graph 1.

    Args:
        features1 (np.ndarray): Feature matrix of the first graph (num_nodes1, feature_dim).
        adjacency1 (np.ndarray): Adjacency matrix of the first graph (num_nodes1, num_nodes1).
        features2 (np.ndarray): Feature matrix of the second graph (num_nodes2, feature_dim).
        adjacency2 (np.ndarray): Adjacency matrix of the second graph (num_nodes2, num_nodes2).

    Returns:
        new_features1 (np.ndarray): Modified feature matrix of graph 1.
        new_adjacency1 (np.ndarray): Modified adjacency matrix of graph 1.
        swapped_node2 (int): Index of the swapped node from graph 2.
    """
    num_nodes1 = features1.shape[0]
    num_nodes2 = features2.shape[0]
    
    # Select a random node from graph 2
    node2 = random.randint(0, num_nodes2 - 1)
    
    # Add node2's features to graph 1's feature matrix
    new_features1 = np.vstack([features1, features2[node2]])
    
    # Expand adjacency1 to accommodate the new node
    new_adjacency1 = np.zeros((num_nodes1 + 1, num_nodes1 + 1), dtype=int)
    new_adjacency1[:num_nodes1, :num_nodes1] = adjacency1
    
    # Integrate connections for the new node.
    # Only include valid (1) connections up to the size of graph 1:
    node2_connections = adjacency2[node2, :num_nodes1]  # Only take up to the node count of graph1
    new_adjacency1[num_nodes1, :num_nodes1] = node2_connections
    new_adjacency1[:num_nodes1, num_nodes1] = node2_connections  # Ensure symmetry for undirected graph
    
    return new_features1, new_adjacency1, node2

def process_graph_and_save(base_dir, output_dir):
    """
    Integrates a random node from one graph into another and saves the modified graph.

    Args:
        base_dir (str): Path to the directory containing input graphs.
        output_dir (str): Path to save the modified graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]

    # Select two random graphs
    graph1_dir, graph2_dir = random.sample(folders, 2)
    
    features1_path = os.path.join(base_dir, graph1_dir, f"{os.path.basename(graph1_dir)}_feature_vectors_extracted_1024.npy")
    adjacency1_path = os.path.join(base_dir, graph1_dir, "adjacency_matrix_inverse.npy")
    
    features2_path = os.path.join(base_dir, graph2_dir, f"{os.path.basename(graph2_dir)}_feature_vectors_extracted_1024.npy")
    adjacency2_path = os.path.join(base_dir, graph2_dir, "adjacency_matrix_inverse.npy")
    
    features1 = np.load(features1_path)
    adjacency1 = np.load(adjacency1_path)
    adjacency1 = np.logical_not(adjacency1 == 0).astype(int)
    
    features2 = np.load(features2_path)
    adjacency2 = np.load(adjacency2_path)
    adjacency2 = np.logical_not(adjacency2 == 0).astype(int)
    
    # Integrate node from graph 2 into graph 1
    new_features1, new_adjacency1, swapped_node2 = integrate_node_from_graph(features1, adjacency1, features2, adjacency2)
    
    # Save the modified graph
    output_subdir = os.path.join(output_dir, f"{graph1_dir}_with_node_from_{graph2_dir}")
    os.makedirs(output_subdir, exist_ok=True)
    
    np.save(os.path.join(output_subdir, f"{os.path.basename(graph1_dir)}_modified_feature_vectors.npy"), new_features1)
    np.save(os.path.join(output_subdir, "modified_adjacency_matrix.npy"), new_adjacency1)
    
    print(f"Saved modified graph for {graph1_dir}, with node {swapped_node2} from {graph2_dir}.")

# Paths for the base and output directories
base_dir = path/to/preprocessed
output_dir = path/to/generated_graphs/exchange_one_node

# Process one random graph pair
process_graph_and_save(base_dir, output_dir)
