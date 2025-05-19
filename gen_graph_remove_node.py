import os
import numpy as np
import random
import networkx as nx

def remove_node_from_graph(adjacency_matrix, features, node_to_remove):
    """
    Removes a node and its connections from the adjacency matrix and feature vector.
    
    Args:
        adjacency_matrix (np.ndarray): Original adjacency matrix.
        features (np.ndarray): Feature vectors associated with the nodes.
        node_to_remove (int): Index of the node to remove.

    Returns:
        new_adjacency (np.ndarray): Adjacency matrix with the node removed.
        new_features (np.ndarray): Feature vectors with the node removed.
    """
    new_adjacency = np.delete(adjacency_matrix, node_to_remove, axis=0)
    new_adjacency = np.delete(new_adjacency, node_to_remove, axis=1)
    new_features = np.delete(features, node_to_remove, axis=0)
    
    return new_adjacency, new_features

def process_graphs(base_dir, output_dir):
    """
    Loads graphs from the base directory, removes two random nodes from each graph,
    and saves two new graphs for each input graph.
    
    Args:
        base_dir (str): Path to the directory containing input graphs.
        output_dir (str): Path to save the modified graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]

    for folder in folders:
        # Load the adjacency matrix and feature vectors
        adj_path = os.path.join(base_dir, folder, "adjacency_matrix_inverse.npy")
        features_path = os.path.join(base_dir, folder, f"{os.path.basename(folder)}_feature_vectors_extracted_1024.npy")
        
        adjacency = np.load(adj_path)
        features = np.load(features_path)
        
        num_nodes = adjacency.shape[0]
        if num_nodes <= 2:
            print(f"Skipping graph {folder}, not enough nodes to remove.")
            continue

        # Create a list of node indices to ensure different nodes are removed
        node_indices = list(range(num_nodes))
        
        for i in range(2):
            # Randomly select a node to remove from the original node set
            node_to_remove = random.choice(node_indices)
            
            new_adjacency, new_features = remove_node_from_graph(adjacency, features, node_to_remove)

            # Save the new graph
            output_subdir = os.path.join(output_dir, f"{folder}_modified_{i+1}")
            os.makedirs(output_subdir, exist_ok=True)
            np.save(os.path.join(output_subdir, "adjacency_matrix_inverse.npy"), new_adjacency)
            np.save(os.path.join(output_subdir, f"{os.path.basename(folder)}_modified_{i+1}_feature_vectors_extracted_1024.npy"), new_features)

            print(f"Graph {folder}, iteration {i+1}: Removed node {node_to_remove}.")
            
            # Remove the node from the list of indices to prevent selecting it again
            node_indices.remove(node_to_remove)

        print(f"Processed graph {folder}, generated 2 new graphs.")

# Paths for the base and output directories
base_dir = path/to/preprocessed
output_dir = path/to/generated_graphs/removed_node

# Process all graphs in the base directory
process_graphs(base_dir, output_dir)
