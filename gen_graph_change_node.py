import os
import numpy as np
import random

def modify_one_hot_encoding(features, num_classes=4):
    """
    Modifies the one-hot class encoding of a single random node.

    Args:
        features (np.ndarray): Feature vectors of shape (num_nodes, feature_dim).
        num_classes (int): Number of possible classes in the one-hot encoding.

    Returns:
        new_features (np.ndarray): Modified feature vectors.
    """
    new_features = features.copy()
    
    # Assuming one-hot encoding is in the first `num_classes` columns
    num_nodes = new_features.shape[0]
    random_node = random.randint(0, num_nodes - 1)
    
    # Get current class ID of the random node
    current_class = np.argmax(new_features[random_node, :num_classes])
    
    # Generate a new class ID different from the current one
    new_class = random.choice([c for c in range(num_classes) if c != current_class])
    
    # Update the one-hot encoding for the selected node
    new_features[random_node, :num_classes] = 0
    new_features[random_node, new_class] = 1
    
    print(f"Modified node {random_node}: changed class from {current_class} to {new_class}")
    
    return new_features, random_node, current_class, new_class

def process_graphs(base_dir, output_dir, num_classes=4):
    """
    Loads graphs from the base directory, modifies one random node's class encoding,
    and saves the modified graphs.

    Args:
        base_dir (str): Path to the directory containing input graphs.
        output_dir (str): Path to save the modified graphs.
        num_classes (int): Number of possible classes in the one-hot encoding.
    """
    os.makedirs(output_dir, exist_ok=True)
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('results')]

    for folder in folders:
        # Load the feature vectors
        features_path = os.path.join(base_dir, folder, f"{os.path.basename(folder)}_feature_vectors_extracted_1024.npy")
        features = np.load(features_path)
        
        # Modify the one-hot class encoding of a random node
        new_features, node, old_class, new_class = modify_one_hot_encoding(features, num_classes)
        
        # Save the modified features
        output_subdir = os.path.join(output_dir, f"{folder}_modified_class")
        os.makedirs(output_subdir, exist_ok=True)
        np.save(os.path.join(output_subdir, f"{os.path.basename(folder)}_modified_feature_vectors_extracted_1024.npy"), new_features)
        
        print(f"Processed {folder}: node {node} class {old_class} -> {new_class}")
        
# Paths for the base and output directories
base_dir = path/to/preprocessed
output_dir = path/to/generated_graphs/change_one_node

# Process all graphs in the base directory
process_graphs(base_dir, output_dir)
