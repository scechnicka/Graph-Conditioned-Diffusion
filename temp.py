import numpy as np
import random
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
    
def concatenate_ordered_subgraphs(subgraph_matrices1, subgraph_feature_vectors1, subgraph_matrices2, subgraph_feature_vectors2):
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


# Example usage with the previous adjacency_matrix_example and a placeholder feature_vectors_example
feature_vectors_example1 = np.array([
    [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]
])
feature_vectors_example2 = np.array([
    [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35]
])
# Example usage with a placeholder adjacency matrix
adjacency_matrix_example1 = np.array([
    [1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1]
])
# Example usage with a placeholder adjacency matrix
adjacency_matrix_example2 = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 1]
])

adjacency_matrix_example3 = np.array([
    [0.1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 1, 1, 2, 0, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1]
])

# Find and remove bridges
adjacency_matrix = np.logical_not(adjacency_matrix_example3 == 0).astype(int)
print(adjacency_matrix)
modified_adjacency_matrix1 = find_bridges_remove_avoid_lone_nodes(adjacency_matrix_example1.copy())
print(modified_adjacency_matrix1)
modified_adjacency_matrix2 = find_bridges_remove_avoid_lone_nodes(adjacency_matrix_example2.copy())
print(modified_adjacency_matrix2)

subgraph_matrices1, subgraph_feature_vectors1 = find_and_separate_subgraphs_with_features(adjacency_matrix_example1, feature_vectors_example1)
subgraph_matrices2, subgraph_feature_vectors2 = find_and_separate_subgraphs_with_features(adjacency_matrix_example2, feature_vectors_example2)
for i, (subgraph_matrix1, subgraph_features1) in enumerate(zip(subgraph_matrices1, subgraph_feature_vectors1)):
    print(f"Subgraph {i} Matrix:\n{subgraph_matrix1}\nFeatures:\n{subgraph_features1}\n")
for i, (subgraph_matrix2, subgraph_features2) in enumerate(zip(subgraph_matrices2, subgraph_feature_vectors2)):
    print(f"Subgraph {i} Matrix:\n{subgraph_matrix2}\nFeatures:\n{subgraph_features2}\n")



# Concatenate a random number of subgraphs from both sets into a final graph
final_adj_matrix, final_features = concatenate_ordered_subgraphs(subgraph_matrices1, subgraph_feature_vectors1, subgraph_matrices2, subgraph_feature_vectors2)


print(final_adj_matrix, final_features)



