import numpy as np
import os

# Base directory paths
base_path_class_encoding = '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/'
base_path_positional_embeddings = '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/'
base_path_feature_vecs = '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/BYOL/feature_vecs/'
base_output_path = '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed'

# Loop through each set of files
for i in range(1982):  # Assuming X ranges from 0 to 1981
    # Construct the file paths
    class_encoding_path = os.path.join(base_path_class_encoding, f'results{i}', 'class_encoding.npy')
    positional_embeddings_path = os.path.join(base_path_positional_embeddings, f'results{i}', f'results{i}', 'positional_embeddings.npy')
    feature_vecs_path = os.path.join(base_path_feature_vecs, f'result_{i}.npy')
    output_file_path = os.path.join(base_output_path, f'results{i}', f'results{i}_feature_vectors_extracted_1024.npy')

    # Load the files
    class_encoding = np.load(class_encoding_path)
    positional_embeddings = np.load(positional_embeddings_path)
    feature_vecs = np.load(feature_vecs_path)

    # Concatenate the vectors: class, position, feature
    concatenated_vectors = np.concatenate([class_encoding, positional_embeddings, feature_vecs], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the concatenated array
    np.save(output_file_path, concatenated_vectors)

    print(f"File saved for iteration {i}: {output_file_path}")
