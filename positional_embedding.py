import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image, ImageDraw

import glob


# Adjusted parameters for RGB images
image_size = 1024  # Example size, adjust as needed
patch_size = 64   # Adjust based on your needs
sequence_length = (image_size // patch_size) ** 2  # Number of patches
embedding_dim = 128  # Example embedding dimension, adjust as needed
base_path = '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/'
# Example image paths (hypothetical, replace with actual paths)
image_paths = [
    '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/results0/results0/segment_1_718_982_class_1.png',
    '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/results1/results1/segment_1_712_979_class_1.png',
    '/vol/biomedic3/sc7718/Graph_conditioned_diffusion/kidney_preprocessed/results1080/results1080/segment_10_425_856_class_1.png'
]

# Sinusoidal Embedding Function
def get_sinusoidal_embeddings(n_positions, embedding_dim):
    position = np.arange(n_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    embeddings = np.zeros((n_positions, embedding_dim))
    embeddings[:, 0::2] = np.sin(position * div_term)
    embeddings[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(embeddings, dtype=torch.float)

# Generate sinusoidal embeddings for the entire image grid
positional_embeddings = get_sinusoidal_embeddings(sequence_length, embedding_dim)


def load_and_process_image(image_path, patch_size=16):
    # Load the image
    image = Image.open(image_path)
    # Convert to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert image to tensor
    tensor_image = transforms.ToTensor()(image)  # Shape: [C, H, W]

    # Initialize variables to track the most significant patch
    max_diff = -1
    selected_patch_idx = None

    C, H, W = tensor_image.shape
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Extract the current patch
            patch = tensor_image[:, i:i+patch_size, j:j+patch_size]

            # Compute the average color of the patch
            avg_color = patch.mean(dim=[1, 2])

            # Compute the difference from black (or another baseline color if necessary)
            # Here, we use the L2 norm which considers the patch's deviation from pure black
            diff = torch.norm(avg_color, p=2).item()

            # Update the selected patch if this one has a higher difference
            if diff > max_diff:
                max_diff = diff
                selected_patch_idx = (i // patch_size) * (W // patch_size) + (j // patch_size)

    # Convert 2D index to 1D index
    return selected_patch_idx
# Function to process an image and draw a red box around the selected patch
def process_and_draw_image(image_path, patch_size=16):
    # Load the image
    image = Image.open(image_path)
    # Convert to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    tensor_image = transforms.ToTensor()(image)  # Convert image to tensor

    # Initialize variables to track the most significant patch
    max_diff = -1
    selected_patch_corner = None

    C, H, W = tensor_image.shape
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Extract the current patch
            patch = tensor_image[:, i:i+patch_size, j:j+patch_size]

            # Compute the average color of the patch
            avg_color = patch.mean(dim=[1, 2])

            # Compute the difference from black (or another baseline color if necessary)
            diff = torch.norm(avg_color, p=2).item()

            # Update the selected patch if this one has a higher difference
            if diff > max_diff:
                max_diff = diff
                selected_patch_corner = (j, i)  # Note the swap to match PIL's (x, y) coordinate system

    # Convert the tensor image back to PIL image for drawing
    pil_image = transforms.ToPILImage()(tensor_image)

    # Draw a red box around the selected patch
    if selected_patch_corner is not None:
        draw = ImageDraw.Draw(pil_image)
        bottom_right_corner = (selected_patch_corner[0] + patch_size, selected_patch_corner[1] + patch_size)
        draw.rectangle([selected_patch_corner, bottom_right_corner], outline="red", width=3)
    else:
        print("No significant patch identified.")

    return pil_image

for image_path in image_paths:
    processed_image = process_and_draw_image(image_path, patch_size=64)
    processed_image.show()  # Display the image with the patch highlighted, or use processed_image.save(path) to save it.

for i in range(0, 1982):
    print(f"Processing folder: results{i}")  # Progress indication for folders
    folder_path = os.path.join(base_path, f'results{i}')
    embeddings = []  # Reset embeddings list for each folder

    if os.path.isdir(folder_path):
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                print(f"  Sub-folder: {sub_folder}")  # Progress indication for sub-folders
                for image_file in os.listdir(sub_folder_path):
                    image_path = os.path.join(sub_folder_path, image_file)
                    if os.path.isfile(image_path) and image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                        print(f"    Processing image: {image_file}")  # Progress indication for images
                        patch_idx = load_and_process_image(image_path, patch_size=patch_size)
                        emb = positional_embeddings[patch_idx].numpy()
                        embeddings.append(emb)

                if embeddings:
                    embeddings_matrix = np.stack(embeddings)
                    np.save(os.path.join(sub_folder_path, 'positional_embeddings.npy'), embeddings_matrix)
                    print(f"    Embeddings saved for {sub_folder}")