import torch
import numpy as np
import os

def calculate_mean_std(folder_path):
    # List to store all image data
    images_data = []
    
    # Iterate over each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            # Load the image data from .npy file
            img_data = np.load(os.path.join(folder_path, file))
            # Ensure the image is in the format (H, W, C)
            if img_data.shape[-1] == 3:
                images_data.append(torch.from_numpy(img_data))
    
    # Stack all image tensors
    all_images = torch.stack(images_data, dim=0)
    # Rearrange the tensor to (C, N, H, W) for PyTorch
    all_images = all_images.permute(3, 0, 1, 2).float()
    
    # Calculate mean and std
    mean = all_images.mean(dim=[1, 2, 3]) / 255.0
    std = all_images.std(dim=[1, 2, 3]) / 255.0
    
    return mean, std

# Example usage
folder_path = folder_with_results
mean, std = calculate_mean_std(folder_path)
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
