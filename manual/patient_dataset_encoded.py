from collections import Counter

import h5py
import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import slideio
from tqdm import tqdm
from skimage import color
import numpy as np
from PIL import Image

NUM_FLIPS_ROTATIONS = 8
NUM_TRANSLATIONS =  4


class PatientDataset(Dataset):
    def __init__(self, data_path, patch_size=256, image_size=64, transformations=True, unconditional=False, more_patches=False):
        super().__init__()

        self.transformations = transformations
        self.unconditional = unconditional
        self.patch_size = patch_size
        self.image_size = image_size
        self.data_path = data_path

        # Iterate directory and add images and graphs to list
        
        self.image_graph_files = self.find_graph_and_image_npy_files(data_path)
        self.num_patches=len(self.image_graph_files)
        print(np.shape(self.image_graph_files))
        print(self.num_patches)


    def __len__(self):
        if self.transformations:
            return NUM_FLIPS_ROTATIONS * self.num_patches
        else:
            return self.num_patches


    def __getitem__(self, index):

        if self.transformations:
            patch_index = index // NUM_FLIPS_ROTATIONS
        else:
            patch_index = index
        
        patch = np.load(self.image_graph_files[patch_index][0])
        
        graph = np.load(self.image_graph_files[patch_index][1])

        # Convert the patch to a tensor
        patch = torch.from_numpy(patch / 255).permute((2, 0, 1)).float()
        graph = torch.from_numpy(graph).squeeze().float()
        mask = graph.norm(dim=-1)!=0
        #print(labelmap.size(), patch.size())

        # Rotate and flip the patch
        if index % NUM_FLIPS_ROTATIONS == 0 or not self.transformations:
            return patch, graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 1:
            return patch.flip(2), graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 2:
            return patch.flip(1), graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 3:
            return patch.flip(1).flip(2), graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 4:
            return patch.transpose(1, 2), graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 5:
            return patch.transpose(1, 2).flip(2), graph, mask
        elif index % NUM_FLIPS_ROTATIONS == 6:
            return patch.transpose(1, 2).flip(1), graph, mask
        else:
            return patch.transpose(1, 2).flip(1).flip(2), graph, mask

    def find_graph_and_image_npy_files(self, directory):
        graph_and_image_files = []
        # Walk through the directory and its subdirectories
        for root, dirs, files in os.walk(directory):
            graph_image_pair = []
            graph_path = None
            image_path = None
            # Iterate through the files in the current directory
            for file in files:
                # Check if the file is called 'conditioning_manual_1024.npy'
                if file == 'conditioning_manual_1024.npy':
                    graph_path = os.path.join(root, file)
                # Check if the file is called 'image.npy'
                elif file == 'image.npy':
                    image_path = os.path.join(root, file)
            # If both graph.npy and image.npy are found, add them to the list
            if graph_path and image_path:
                # Make sure image path is always the first entry in the pair
                graph_image_pair.append(image_path)
                graph_image_pair.append(graph_path)
                graph_and_image_files.append(graph_image_pair)
        return graph_and_image_files




