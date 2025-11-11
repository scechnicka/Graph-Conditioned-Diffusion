from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from train_graph import unet_generator, init_imagen

#from patient_dataset import PatientDataset
import os
import gc
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import re

def generate_images(unet_number, args, deep_labelmap, num_variants, lowres_images=None):
    imagen = init_imagen(unet_number)
    trainer = ImagenTrainer(imagen=imagen)

    if unet_number == 1:
        path = args.unet1_checkpoint
    elif unet_number == 2:
        path = args.unet2_checkpoint
    else:
        path = args.unet3_checkpoint

    trainer.load(path)

    deep_labelmap = torch.tensor(deep_labelmap).unsqueeze(0).repeat_interleave(num_variants, dim=0).float().cuda()

    images = trainer.sample(
        batch_size=num_variants,
        return_pil_images=(unet_number==3),
        start_image_or_video=lowres_images,
        cond_images=deep_labelmap,
        start_at_unet_number=unet_number,
        stop_at_unet_number=unet_number,
    )

    del trainer
    del imagen
    gc.collect()
    torch.cuda.empty_cache()

    return images
    
def find_graph_and_image_npy_files(directory):
    graph_and_image_files = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        graph_image_pair = []
        graph_path = None
        image_path = None
        # Iterate through the files in the current directory
        for file in files:
            # Check if the file is called 'graph.npy'
            if file == 'graph.npy':
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


def main():

    args = parse_args()
    try:
        os.makedirs(f"samples")
    except FileExistsError:
        pass
        
    #Load the labelmasks:
    file_list = find_graph_and_image_npy_files(args.data_path)
    for file in tqdm(file_list[1:10]):
        out_path = args.out_path + file[0].split('/')[-2]
        graph = np.load(file[1])
        # Convert the patch to a tensor
        graph_tensor = torch.from_numpy(graph / 255).permute((2, 0, 1)).float()
        
        #Save the graph and the real image: 
        # Create the subdirectory if it doesn't exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        image = Image.fromarray((graph).astype(np.uint8))
        # Save the PIL Image as PNG
        image.save(out_path+'/graph.png')
        image = Image.fromarray((np.load(file[0]).astype(np.uint8)))
        # Save the PIL Image as PNG
        image.save(out_path+'/real_image.png')
        
        lowres_images = generate_images(1, args, graph_tensor, 2)
        medres_images = generate_images(2, args, graph_tensor, 2, lowres_images=lowres_images)
        highres_images = generate_images(3, args, graph_tensor, 2, lowres_images=medres_images)        
        
        for j, image in enumerate(highres_images):
            image.save(out_path+'/sample_'+str(j)+'.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='checkpoints_graph/unet1_checkpoint_250000.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='checkpoints_graph/unet2_checkpoint_250000.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='checkpoints_graph/unet3_checkpoint_150000.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--data_path', type=str, default='', help='Path of training patches')
    parser.add_argument('--out_path', type=str, default='generated_samples/')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for segmentation mask')
    parser.add_argument('--end_index', type=int, default=0, help='End index for segmentation mask')
    parser.add_argument('--only_glom_tubules', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
