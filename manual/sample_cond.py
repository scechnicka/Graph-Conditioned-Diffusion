from uuid import uuid4

import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from torch import nn

from train_encoded import unet_generator, init_imagen

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

def generate_images(unet_number, args, texts, masks, num_variants, lowres_images=None):
    imagen = init_imagen(unet_number)
    trainer = ImagenTrainer(imagen=imagen)

    if unet_number == 1:
        path = args.unet1_checkpoint
    elif unet_number == 2:
        path = args.unet2_checkpoint
    else:
        path = args.unet3_checkpoint

    trainer.load(path)

    texts = torch.tensor(texts).unsqueeze(0).repeat_interleave(num_variants, dim=0).float().cuda()
    masks = torch.tensor(masks).unsqueeze(0).repeat_interleave(num_variants, dim=0).cuda()

    images = trainer.sample(
        batch_size=num_variants,
        return_pil_images=(unet_number==3),
        start_image_or_video=lowres_images,
        text_masks=masks,
        text_embeds=texts,
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
        text_path = None
        # Iterate through the files in the current directory
        for file in files:
            # Check if the file is called 'graph.npy'
            if file == 'graph.npy':
                graph_path = os.path.join(root, file)
            # Check if the file is called 'image.npy'
            elif file == 'image.npy':
                image_path = os.path.join(root, file)
            elif file == 'conditioning_manual_1024.npy':
                text_path = os.path.join(root, file)
        # If both graph.npy and image.npy are found, add them to the list
        if graph_path and image_path and text_path:
            # Make sure image path is always the first entry in the pair
            graph_image_pair.append(image_path)
            graph_image_pair.append(graph_path)
            graph_image_pair.append(text_path)
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
    for file in tqdm(file_list[4:14]):
        out_path = args.out_path + file[0].split('/')[-2]
        graph = np.load(file[1])
        #Save the graph and the real image: 
        # Create the subdirectory if it doesn't exist
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        image = Image.fromarray((graph).astype(np.uint8))
        image.save(out_path+'/graph.png')
        image = Image.fromarray((np.load(file[0]).astype(np.uint8)))
        image.save(out_path+'/real_image.png')
        
        #Now load the text and calculate the text mask
        text = np.load(file[2])
        text = torch.from_numpy(text).squeeze()
        mask = text.norm(dim=-1)!=0
        
        lowres_images = generate_images(1, args, text, mask, args.num_variants)
        medres_images = generate_images(2, args, text, mask, args.num_variants, lowres_images=lowres_images)
        highres_images = generate_images(3, args, text, mask, args.num_variants, lowres_images=medres_images)        
        
        for j, image in enumerate(highres_images):
            image.save(out_path+'/sample_'+str(j)+'.png')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='checkpoints_encoded/unet1_checkpoint_100000.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='checkpoints_encoded/unet2_checkpoint_100000.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='checkpoints_encoded/unet3_checkpoint_50000.pt', help='Path to checkpoint for unet3 model')
    parser.add_argument('--data_path', type=str, default='', help='Path of training patches')
    parser.add_argument('--out_path', type=str, default='generated_samples/')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for segmentation mask')
    parser.add_argument('--end_index', type=int, default=0, help='End index for segmentation mask')
    parser.add_argument('--num_variants', type=int, default=2, help='Number of samples per graph')
    parser.add_argument('--only_glom_tubules', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
