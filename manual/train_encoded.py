from uuid import uuid4

import matplotlib
import numpy as np
import torch
import argparse

from imagen_pytorch import Unet, ImagenTrainer, Imagen, NullUnet
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T

from patient_dataset_encoded import PatientDataset
import os
import pandas as pd
from glob import glob
import wandb

import re
import gc


TEXT_EMBED_DIM = 3
SPLIT_VALID_FRACTION = 0.025


def unet_generator(unet_number):
    if unet_number == 1:
        return Unet(
            dim=256,
            dim_mults=(1, 2, 4, 8),
            cond_dim=512,
            num_resnet_blocks=3,
            text_embed_dim=1024,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
        )

    if unet_number == 2:
        return Unet(
            dim=128,
            cond_dim=512,
            text_embed_dim=1024,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=2,
            memory_efficient=True,
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, True, True),
            init_conv_to_final_conv_residual=True,
        )
    
    if unet_number == 3:
        return Unet(
            dim=128,
            cond_dim=512,
            text_embed_dim=1024,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 4, 4),
            memory_efficient=True,
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            init_conv_to_final_conv_residual=True,
        )

    return None


class FixedNullUnet(NullUnet):
    def __init__(self, lowres_cond=False, *args, **kwargs):
        super().__init__()
        self.lowres_cond = lowres_cond
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        return self

    def forward(self, x, *args, **kwargs):
        return x


def init_imagen(unet_number):
    imagen = Imagen(
        condition_on_text = True,
        unets=(
            unet_generator(1) if unet_number == 1 else FixedNullUnet(),
            unet_generator(2) if unet_number == 2 else FixedNullUnet(lowres_cond=True),
            unet_generator(3) if unet_number == 3 else FixedNullUnet(lowres_cond=True),
        ),
        image_sizes=(64, 256, 1024),
        timesteps=(1024, 256, 256),
        text_embed_dim=1024,
        pred_objectives=("noise", "noise", "noise"),
        random_crop_sizes=(None, None, 256),
    ).cuda()

    return imagen

def log_wandb(cur_step, loss, validation=False):
    wandb.log({
        "loss" if not validation else "val_loss" : loss,
        "step": cur_step,
    })

def main():
    args = parse_args()
    
    imagen = init_imagen(args.unet_number)
    trainer = ImagenTrainer(
        imagen=imagen,
        dl_tuple_output_keywords_names=('images', 'text_embeds', 'text_masks'),
        fp16=True,
    )

    # Load the data and the graphs
    data_path = f'{args.data_path}'

    # Initialise PatientDataset
    dataset = PatientDataset(data_path, patch_size=1024, image_size=1024, transformations=False)


    train_size = int((1 - SPLIT_VALID_FRACTION) * len(dataset))
    indices = list(range(len(dataset)))
    train_dataset = Subset(dataset, np.random.permutation(indices[:train_size]))
    valid_dataset = Subset(dataset, np.random.permutation(indices[train_size:]))

    trainer.accelerator.print(f'training with dataset of {len(train_dataset)} samples and validating with {len(valid_dataset)} samples')


    trainer.add_train_dataset(train_dataset, batch_size=8, num_workers=args.num_workers)
    trainer.add_valid_dataset(valid_dataset, batch_size=8, num_workers=args.num_workers)

    if args.unet_number == 1:
        checkpoint_path = args.unet1_checkpoint
    elif args.unet_number == 2:
        checkpoint_path = args.unet2_checkpoint
    else:
        checkpoint_path = args.unet3_checkpoint

    trainer.load(checkpoint_path, noop_if_not_exist=True)

    run_id = None

    if trainer.is_main:
        run_id = wandb.util.generate_id()
        if args.run_id is not None:
            run_id = args.run_id
        trainer.accelerator.print(f"Run ID: {run_id}")

        try:
            os.makedirs(f"samples/{run_id}")
        except FileExistsError:
            pass

        wandb.init(project=f"training_unet{args.unet_number}", resume=args.resume, id=run_id)

    trainer.accelerator.wait_for_everyone()
    while True:
        print('hey')
        step_num = trainer.num_steps_taken(args.unet_number)
        loss = trainer.train_step(unet_number=args.unet_number)
        trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} loss: {loss}')

        if trainer.is_main:
            log_wandb(step_num, loss)

        if not (step_num % 50):
            valid_loss = trainer.valid_step(unet_number=args.unet_number)
            trainer.accelerator.print(f'step {step_num}: unet{args.unet_number} validation loss: {valid_loss}')
            if trainer.is_main:
                log_wandb(step_num, loss, validation=True)

        if not (step_num % args.save_freq) and step_num > 0:
            trainer.accelerator.wait_for_everyone()
            unique_path = f"{re.sub(r'.pt$', '', checkpoint_path)}_{step_num}.pt"
            trainer.accelerator.print("Saving model...")
            trainer.save(unique_path)
            trainer.accelerator.print("Saved model under unique name:")

        if not (step_num % args.sample_freq):
            trainer.accelerator.wait_for_everyone()
            trainer.accelerator.print()
            trainer.accelerator.print("Saving model and sampling")

            if trainer.is_main:
                lowres_image, conds, mask = dataset[0]
                rand_image, rand_conds, rand_mask = dataset[np.random.randint(len(dataset))]

                with torch.no_grad():
                    images = trainer.sample(
                        batch_size=2,
                        return_pil_images=False,
                        start_image_or_video=torch.stack([lowres_image, rand_image]),
                        text_masks=torch.stack([mask, rand_mask]),
                        text_embeds=torch.stack([conds, rand_conds]),
                        start_at_unet_number=args.unet_number,
                        stop_at_unet_number=args.unet_number,
                    )

                for index in range(len(images)):
                    T.ToPILImage()(images[index]).save(f'samples/{run_id}/sample-{step_num}-{run_id}-{index}.png')
                    wandb.log({f"sample{'' if index == 0 else f'-{index}'}": wandb.Image(images[index])})
    
            trainer.accelerator.wait_for_everyone()
            trainer.save(checkpoint_path)
            trainer.accelerator.print("Finished sampling and saving model!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet1_checkpoint', type=str, default='checkpoints_encoded/unet1_checkpoint.pt', help='Path to checkpoint for unet1 model')
    parser.add_argument('--unet2_checkpoint', type=str, default='checkpoints_encoded/unet2_checkpoint.pt', help='Path to checkpoint for unet2 model')
    parser.add_argument('--unet3_checkpoint', type=str, default='checkpoints_encoded/unet3_checkpoint.pt', help='Path to checkpoint for unet3 model')  
    parser.add_argument('--unet_number', type=int, choices=range(1, 4), help='Unet to train')
    parser.add_argument('--data_path', type=str, default='', help='Path of training patches')
    parser.add_argument('--sample_freq', type=int, default=500, help='How many epochs between sampling and checkpoint.pt saves')
    parser.add_argument('--save_freq', type=int, default=50000, help='How many steps between saving a checkpoint under a unique name')
    parser.add_argument('--annotated_dataset', action='store_true', help='Train with an annotated dataset')
    parser.add_argument('--resume', action='store_true', help='Resume previous run using wandb')
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--unconditional', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
