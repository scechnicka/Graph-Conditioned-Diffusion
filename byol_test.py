import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from byol_pytorch import BYOL, BYOLTrainer
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 20  # Adjust according to your classification task
img_path = path/to/kidney_preprocessed_test/
learning_rate = 3e-4
save_every_epoch = 1000
batch_size = 16

import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
        
class RCrop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return T.RandomResizedCrop((x.size()[2], x.size()[3]))(x)

class Net(nn.Module):
    def __init__(self, num_classes):  # num_classes is directly relevant for classification tasks
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Use global average pooling to handle arbitrary input sizes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Additional fully connected layer to map to a feature vector of size 892
        self.feature_fc = nn.Linear(64, 892)
        # The final classification layer that outputs the desired number of classes
        self.fc = nn.Linear(892, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Apply global average pooling to the output of the last convolution layer
        x = self.global_avg_pool(x)
        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)
        # Pass through the feature mapping layer
        features = F.relu(self.feature_fc(x))
        # Final classification
        x = self.fc(features)
        return x, features
        
# Example usage:
# Instantiate the model with the number of classes

class segment_dataset(Dataset):
    def __init__(self, folder_path):
        self.png_paths = self.find_segment_png_files(img_path)
        
    def __getitem__(self, index):
        image = Image.open(self.png_paths[index])
        image = self.crop_to_content(image)
        image = torch.from_numpy(image/255).permute((2, 0, 1)).float()
        return image.to('cuda')
        
    def __len__(self):
        return len(self.png_paths)

    def find_segment_png_files(self, folder_path):
        segment_png_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.startswith('segment') and file.lower().endswith('.png'):
                    segment_png_files.append(os.path.join(root, file))

        return segment_png_files
        
    def crop_to_content(self, image, margin=10):
        image_array = np.array(image)
        top, left, bottom, right = self.find_bounding_box(image_array, margin)
        cropped_image = image_array[top:bottom, left:right]
        return cropped_image
            
    def find_bounding_box(self, image_array, margin=10):
        # Assuming black is the background and thus is the color we want to filter out
        # We're looking for non-black pixels
        non_black_pixels_mask = np.any(image_array != [0, 0, 0], axis=-1)

        # Find the bounding box of those pixels
        non_black_pixels_coords = np.argwhere(non_black_pixels_mask)
        top_left = non_black_pixels_coords.min(axis=0)
        bottom_right = non_black_pixels_coords.max(axis=0)

        # Adding some margin
        top_left = np.maximum([0, 0], top_left - margin)
        bottom_right = np.minimum(image_array.shape[:2], bottom_right + margin)
        return top_left[0], top_left[1], bottom_right[0], bottom_right[1]
        
def custom_collate(batch):
    batch = pad_tensors_to_same_size(batch)
    return torch.stack(batch, 0)

def pad_tensors_to_same_size(tensor_list):
    # Determine the maximum dimensions along each axis
    max_shape = tuple(max(dim_size) for dim_size in zip(*[tensor.shape for tensor in tensor_list]))
    # Initialize a list to store padded tensors
    padded_tensors = []

    # Pad each tensor to the maximum size
    for tensor in tensor_list:
        h_pad = [(max_shape[0] - tensor.shape[0])//2, (max_shape[0] - tensor.shape[0])-(max_shape[0] - tensor.shape[0])//2]
        v_pad = [(max_shape[1] - tensor.shape[1])//2, (max_shape[1] - tensor.shape[1])-(max_shape[1] - tensor.shape[1])//2]
        n_pad = [(max_shape[2] - tensor.shape[2])//2, (max_shape[2] - tensor.shape[2])-(max_shape[2] - tensor.shape[2])//2]
        pad_width = (n_pad[0],n_pad[1],v_pad[0], v_pad[1], h_pad[0], h_pad[1] )
        padded_tensor = torch.nn.functional.pad(tensor, pad=pad_width, mode='constant', value=0)
        padded_tensors.append(padded_tensor)

    return padded_tensors



net = Net(num_classes).to('cuda')#models.resnet18(pretrained=True).to('cuda')#

data = segment_dataset(img_path)
dataloader = DataLoader(data, shuffle = True, batch_size = batch_size, collate_fn = custom_collate)

augmentations = torch.nn.Sequential(RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.3), T.RandomGrayscale(p=0.2),  
            T.RandomHorizontalFlip(), RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p = 0.2), RCrop(),
            T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),)

learner = BYOL(
    net,
    image_size = 1024,
    hidden_layer = 'feature_fc',
    augment_fn = augmentations,
    use_momentum = False       # turn off momentum in the target encoder
)

opt = torch.optim.Adam(learner.parameters(), lr=learning_rate)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, images in enumerate(tqdm(dataloader)):
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        if i!=0 and i % 100 == 0:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    torch.save(net.state_dict(), 'Byol_test/checkpoints_final/epoch_'+str(epoch)+'_save.pt')
print('Finished Training')
