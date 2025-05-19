import torch
from torch import nn
from einops import repeat
import os
import numpy as np
"""Based on https://github.com/lucidrains/x-transformers/tree/main/x_transformers
https://github.com/pesser/stable-diffusion/blob/main/ldm/modules/x_transformer.py"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple
from einops import rearrange, repeat, reduce
import glob
import re

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            use_graph_distances=False,
            dropout=0.,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.use_graph_distances = use_graph_distances

        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.attn_fn = F.softmax

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
            self,
            x,
            mask,
    ):
        q_input = x
        k_input = x 
        v_input = x

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if not self.use_graph_distances:
            # should be implicit by position
            bin_mask = mask == 0
        else: 
            raise NotImplementedError("Graph distances not implemented")
            o
        dots.masked_fill_(~bin_mask, mask_value)

        attn = self.attn_fn(dots, dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), None 



class GraphEncoder(nn.Module):
    """Uses a self-attention layer to encode information with inherent graph restraints"""
    def __init__(self, latent_dim=1024, max_length=77, use_distances=False): 
        super().__init__()
        self.max_length = max_length  
        self.token_emb = nn.Embedding(1, latent_dim)
        self.attention_layer = Attention(dim=latent_dim, use_graph_distances=use_distances)
        self.use_distances = use_distances

    def forward(self, embedding, adajcency_matrix):
        """
        Forward pass for GraphEncoder model. 

        Args:
            embedding (list): list of B samples where each sample has the shape l x self.latent_dim with l <= self.max_length 
            adjacency matrix (torch.tensor): B x self.max_length x self.max_length adjacency matrix.  

        """

        # missing embeddings are padded with a learnable embedding
        for i in range(len(embedding)):  # batch size
            if len(embedding[i]) != self.max_length: 
                embedding_padded = torch.cat([embedding[i], self.token_emb(torch.LongTensor([0,] * (self.max_length - len(embedding[i]) )).cuda())])
            embedding[i] = embedding_padded
        embeddings = torch.stack(embedding)
        z = self.attention_layer(embeddings, mask=adjacency_matrix) 
        return z




if __name__ == "__main__":
    # distance is supposed to be between 1 (same node, 0 distance) to 0 (not connected). Small values mean far away
    #adj_a = [[  1, 0.5, 0.5,   0], 
    #         [0.5,   1, 0.5,   0],
    #         [0.5, 0.5,   1, 0.2],
    #         [  0,   0, 0.2,   1],
    #         ]
    #use_distances = True # if false treats all distances > 0 as 'connected' and binarizes mask
    # if we want to add distances we have to add it to the Attention class
    # New single folder path
    input_folder = '/home/atuin/b143dc/b143dc22/GCD/generated_graphs/partial_manual_1024'

    # Scan the directory for feature vector and adjacency matrix files
    feature_vector_files = glob.glob(os.path.join(input_folder, 'partial*_features.npy'))
    adjacency_matrix_files = glob.glob(os.path.join(input_folder, 'partial*_adjacency.npy'))

    # Organize files into a dict with the graph identifier as the key
    graphs = {}
    for f in feature_vector_files:
        identifier = "_".join(os.path.basename(f).split('_')[1:4])  # Extract A_B_C
        if identifier not in graphs:
            graphs[identifier] = {'features': None, 'adjacency': None}
        graphs[identifier]['features'] = f

    for a in adjacency_matrix_files:
        identifier = "_".join(os.path.basename(a).split('_')[1:4])  # Extract A_B_C
        if identifier in graphs:
            graphs[identifier]['adjacency'] = a

    # Process each graph
    for identifier, files in graphs.items():
        if not files['features'] or not files['adjacency']:
            print(f"Missing files for graph {identifier}")
            continue

        # Load feature vectors and adjacency matrix
        features = np.load(files['features'])

        use_distances = False
        bs = 1 # batchsize
        max_context_length=256 # max context of language encoder. Maximum number of nodes we can have 
        latent_dim = 1024 # output dim of feature encoder

        # model
        graph_encoder = GraphEncoder(max_length=max_context_length, use_distances=False).cuda()
        
        # example adjacency
        adj_a = np.load(files['adjacency'])# Load the adjacency matrix from the file
        
            
            
        # covert adjacency matrix to B x Lmax x Lmax - has to be done sample wise
        adjacency_matrix = repeat(torch.diag(torch.ones(max_context_length).cuda()), "h w -> b h w", b=bs)
        adjacency_matrix[0][:len(adj_a), :len(adj_a[0])] = torch.Tensor(adj_a).cuda()
        if not use_distances: 
            adjacency_matrix = adjacency_matrix == 0
            
        # list of B learned embeddings where B is the number of nodes. 
        # Graph size can be different for each sample but cannot exceed max_context_length.
        # Embeddings hold information of absolute position and content of the node
        graph_size = len(adj_a) 
        feature_vectors = np.load(files['features']) # Load the feature vectors from the file
        embedding = [torch.tensor(feature_vectors).float().cuda()]
        print(embedding[0].shape[0],embedding[0].shape[1])

        # Ensure the loaded embeddings are within the expected dimensions
        if embedding[0].shape[0] > max_context_length or embedding[0].shape[1] != latent_dim:
            raise ValueError("Loaded feature vectors' shape is not compatible with the expected dimensions.")

        
        # forward pass
        conditioning, _ = graph_encoder(embedding, adjacency_matrix)
        # Save the output vector with the new naming convention
        conditioning = conditioning.cpu().detach().numpy()
        output_filename = f'conditioning_{identifier}_1024.npy'
        output_path = os.path.join(input_folder, output_filename)
        np.save(output_path, conditioning)

        print(f"Processed and saved {output_filename}")
