import torch
from torch import nn
from einops import rearrange, repeat
import os
import numpy as np
import glob

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, use_graph_distances=False, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_graph_distances = use_graph_distances
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_fn = nn.functional.softmax
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask):
        q_input = x
        k_input = x 
        v_input = x
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)
        bin_mask = mask == 0
        dots.masked_fill_(~bin_mask, mask_value)
        attn = self.attn_fn(dots, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), None 

class GraphEncoder(nn.Module):
    def __init__(self, latent_dim=1024, max_length=77, use_distances=False): 
        super().__init__()
        self.max_length = max_length  
        self.token_emb = nn.Embedding(1, latent_dim)
        self.attention_layer = Attention(dim=latent_dim, use_graph_distances=use_distances)
        self.use_distances = use_distances

    def forward(self, embedding, adjacency_matrix):
        for i in range(len(embedding)):
            if len(embedding[i]) != self.max_length: 
                embedding_padded = torch.cat([embedding[i], self.token_emb(torch.LongTensor([0] * (self.max_length - len(embedding[i]))).cuda())])
            embedding[i] = embedding_padded
        embeddings = torch.stack(embedding)
        z = self.attention_layer(embeddings, mask=adjacency_matrix) 
        return z

if __name__ == "__main__":
    input_folder = path/where/generated_graphs/removed_nodes_live

    subdirectories = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]

    max_context_length = 256  # Adjust based on your requirement
    latent_dim = 1024
    use_distances = False

    graph_encoder = GraphEncoder(max_length=max_context_length, use_distances=use_distances).cuda()

    for subdir in subdirectories:
        subdir_path = os.path.join(input_folder, subdir)

        print(subdir_path)
        adjacency_matrix_path = os.path.join(subdir_path, 'adjacency_matrix_inverse.npy')
        feature_vector_path = glob.glob(os.path.join(subdir_path, f'{subdir}_feature_vectors_extracted_1024.npy'))
        if feature_vector_path:
            feature_vector_path = feature_vector_path[0]
        else:
            print(f"No feature vector file found in {subdir_path}")
            continue    
        if not os.path.exists(adjacency_matrix_path) or not os.path.exists(feature_vector_path):
            print(f"Missing files in {subdir}")
            continue

        adj_a = np.load(adjacency_matrix_path)
        adjacency_matrix = repeat(torch.diag(torch.ones(max_context_length).cuda()), "h w -> b h w", b=1)
        adjacency_matrix[0][:len(adj_a), :len(adj_a[0])] = torch.Tensor(adj_a).cuda()

        if not use_distances: 
            adjacency_matrix = adjacency_matrix == 0

        feature_vectors = np.load(feature_vector_path)
        embedding = [torch.tensor(feature_vectors).float().cuda()]

        if embedding[0].shape[0] > max_context_length or embedding[0].shape[1] != latent_dim:
            raise ValueError("Feature vector shape is incompatible.")

        conditioning, _ = graph_encoder(embedding, adjacency_matrix)
        conditioning = conditioning.cpu().detach().numpy()

        output_filename = f'conditioning_{subdir}_1024.npy'
        output_path = os.path.join(subdir_path, output_filename)
        np.save(output_path, conditioning)

        print(f"Processed and saved {output_filename}")
