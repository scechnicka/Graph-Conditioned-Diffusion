# Graph Conditioned Diffusion for Controllable Histopathology Image Generation

This repository contains the code and model implementation for the paper **"Graph Conditioned Diffusion for Controllable Histopathology Image Generation"** by Sarah Cechnicka, Matthew Baugh, Weitong Zhang, Mischa Dombrowski, Zhe Li, Johannes C. Paetzold, Candice Roufosse, and Bernhard Kainz.

Graph Conditioned Diffusion (GCD) introduces a novel approach to generate synthetic histopathology images with explicit control over structure and content through graph-based representations. This enables the generation of diverse, privacy-preserving datasets that maintain the statistical properties necessary for training robust downstream models.

📜 [Read the Paper](https://arxiv.org/abs/YOUR_ARXIV_ID)  
🤗 [Try our Interactive Demo](https://huggingface.co/YOUR_DEMO_LINK)

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Graph Construction](#graph-construction)
- [Model Training](#model-training)
- [Generating Synthetic Images](#generating-synthetic-images)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

Graph Conditioned Diffusion addresses key challenges in medical image synthesis:

- **Controllable Generation**: Use graph representations to explicitly control the structure, arrangement, and types of anatomical objects in generated images
- **Enhanced Diversity**: Generate samples that better represent the full distribution of real datasets, including rare cases
- **Privacy Preservation**: Create synthetic datasets that can be shared without privacy concerns while maintaining utility for downstream tasks
- **Downstream Performance**: Synthetic images generated with GCD achieve segmentation performance comparable to models trained on real data

### Key Features

1. **Graph-based Conditioning**: Images are represented as graphs where nodes correspond to anatomical structures (e.g., tubules, glomeruli) and edges encode spatial relationships
2. **Flexible Graph Interventions**: Modify graphs through node removal, class changes, or interpolation to generate targeted variations
3. **Multi-scale Generation**: Cascaded diffusion models generate high-resolution images (1024×1024) with fine structural details
4. **Validated Utility**: Generated images demonstrate effectiveness on real-world segmentation tasks

---

## Environment Setup

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (tested on NVIDIA A100)
- Conda or virtualenv

### Installation

Create a new conda environment and install dependencies:
bash
conda create -y -n gcd python=3.11
conda activate gcd
pip install -e .
Note: Exact package versions are available in requirements.txt if needed.
Dependencies
The main dependencies include:

PyTorch
Diffusers
Transformers
NetworkX (for graph operations)
OpenCV
NumPy, SciPy, Pandas


Data Preparation
Dataset Structure
Organize your histopathology dataset with the following structure:
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
Each image should have a corresponding segmentation mask where different anatomical structures are labeled with distinct class IDs.
Preprocessing

Extract patches from whole slide images (WSIs) at your desired resolution:

bashpython scripts/extract_patches.py \
    --input /path/to/wsi \
    --output data/images \
    --patch_size 1024 \
    --overlap 0

Generate multi-resolution copies for cascaded diffusion training:

bashpython scripts/create_multires_dataset.py \
    --input data/images \
    --output data/multires \
    --sizes 64 256 1024

Graph Construction
Creating Ground Truth Graphs
Generate graph representations from your segmentation masks:
bashpython graphmaker.py \
    --masks data/masks/train \
    --output data/graphs/train \
    --num_classes 4
This creates graphs where:

Nodes represent individual anatomical structures (e.g., each tubule, each glomerulus)
Edges connect structures that are spatially adjacent without obstruction
Node features include class labels, BYOL embeddings, and positional encodings

Graph Feature Extraction
Extract rich feature representations for each graph node:
bashpython concatenate_vectors.py \
    --images data/images/train \
    --masks data/masks/train \
    --graphs data/graphs/train \
    --output data/graph_features/train \
    --byol_model models/byol.pth
Graph Interventions
Generate augmented graphs through various interventions:
Node Removal
bashpython gen_graph_remove_node.py \
    --input data/graphs/train \
    --output data/graphs_augmented/removed
Node Class Change
bashpython gen_graph_change_node.py \
    --input data/graphs/train \
    --output data/graphs_augmented/changed \
    --source_class 1 \
    --target_class 2
Graph Interpolation
bashpython gen_graph_blended.py \
    --input data/graphs/train \
    --output data/graphs_augmented/interpolated \
    --num_samples 1000
Cut-Paste Augmentation
bashpython gen_graph_partial.py \
    --input data/graphs/train \
    --output data/graphs_augmented/cutpaste \
    --max_subgraphs 3

Model Training
Training the Graph Transformer
The graph transformer processes graph structures and generates embeddings for diffusion conditioning:
bashpython graph_conditioning_embedder.py \
    --graph_dir data/graphs/train \
    --features_dir data/graph_features/train \
    --output models/graph_transformer \
    --num_layers 6 \
    --hidden_dim 512 \
    --num_heads 8 \
    --epochs 100
Training the Cascaded Diffusion Models
Base Model (64×64)
bashpython train_diffusion.py \
    --images data/multires/64 \
    --graphs data/graphs/train \
    --graph_model models/graph_transformer \
    --output models/diffusion_base \
    --resolution 64 \
    --batch_size 64 \
    --epochs 500
Super-Resolution Model 1 (64→256)
bashpython train_diffusion.py \
    --images data/multires/256 \
    --graphs data/graphs/train \
    --graph_model models/graph_transformer \
    --conditioning_images data/multires/64 \
    --output models/diffusion_sr1 \
    --resolution 256 \
    --batch_size 32 \
    --epochs 300
Super-Resolution Model 2 (256→1024)
bashpython train_diffusion.py \
    --images data/multires/1024 \
    --graphs data/graphs/train \
    --graph_model models/graph_transformer \
    --conditioning_images data/multires/256 \
    --output models/diffusion_sr2 \
    --resolution 1024 \
    --batch_size 8 \
    --epochs 300

Generating Synthetic Images
Basic Generation
Generate synthetic images from existing graphs:
bashpython generate_images.py \
    --graphs data/graphs/test \
    --base_model models/diffusion_base \
    --sr1_model models/diffusion_sr1 \
    --sr2_model models/diffusion_sr2 \
    --graph_model models/graph_transformer \
    --output samples/synthetic \
    --num_samples 100 \
    --batch_size 4
Controlled Generation with Interventions
Generate images with modified graph structures:
bashpython generate_images.py \
    --graphs data/graphs_augmented/interpolated \
    --base_model models/diffusion_base \
    --sr1_model models/diffusion_sr1 \
    --sr2_model models/diffusion_sr2 \
    --graph_model models/graph_transformer \
    --output samples/synthetic_controlled \
    --num_samples 500 \
    --batch_size 4
Complete Pipeline
Run the full generation pipeline including graph creation and interventions:
bashbash scripts/generate_full_dataset.sh \
    --size 1000 \
    --output datasets/synthetic

Evaluation
Image Quality Metrics
Evaluate generated images using FID, Precision, and Recall:
bashpython evaluation.py \
    --real_images data/images/test \
    --generated_images samples/synthetic \
    --metrics fid precision recall \
    --output results/quality_metrics.json
Downstream Segmentation Task
Train a segmentation model on synthetic data and evaluate on real test data:
bashpython train_segmentation.py \
    --train_images samples/synthetic \
    --train_masks samples/synthetic_masks \
    --test_images data/images/test \
    --test_masks data/masks/test \
    --output experiments/segmentation \
    --epochs 100
Expected metrics:

Dice Score: ~89.85%
AJI Score: ~66.60%

Improved Precision and Recall
Compute improved precision and recall metrics:
bashpython impr_prec_rec.py \
    --real_dir data/images/test \
    --generated_dir samples/synthetic \
    --output results/precision_recall.json

Results
Image Quality Comparison
MethodIP↑IR↑FID↓Unconditional Diffusion0.820.5710.35Mask Conditioned Diffusion0.360.07162.43GCD (Ours, Image)0.900.3079.11GCD (Ours, Text)0.770.6439.78
Downstream Segmentation Performance
MethodDice (%)↑AJI (%)↑Trained on Real Data88.0162.05Unconditional Diffusion90.4466.80Mask Conditioned82.0042.40GCD (Interpolated)89.8566.60GCD (Text)86.0559.45

Citation
If you use this code or find our work helpful, please cite:
bibtex@inproceedings{cechnicka2024graph,
  title={Graph Conditioned Diffusion for Controllable Histopathology Image Generation},
  author={Cechnicka, Sarah and Baugh, Matthew and Zhang, Weitong and Dombrowski, Mischa and Li, Zhe and Paetzold, Johannes C. and Roufosse, Candice and Kainz, Bernhard},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024},
  organization={Springer}
}

Acknowledgements
This work was supported by:

UKRI Centre for Doctoral Training in AI for Healthcare (EP/S023283/1)
ERC project MIA-NORMAL 101083647
State of Bavaria (HTA) and DFG 512819079
NHR@FAU of FAU Erlangen-Nürnberg (NHR project b180dc)
NIHR Biomedical Research Centre at Imperial College Healthcare NHS Trust
Support from Sidharth and Indira Burman

Human samples used in this research were obtained from the Imperial College Healthcare Tissue & Biobank (ICHTB), approved by Wales REC3 (22/WA/2836).

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or issues, please:

Open an issue on GitHub
Contact: sc7718@imperial.ac.uk


Repository Structure
.
├── README.md
├── requirements.txt
├── graphmaker.py                          # Graph construction from masks
├── graph_conditioning_embedder.py         # Graph transformer training
├── graph_conditioning_embedder_generated.py
├── graph_conditioning_embedder_generated_change.py
├── graph_conditioning_embedder_generated_remove.py
├── evaluation.py                          # Quality metrics evaluation
├── impr_prec_rec.py                      # Precision/recall computation
├── gen_graph_remove_node.py              # Node removal intervention
├── gen_graph_change_node.py              # Node class change intervention
├── gen_graph_one_node_swaped.py
├── gen_graph_blended.py                  # Graph interpolation
├── gen_graph_blended_extracted.py
├── gen_graph_partial.py                  # Cut-paste augmentation
├── gen_graph_partial_extracted.py
├── gen_graph_partial_extracted_short.py
├── gen_graph_partial_short.py
├── gen_changed_image_graph.py
├── gen_removed_image_graph.py
├── gen_graph_augmentation.py
├── concatenate_vectors.py                # Feature extraction
├── positional_embedding.py               # Positional encodings
├── transformer_module.py                 # Graph transformer architecture
├── byol_test.py                          # BYOL feature extraction
├── ccn_extractor.py
├── class_encoding.py
├── converter.py
├── graphmaker_test.py
├── manual_extractor.py
├── mean_counter.py
├── temp.py
└── generate_graphs/                      # Graph generation utilities
