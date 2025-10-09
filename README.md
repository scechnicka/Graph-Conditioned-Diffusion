# Graph Conditioned Diffusion for Controllable Histopathology Image Generation

This repository contains the code and model implementation for the paper **"Graph Conditioned Diffusion for Controllable Histopathology Image Generation"** by Sarah Cechnicka, Matthew Baugh, Weitong Zhang, Mischa Dombrowski, Zhe Li, Johannes C. Paetzold, Candice Roufosse, and Bernhard Kainz.

Graph Conditioned Diffusion (GCD) introduces a novel approach to generate synthetic histopathology images with explicit control over structure and content through graph-based representations. This enables the generation of diverse, privacy-preserving datasets that maintain the statistical properties necessary for training robust downstream models.

📜 [Read the Paper](https://arxiv.org/abs/2510.07129)  

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

```bash
conda create -y -n gcd python=3.11
conda activate gcd
pip install -e .
```

Note: Exact package versions are available in `requirements.txt` if needed.

### Dependencies

The main dependencies include:

- PyTorch
- Diffusers
- Transformers
- NetworkX (for graph operations)
- OpenCV
- NumPy, SciPy, Pandas

---

## Data Preparation

### Dataset Structure

Organize your histopathology dataset with the following structure:

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── masks/
    ├── train/
    ├── val/
    └── test/
```

Each image should have a corresponding segmentation mask where different anatomical structures are labeled with distinct class IDs.

### Preprocessing

Prepare your data by:

1. Extracting patches from whole slide images (WSIs) at your desired resolution (e.g., 1024×1024)
2. Creating corresponding segmentation masks with distinct class IDs for each anatomical structure
3. Generating multi-resolution versions (64×64, 256×256, 1024×1024) for cascaded diffusion training
---

## Graph Construction

### Creating Ground Truth Graphs

Generate graph representations from your segmentation masks:

```bash
python graphmaker.py --masks data/masks/train --output data/graphs/train --num_classes 4
```

This creates graphs where:

- Nodes represent individual anatomical structures (e.g., each tubule, each glomerulus)
- Edges connect structures that are spatially adjacent without obstruction
- Node features include class labels, BYOL embeddings, and positional encodings

### Graph Feature Extraction

Extract rich feature representations for each graph node:

```bash
python concatenate_vectors.py --images data/images/train --masks data/masks/train --graphs data/graphs/train --output data/graph_features/train --byol_model models/byol.pth
```

### Graph Interventions

Generate augmented graphs through various interventions:

**Node Removal:**

```bash
python gen_graph_remove_node.py --input data/graphs/train --output data/graphs_augmented/removed
```

**Node Class Change:**

```bash
python gen_graph_change_node.py --input data/graphs/train --output data/graphs_augmented/changed --source_class 1 --target_class 2
```

**Graph Interpolation:**

```bash
python gen_graph_blended.py --input data/graphs/train --output data/graphs_augmented/interpolated --num_samples 1000
```

**Cut-Paste Augmentation:**

```bash
python gen_graph_partial.py --input data/graphs/train --output data/graphs_augmented/cutpaste --max_subgraphs 3
```

---

## Model Training

### Training the Graph Transformer

The graph transformer processes graph structures and generates embeddings for diffusion conditioning:

```bash
python graph_conditioning_embedder.py --graph_dir data/graphs/train --features_dir data/graph_features/train --output models/graph_transformer --num_layers 6 --hidden_dim 512 --num_heads 8 --epochs 100
```

### Training the Cascaded Diffusion Models

**Base Model (64×64):**

```bash
python train_diffusion.py --images data/multires/64 --graphs data/graphs/train --graph_model models/graph_transformer --output models/diffusion_base --resolution 64 --batch_size 64 --epochs 500
```

**Super-Resolution Model 1 (64→256):**

```bash
python train_diffusion.py --images data/multires/256 --graphs data/graphs/train --graph_model models/graph_transformer --conditioning_images data/multires/64 --output models/diffusion_sr1 --resolution 256 --batch_size 32 --epochs 300
```

**Super-Resolution Model 2 (256→1024):**

```bash
python train_diffusion.py --images data/multires/1024 --graphs data/graphs/train --graph_model models/graph_transformer --conditioning_images data/multires/256 --output models/diffusion_sr2 --resolution 1024 --batch_size 8 --epochs 300
```

---

## Generating Synthetic Images

### Basic Generation

Generate synthetic images from existing graphs:

```bash
python generate_images.py --graphs data/graphs/test --base_model models/diffusion_base --sr1_model models/diffusion_sr1 --sr2_model models/diffusion_sr2 --graph_model models/graph_transformer --output samples/synthetic --num_samples 100 --batch_size 4
```

### Controlled Generation with Interventions

Generate images with modified graph structures:

```bash
python generate_images.py --graphs data/graphs_augmented/interpolated --base_model models/diffusion_base --sr1_model models/diffusion_sr1 --sr2_model models/diffusion_sr2 --graph_model models/graph_transformer --output samples/synthetic_controlled --num_samples 500 --batch_size 4
```

### Complete Pipeline

Run the full generation pipeline including graph creation and interventions:

```bash
bash scripts/generate_full_dataset.sh --size 1000 --output datasets/synthetic
```

---

## Evaluation

### Image Quality Metrics

Evaluate generated images using FID, Precision, and Recall:

```bash
python evaluation.py --real_images data/images/test --generated_images samples/synthetic --metrics fid precision recall --output results/quality_metrics.json
```

### Downstream Segmentation Task

Train a segmentation model on synthetic data and evaluate on real test data:

```bash
python train_segmentation.py --train_images samples/synthetic --train_masks samples/synthetic_masks --test_images data/images/test --test_masks data/masks/test --output experiments/segmentation --epochs 100
```

Expected metrics:

- Dice Score: ~89.85%
- AJI Score: ~66.60%

### Improved Precision and Recall

Compute improved precision and recall metrics:

```bash
python impr_prec_rec.py --real_dir data/images/test --generated_dir samples/synthetic --output results/precision_recall.json
```

---

## Results

### Image Quality Comparison

| Method | IP ↑ | IR ↑ | FID ↓ |
|--------|------|------|-------|
| Unconditional Diffusion | 0.82 | 0.57 | 10.35 |
| Mask Conditioned Diffusion | 0.36 | 0.07 | 162.43 |
| GCD (Ours, Image) | 0.90 | 0.30 | 79.11 |
| GCD (Ours, Text) | 0.77 | 0.64 | 39.78 |

### Downstream Segmentation Performance

| Method | Dice (%) ↑ | AJI (%) ↑ |
|--------|------------|-----------|
| Trained on Real Data | 88.01 | 62.05 |
| Unconditional Diffusion | 90.44 | 66.80 |
| Mask Conditioned | 82.00 | 42.40 |
| GCD (Interpolated) | 89.85 | 66.60 |
| GCD (Text) | 86.05 | 59.45 |

---

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-06103-4_17,
author="Cechnicka, Sarah
and Baugh, Matthew
and Zhang, Weitong
and Dombrowski, Mischa
and Li, Zhe
and Paetzold, Johannes C.
and Roufosse, Candice
and Kainz, Bernhard",
editor="Felsner, Lina
and K{\"u}stner, Thomas
and Maier, Andreas
and Qin, Chen
and Ahmadi, Seyed-Ahmad
and Kazi, Anees
and Hu, Xiaoling",
title="Graph Conditioned Diffusion for Controllable Histopathology Image Generation",
booktitle="Reconstruction and Imaging Motion Estimation, and Graphs in Biomedical Image Analysis",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="172--183",
abstract="Recent advances in Diffusion Probabilistic Models (DPMs) have set new standards in high-quality image synthesis. Yet, controlled generation remains challenging particularly in sensitive areas such as medical imaging. Medical images feature inherent structure such as consistent spatial arrangement, shape or texture, all of which are critical for diagnosis. However, existing DPMs operate in noisy latent spaces that lack semantic structure and strong priors, making it difficult to ensure meaningful control over generated content. To address this, we propose graph-based object-level representations for Graph-Conditioned-Diffusion. Our approach generates graph nodes corresponding to each major structure in the image, encapsulating their individual features and relationships. These graph representations are processed by a transformer module and integrated into a diffusion model via the text-conditioning mechanism, enabling fine-grained control over generation. We evaluate this approach using a real-world histopathology use case, demonstrating that our generated data can reliably substitute for annotated patient data in downstream segmentation tasks. The code is available here.",
isbn="978-3-032-06103-4"
}

```

---

## Acknowledgements

This work was supported by:

- UKRI Centre for Doctoral Training in AI for Healthcare (EP/S023283/1)
- ERC project MIA-NORMAL 101083647
- State of Bavaria (HTA) and DFG 512819079
- NHR@FAU of FAU Erlangen-Nürnberg (NHR project b180dc)
- NIHR Biomedical Research Centre at Imperial College Healthcare NHS Trust
- Support from Sidharth and Indira Burman

Human samples used in this research were obtained from the Imperial College Healthcare Tissue & Biobank (ICHTB), approved by Wales REC3 (22/WA/2836).

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or issues, please:

- Open an issue on GitHub
- Contact: sc7718@imperial.ac.uk

---

## Repository Structure

```
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
```
