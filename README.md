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
```bash
conda create -y -n gcd python=3.11
conda activate gcd
pip install -e .
