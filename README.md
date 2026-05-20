# Universal Monocular Metric Depth Estimation (PoC)

This repository contains the official PyTorch implementation and Proof-of-Concept (PoC) code for our Master's Computer Vision paper on adapting zero-shot relative depth models for metric scale accuracy.

## Overview
State-of-the-art foundation models for depth estimation (e.g., Depth Anything) are highly capable of predicting relative depth (structural layout) but suffer from severe scale drift when estimating absolute metric depth. This project proposes a **Lightweight Metric Adaptation Head (LMAH)** that maps normalized relative depth to absolute real-world distances (in meters) without requiring massive computational resources.

By freezing the massive feature extraction backbone and training only a ~2,400-parameter head, this pipeline successfully corrects global scale ambiguity and effectively doubles localized threshold accuracy.

## Architecture Pipeline
1. **Input:** 2D RGB Image.
2. **Frozen Backbone:** Pre-trained `nielsr/depth-anything-small` (loaded via Hugging Face). All parameters are completely frozen; no gradient flows into the feature extractor.
3. **LMAH (Ours):** A 3-layer Convolutional block with GELU activations and a Softplus output to ensure strictly positive depth predictions. 
4. **Output:** Dense, absolute metric depth map (meters).

## Dataset
Evaluation is performed on a micro-benchmark of the **NYU Depth V2** dataset (indoor RGB-D pairs captured via Microsoft Kinect). 
* **Streaming Protocol:** To bypass the 30GB+ archive download and avoid Hugging Face legacy script deprecation errors, the `DataLoader` uses an instant Parquet streaming implementation targeting `nielsr/nyu-depth-v2`.

## Quantitative Results
Evaluated on the NYU Depth V2 micro-benchmark (Test split). To ensure fair comparison, the frozen baseline predictions are aligned using standard median-ratio scaling prior to evaluation.

| Method | RMSE (meters) $\downarrow$ | Threshold Accuracy ($\delta < 1.25$) $\uparrow$ |
| :--- | :---: | :---: |
| Baseline (Depth Anything scaled) | 3.641 | 15.1% |
| **Ours (Frozen Backbone + LMAH)** | **2.125** | **30.2%** |

*Note: The proposed LMAH drops the Root Mean Squared Error by over 1.5 meters and doubles the structural threshold accuracy.*

## Installation and Usage
This script is highly optimized and can be run end-to-end on a free Google Colab T4 GPU in under 5 minutes.

### 1. Requirements
Ensure you have the following libraries installed:
```bash
pip install torch torchvision transformers datasets matplotlib pillow
