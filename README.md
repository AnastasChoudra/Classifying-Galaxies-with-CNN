# 🌌 Classifying Galaxies with Convolutional Neural Networks (CNNs)

&gt; An end-to-end deep-learning pipeline that automatically categorises deep-space galaxy images into four morphological classes using TensorFlow-GPU.

---

## 📋 Purpose & Context

Modern telescopes (orbital and ground-based) collect millions of images every night.  
Before scientists can study them, the pictures must be **labelled**—a slow, manual task.  
This project builds a CNN that replicates crowd-sourced Galaxy Zoo annotations, letting researchers instantly filter for **galaxies with rings, mergers or irregular structure**.

| One-hot vector | Class description |
|----------------|-------------------|
| `[1,0,0,0]`    | Regular (featureless) |
| `[0,1,0,0]`    | Ringed |
| `[0,0,1,0]`    | Merger |
| `[0,0,0,1]`    | Irregular / Other |

---

## ⚙️ Method

1. **Data**  
   - 1 400 RGB galaxy images (128 × 128 px) from Galaxy Zoo  
   - Train / validation split: 80 / 20 %, stratified on one-hot labels

2. **Pre-processing**  
   - `ImageDataGenerator` with rescale `1./255`  
   - Batch size = 5 (GPU memory conservative)

    **Total parameters:** 7.164

3. **Training**  
- Optimiser: Adam (lr = 0.001)  
- Loss: Categorical Cross-entropy  
- Metrics: Categorical Accuracy + AUC  
- Epochs: 8  
- Hardware: NVIDIA RTX 4070-Super 12 GB (CUDA 11.2 + cuDNN 8.1)

4. **Visualisation**  
Activation heat-maps for each convolutional layer are saved to `static/images/` for qualitative inspection.

---

## 📊 Results (8 epochs)

| metric | training | validation |
|--------|----------|------------|
| Categorical Accuracy | 0.575 | **0.707** |
| AUC | 0.817 | **0.891** |

Sample predictions on held-out images:

| Galaxy | True class | Predicted probabilities | Correct? |
|--------|------------|-------------------------|----------|
| Galaxy_0 | Other | [0.19, 0.19, 0.27, **0.34**] | ✅ |
| Galaxy_3 | Regular | [**0.47**, 0.22, 0.14, 0.16] | ✅ |
| Galaxy_4 | Merger | [0.09, 0.02, **0.76**, 0.14] | ✅ |

Training time per epoch: **≈ 1 s on GPU** vs. **≈ 10–15 s on CPU**.

---

## 🚀 Quick Start

1. Clone the repo
2. Install requirements
```bash
pip install -r requirements.txt
