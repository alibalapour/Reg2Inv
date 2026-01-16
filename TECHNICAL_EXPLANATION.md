# Reg2Inv: Technical Explanation

## Overview

**Reg2Inv** (Registration is a Powerful Rotation-Invariance Learner for 3D Anomaly Detection) is a deep learning framework designed to detect anomalies in 3D point cloud data. This method is particularly useful in industrial quality control scenarios where identifying structural defects in manufactured objects is critical.

The core innovation of Reg2Inv is the integration of **point cloud registration** with **memory-based anomaly detection**. By learning features through a registration task, the model acquires rotation-invariant representations that are highly effective for detecting anomalies.

---

## System Architecture

The Reg2Inv pipeline consists of two main stages:

### 1. Feature Learning Stage (Training)

During training, the model learns to extract features through a **point cloud registration task**. This involves:

- **Geometric Alignment**: Learning to align pairs of point clouds with different rotations and transformations
- **Multi-scale Consistency**: Enforcing feature consistency across different scales of point cloud resolution

### 2. Anomaly Detection Stage (Inference)

At inference time, the model:

1. Computes a **registration transformation matrix** to align the test point cloud with a reference prototype
2. Extracts **rotation-invariant features** for anomaly detection
3. Uses a **memory bank** (PatchCore-based) to score anomalies by comparing features against normal training samples

---

## Core Components

### 1. GeoTransformer Model (`model.py`)

The main model class that orchestrates the entire pipeline.

```python
class GeoTransformer(nn.Module):
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `RIKPConvFPN` (Backbone) | Extracts multi-scale rotation-invariant features from point clouds |
| `GeometricTransformer` | Conditional transformer for cross-attention between source and reference point clouds |
| `SuperPointTargetGenerator` | Generates ground truth correspondences during training |
| `SuperPointMatching` | Performs coarse-level point matching using feature similarity |
| `LocalGlobalRegistration` | Fine-grained registration with RANSAC-based pose estimation |
| `LearnableLogOptimalTransport` | Sinkhorn-based optimal transport for soft correspondence matching |

**Forward Pass Flow:**

1. **Point Cloud Downsampling**: Multi-scale voxelization of input point clouds
2. **Ground Truth Generation**: Computing node correspondences from transformation matrix
3. **Feature Extraction**: RIKPConvFPN backbone produces multi-scale features
4. **Transformer Processing**: Cross-attention between source and reference features
5. **Coarse Matching**: Finding superpoint correspondences
6. **Fine Matching**: Optimal transport for point-level matching
7. **Transform Estimation**: RANSAC-based pose estimation (inference only)

---

### 2. Rotation-Invariant Backbone (`backbone.py`)

```python
class RIKPConvFPN(nn.Module):
```

This is a **Feature Pyramid Network (FPN)** architecture combining:

- **RIConv2** (Rotation-Invariant Convolution): Extracts rotation-invariant local features
- **KPConv** (Kernel Point Convolution): Efficient 3D point cloud convolution

**Encoder-Decoder Architecture:**

```
Encoder:
├── encoder0 (RIConv2SetAbstraction): Initial RI feature extraction
├── encoder1_1/1_2: First-scale KPConv blocks
├── encoder2_1/2_2/2_3: Second-scale KPConv blocks (with strided downsampling)
└── encoder3_1/3_2/3_3: Third-scale KPConv blocks (with strided downsampling)

Decoder:
├── decoder2: Upsamples and fuses s3 + s2 features
└── decoder1: Upsamples and fuses to produce final features
```

**Output**: Three levels of features `[feats_f, feats_i, feats_c]`:
- `feats_f`: Fine-level features (highest resolution)
- `feats_i`: Intermediate features (used for anomaly detection)
- `feats_c`: Coarse-level features (for registration matching)

---

### 3. Rotation-Invariant Features (`utils/riconv_utils.py`)

The rotation invariance is achieved through **Local Reference Axis (LRA)** computation and **geometric feature encoding**.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `compute_LRA()` | Computes weighted Local Reference Axis using PCA on local neighborhoods |
| `compute_norms()` | Estimates surface normals using Open3D |
| `sample_and_group()` | Groups nearby points and computes RI features |
| `RI_features()` | Computes 8-dimensional rotation-invariant features |

**8 Rotation-Invariant Features:**

1. `grouped_xyz_length`: Distance from center point to neighbors
2. `proj_inner_angle_feat`: Angle differences on projected plane
3. `grouped_xyz_angle_0`: Angle between neighbor direction and surface normal
4. `grouped_xyz_angle_1`: Angle between neighbor direction and neighbor's normal
5. `grouped_xyz_angle_norm`: Signed angle between normals
6. `grouped_xyz_inner_angle_0`: Inner edge angle feature 1
7. `grouped_xyz_inner_angle_1`: Inner edge angle feature 2
8. `grouped_xyz_inner_angle_2`: Angle between adjacent neighbor normals

---

### 4. Loss Functions (`loss.py`)

The training uses a **multi-task loss** combining three components:

```python
loss = weight_coarse * coarse_loss + weight_fine * fine_loss + weight_in * in_loss
```

#### CoarseMatchingLoss
- **Purpose**: Supervises superpoint (node) correspondence learning
- **Method**: Weighted Circle Loss on feature distances
- **Positive pairs**: Points with high overlap in ground truth
- **Negative pairs**: Points with zero overlap

#### FineMatchingLoss
- **Purpose**: Supervises point-level correspondence learning
- **Method**: Negative log-likelihood on optimal transport matching scores
- **Labels**: Binary ground truth correspondences based on distance threshold

#### InMatchingLoss
- **Purpose**: Additional intermediate-level matching supervision
- **Method**: Same as FineMatchingLoss but on intermediate features (`feats_i`)

---

### 5. PatchCore Anomaly Detection (`patchcore/patchcore.py`)

After feature extraction, Reg2Inv uses a **memory bank-based** approach for anomaly scoring.

```python
class PatchCore(torch.nn.Module):
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `build_memory_bank()` | Builds a FAISS index from training features (with coreset sampling) |
| `predict_score()` | Scores test samples by nearest neighbor distance to memory bank |

**Anomaly Scoring Process:**

1. **Feature Extraction**: Extract position + shape features from test sample
2. **Normalization**: Normalize features using training set statistics
3. **Patch-level Scoring**: Find k-nearest neighbors in memory bank
4. **Point-level Scoring**: Propagate scores to original point cloud resolution
5. **Image-level Scoring**: Pool patch scores for sample-level anomaly detection

---

## Training Pipeline

### `train_real3dad.py` / `train_shapenet.py`

```python
class Trainer(IterBasedTrainer):
```

**Training Loop:**

1. **Data Loading**: Load point cloud pairs with augmentations (rotation, translation, noise, cropping)
2. **Forward Pass**: Process through GeoTransformer model
3. **Loss Computation**: Calculate multi-task registration loss
4. **Optimization**: Adam optimizer with warmup cosine learning rate schedule
5. **Checkpointing**: Save model snapshots at regular intervals

**Data Augmentation:**
- Random rotation (up to 360°)
- Random translation
- Gaussian noise addition
- Partial point cloud cropping

---

## Testing/Inference Pipeline

### `test_real3dad.py` / `test_shapenet.py`

**Phase 1: Memory Bank Construction**

```python
for data_dict in train_loader:
    output_dict = deep_feature_extractor(data_dict)
    # Extract aligned position + shape features
    train_features.append(features)

memory_feature = PatchCore.build_memory_bank(train_features, memory_size)
```

**Phase 2: Test Sample Evaluation**

```python
for data_dict in test_loader:
    output_dict = deep_feature_extractor(data_dict)
    # Extract features and align using estimated transform
    scores, segmentations = PatchCore.predict_score(...)
```

**Evaluation Metrics:**
- **Image-level AUROC**: Sample-level anomaly detection performance
- **Pixel-level AUROC**: Point-level anomaly localization performance
- **Average Precision (AP)**: Precision-recall based metrics

---

## Configuration System

### `config_real3dad.py` / `config_shapenet.py`

Configuration is managed using **EasyDict** for hierarchical parameter organization.

**Key Configuration Groups:**

| Group | Parameters |
|-------|------------|
| `data` | Dataset paths, voxel size, augmentation settings |
| `backbone` | Network architecture parameters (channels, kernel size, radius) |
| `model` | Global model settings (patch size, Sinkhorn iterations) |
| `geotransformer` | Transformer architecture (heads, blocks, hidden dim) |
| `coarse_matching` | Superpoint matching parameters |
| `fine_matching` | Point-level matching parameters (RANSAC config) |
| `loss` | Loss function weights |
| `optim` | Training hyperparameters (learning rate, iterations) |

---

## Dataset Structure

### Supported Datasets:

1. **Real3D-AD**: Real-world industrial parts (12 categories)
2. **Anomaly-ShapeNet**: Synthetic CAD models (40 categories)

### Data Format:

```
data/
├── Real3D-AD-PCD/
│   └── {category}/
│       ├── train/          # Normal samples (.pcd files)
│       ├── test/           # Test samples with anomalies
│       └── gt/             # Ground truth masks (.txt files)
└── Anomaly-ShapeNet-v2/
    └── dataset/pcd/
        └── {category}/
            ├── train/      # Normal prototype templates
            ├── test/       # Anomalous test samples
            └── GT/         # Ground truth annotations
```

### Dataset Classes:

| Class | Purpose |
|-------|---------|
| `Real3dadPairDataset_train` | Training pairs for registration learning |
| `Real3dadPairDataset_test` | Test samples with prototype alignment |
| `ShapeNetPairDataset_train` | ShapeNet training pairs |
| `ShapeNetPairDataset_test` | ShapeNet test samples |

---

## Key Mathematical Concepts

### 1. Point Cloud Registration

Given two point clouds **P** (source) and **Q** (reference), find transformation **T** = [**R** | **t**] where:
- **R** ∈ SO(3): Rotation matrix
- **t** ∈ ℝ³: Translation vector

### 2. Optimal Transport (Sinkhorn Algorithm)

Soft correspondence matching using entropy-regularized optimal transport:

```
P* = argmin_P <C, P> - ε H(P)
```

Where:
- **C**: Cost matrix (feature distances)
- **P**: Transport plan (soft correspondences)
- **H(P)**: Entropy regularization
- **ε**: Temperature parameter

### 3. KPConv (Kernel Point Convolution)

3D convolution defined by a set of kernel points:

```
f_out(x) = Σ_i w_i · g(||x - x_i||) · f_in(x_i)
```

Where:
- **w_i**: Learnable kernel weights
- **g**: Distance correlation function
- **x_i**: Neighbor points

### 4. Rotation-Invariant Feature Encoding

Features are computed relative to local reference frames:
- Surface normal defines the local z-axis
- Neighbors are ordered by projection angle on the tangent plane
- All features are expressed in this local coordinate system

---

## File Structure Summary

```
Reg2Inv/
├── model.py                 # Main GeoTransformer model
├── backbone.py              # RIKPConvFPN feature extractor
├── loss.py                  # Multi-task loss functions
├── train_real3dad.py        # Training script for Real3D-AD
├── train_shapenet.py        # Training script for Anomaly-ShapeNet
├── test.py                  # Test orchestration script
├── test_real3dad.py         # Testing for Real3D-AD
├── test_shapenet.py         # Testing for Anomaly-ShapeNet
├── config_real3dad.py       # Configuration for Real3D-AD
├── config_shapenet.py       # Configuration for Anomaly-ShapeNet
├── dataset_real3dad.py      # Data loaders for Real3D-AD
├── dataset_shapenet.py      # Data loaders for Anomaly-ShapeNet
├── utils/
│   ├── riconv_utils.py      # Rotation-invariant feature computation
│   ├── cpu_knn.py           # CPU-based KNN utilities
│   ├── visualization.py     # Visualization functions
│   └── utils.py             # General utilities
├── geotransformer/          # GeoTransformer library
│   ├── modules/             # Neural network modules
│   ├── datasets/            # Dataset implementations
│   ├── engine/              # Training engine
│   └── utils/               # Utility functions
└── patchcore/               # PatchCore anomaly detection
    ├── patchcore.py         # Main PatchCore class
    ├── common.py            # Common utilities (FAISS, aggregator)
    ├── sampler.py           # Coreset sampling strategies
    └── metrics.py           # Evaluation metrics
```

---

## Dependencies

The system relies on several key libraries:

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| Open3D | Point cloud processing and visualization |
| KNN-CUDA | GPU-accelerated k-nearest neighbor search |
| FAISS | Efficient similarity search for memory bank |
| scikit-learn | Evaluation metrics and PCA |
| tqdm | Progress bars |
| pandas | Results logging |
| click | CLI argument parsing |

---

## References

1. **GeoTransformer**: Point cloud registration with geometric transformers
2. **RIConv++**: Rotation-invariant point cloud convolution
3. **PatchCore**: Memory-efficient anomaly detection
4. **Real3D-AD**: Industrial 3D anomaly detection benchmark
5. **Anomaly-ShapeNet**: Synthetic 3D anomaly dataset

For more details, see the [original paper](https://arxiv.org/abs/2510.16865).
