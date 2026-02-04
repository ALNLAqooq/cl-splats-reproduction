# CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

### ICCV 2025

[![Website](https://img.shields.io/badge/CL--Splats-%F0%9F%8C%90Website-purple?style=flat)](https://cl-splats.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.21117-b31b1b.svg)](https://arxiv.org/abs/2506.21117) [![Original Repo](https://img.shields.io/badge/Original-Repo-blue?style=flat)](https://github.com/jan-ackermann/cl-splats)

[Jan Ackermann](https://janackermann.info)
[Jonas Kulhanek](https://jkulhanek.com)
[Shengqu Cai](https://primecai.github.io)
[Haofei Xu](https://haofeixu.github.io)
[Marc Pollefeys](https://people.inf.ethz.ch/marc.pollefeys/)
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)
[Songyou Peng](https://pengsongyou.github.io)

![CL-Splats Teaser Graphic](assets/cl-splats-teaser.png)

*TL;DR*: CL-Splats optimizes existing 3DGS scene representations with a small set of images showing the changed region.

---

## 🔄 Reproduction Project / 复现项目

> **This is a reproduction of CL-Splats.**  
> For the original project, see: https://github.com/jan-ackermann/cl-splats
>
> **本项目是对 CL-Splats 的复现工作。**  
> 原始项目请参见：https://github.com/jan-ackermann/cl-splats

📖 [中文文档 / Chinese Documentation](docs/README_CN.md)

---

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Core Pipeline](#core-pipeline)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Known Issues](#known-issues)
- [Citation](#citation)

---

## Install

### Pre-requisites

- NVIDIA GPU with CUDA support (RTX 3090+ recommended)
- COLMAP for camera pose estimation
- CUDA Development Kit

### Environment Setup

```bash
# Clone repository
git clone https://github.com/ALNLAqooq/cl-splats-reproduction.git
cd cl-splats-reproduction

# Create conda environment
conda env create -f environment.yml          # Linux
conda env create -f environment_windows.yml  # Windows

conda activate cl-splats-dev

# Compile CUDA extensions
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ../simple-knn
pip install -e . --no-build-isolation
```

> **Note:** Ensure PyTorch CUDA version matches your system CUDA version.

---

## Quick Start

### 1. Prepare Data

Organize images by timestep:

```
your_dataset/
├── t1/           # Initial scene
│   ├── img_001.png
│   └── ...
└── t2/           # After scene change
    ├── img_101.png
    └── ...
```

### 2. Preprocess (Camera Pose Estimation)

```bash
python clsplats/utils/preprocessing.py --input_dir your_dataset
```

### 3. Train

```bash
python -m clsplats.train dataset.path=your_dataset/output
```

### 4. View Results

Open `outputs/point_cloud_final.ply` with MeshLab or CloudCompare.

---

## Core Pipeline

CL-Splats incremental update pipeline:

1. **Change Detection** - DINOv2-based semantic comparison, generates 2D change masks
2. **2D→3D Lifting** - Multi-view voting to identify active 3D Gaussians
3. **Local Optimization** - Only optimize active Gaussians, freeze unchanged regions
4. **Sphere Pruning** - Constrain active Gaussians within bounding spheres

---

## Module Documentation

| Module | Description | Documentation |
|--------|-------------|---------------|
| Change Detection | DINOv2-based semantic change detection | [README](clsplats/change_detection/README.md) |
| Sampling | GMM-based point sampling | [README](clsplats/sampling/README.md) |
| Dataset | COLMAP data loading | [README](clsplats/dataset/README.md) |
| Pruning | Sphere boundary pruning | [README](clsplats/pruning/README.md) |
| Lifter | 2D→3D multi-view voting | [README](clsplats/lifter/README.md) |

📖 Full usage guide: [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)

---

## Configuration

Main config file: `configs/cl-splats.yaml`

```yaml
train:
  iterations: 30000              # t0 full training
  incremental_iterations: 16000  # t>0 incremental training
  
change_detection:
  threshold: 0.2                 # Lower = more sensitive
  dilate_kernel_size: 13

model:
  optimizer_type: sparse_adam    # Local optimization
```

---

## Known Issues

⚠️ **The basic functionality is implemented, but the reproduction results are not satisfactory.**

The core modules (change detection, sampling, pruning, local optimization) are functional, but the overall reconstruction quality does not match the original paper's results. 

For detailed issue analysis and discussion, see:
- [Issue Report (English)](docs/ISSUE_REPORT_EN.md)
- [问题报告 (中文)](docs/ISSUE_REPORT_FOR_AUTHOR.md)

---

## Citation

```bibtex
@inproceedings{ackermann2025clsplats,
    author={Ackermann, Jan and Kulhanek, Jonas and Cai, Shengqu and Haofei, Xu and Pollefeys, Marc and Wetzstein, Gordon and Guibas, Leonidas and Peng, Songyou},
    title={CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

---

## Links

- [Original Project](https://github.com/jan-ackermann/cl-splats)
- [Project Page](https://cl-splats.github.io/)
- [Paper](https://arxiv.org/abs/2506.21117)
- [3DGS Original Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
