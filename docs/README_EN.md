# CL-Splats Documentation

> **Reproduction Notice**
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

Welcome to CL-Splats! This is a continual learning framework based on 3D Gaussian Splatting that efficiently updates existing 3DGS scene representations with a small set of new images.

## 📚 Documentation Index

### Core Module Documentation

| Module | Description | Status |
|--------|-------------|--------|
| [Change Detection](../clsplats/change_detection/README.md) | DINOv2-based semantic change detection | ✅ Complete |
| [Sampling](../clsplats/sampling/README.md) | Full-scene/region sampling strategies | ✅ Complete |
| [Dataset](../clsplats/dataset/README.md) | COLMAP data reading and preprocessing | ✅ Complete |
| [Pruning](../clsplats/pruning/README.md) | Sphere boundary constraint pruning | ✅ Complete |
| [Lifter](../clsplats/lifter/README.md) | 2D→3D multi-view voting lifting | ✅ Complete |

### Development Documentation

- [Usage Guide](USAGE_GUIDE.md) - Complete installation, data preparation, and training workflow

## 🚀 Quick Start

### 1. Installation

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

### 2. Data Preparation

Organize multi-timestep images as follows:

```
your_dataset/
├── t1/           # Initial scene images
│   ├── img_001.png
│   └── ...
└── t2/           # Changed scene images
    ├── img_101.png
    └── ...
```

Run preprocessing script to obtain camera poses:

```bash
python clsplats/utils/preprocessing.py --input_dir your_dataset
```

### 3. Training

```bash
python -m clsplats.train dataset.path=your_dataset/output
```

## 🔧 Core Pipeline

CL-Splats incremental update pipeline:

1. **Change Detection** - Use DINOv2 to compare rendered images with new observations, generate 2D change masks
2. **2D→3D Lifting** - Lift 2D masks to 3D active Gaussian masks via multi-view voting
3. **Local Optimization** - Only optimize active Gaussians, freeze unchanged regions
4. **Sphere Pruning** - Constrain active Gaussians within bounding spheres to prevent drift

## ⚙️ Key Configuration

Configuration file located at `configs/cl-splats.yaml`:

```yaml
train:
  iterations: 30000              # t0 full training iterations
  incremental_iterations: 16000  # t>0 incremental training iterations
  
change_detection:
  threshold: 0.2                 # Change detection threshold (lower = more sensitive)
  dilate_kernel_size: 13         # Mask dilation kernel size

model:
  optimizer_type: sparse_adam    # Use sparse optimizer for local optimization
```

## 📊 Performance Reference

| Metric | Value |
|--------|-------|
| Change detection speed | ~30ms/image pair |
| Initial reconstruction (100 images) | ~10 minutes |
| Incremental update (25 new images) | ~5 minutes |
| Rendering speed | 220+ FPS |

## 💡 FAQ

**Q: What hardware is required?**

Minimum: NVIDIA GPU (RTX 3090+ recommended), 16GB RAM

**Q: How to improve reconstruction quality?**

- Increase number of images and view coverage
- Lower `change_detection.threshold` (more sensitive)
- Increase training iterations

**Q: What scenes are supported?**

✅ Indoor/outdoor static scenes, object-level scenes  
⚠️ Large-scale outdoor scenes (performance may degrade)  
❌ Dynamic scenes

## 📄 Citation

```bibtex
@inproceedings{ackermann2025clsplats,
    author={Ackermann, Jan and Kulhanek, Jonas and Cai, Shengqu and Haofei, Xu and Pollefeys, Marc and Wetzstein, Gordon and Guibas, Leonidas and Peng, Songyou},
    title={CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## 🔗 Links

- [Original Project](https://github.com/jan-ackermann/cl-splats)
- [Project Page](https://cl-splats.github.io/)
- [Paper](https://arxiv.org/abs/2506.21117)

---

**Last Updated**: February 2026
