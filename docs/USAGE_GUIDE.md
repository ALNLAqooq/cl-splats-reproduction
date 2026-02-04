# CL-Splats 使用教程 / Usage Guide

[中文](#中文) | [English](#english)

---

# 中文

本教程将指导你完成从数据准备到模型训练的完整流程。

## 目录

1. [环境安装](#环境安装)
2. [数据准备](#数据准备)
3. [预处理](#预处理相机位姿估计)
4. [训练](#训练)
5. [配置详解](#配置详解)
6. [进阶用法](#进阶用法)

---

## 环境安装

### 1. 创建 Conda 环境

```bash
# Linux
conda env create -f environment.yml

# Windows
conda env create -f environment_windows.yml

# 激活环境
conda activate cl-splats
```

### 2. 安装 COLMAP

COLMAP 用于预处理阶段的相机位姿估计。

**Ubuntu:**
```bash
sudo apt install colmap
```

**Windows:**
- 从 [COLMAP Releases](https://github.com/colmap/colmap/releases) 下载预编译版本
- 将 `colmap.exe` 所在目录添加到系统 PATH

**验证安装：**
```bash
colmap help
```

### 3. 编译 CUDA 扩展

```bash
# 进入子模块目录编译
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation

cd ../simple-knn
pip install -e . --no-build-isolation
```

> **注意**: 确保 PyTorch 的 CUDA 版本与系统 CUDA 版本一致。Windows 用户建议在 Visual Studio Developer Command Prompt 中运行。

---

## 数据准备

### 输入目录结构

将原始图片按时刻组织：

```
your_dataset/
├── t1/                    # 时刻 1（初始场景）
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── t2/                    # 时刻 2（场景变化后）
│   ├── image_101.png
│   └── ...
└── t3/                    # 时刻 3（可选）
    └── ...
```

### 要求

| 项目 | 要求 |
|------|------|
| 目录命名 | 必须是 `t1`, `t2`, `t3`, ... 格式 |
| 图片格式 | `.jpg`, `.jpeg`, `.png` |
| 图片内容 | 每个时刻多角度拍摄同一场景 |
| 相机 | 建议使用同一相机（共享内参） |

### 拍摄建议

1. **t1 时刻**：拍摄完整场景的多个角度（建议 50-200 张）
2. **后续时刻**：重点拍摄变化区域，但也需覆盖部分不变区域用于配准
3. **重叠率**：相邻图片之间保持 60%-80% 视角重叠
4. **避免**：运动模糊、过曝/欠曝、纯色表面

---

## 预处理（相机位姿估计）

使用 COLMAP 进行增量式相机位姿估计。

### 运行预处理脚本

```bash
python clsplats/utils/preprocessing.py --input_dir /path/to/your_dataset
```

### 处理流程

1. **t1 完整重建**：特征提取 → 特征匹配 → 稀疏重建
2. **t2, t3, ... 增量注册**：将新时刻图像注册到现有模型
3. **全局优化**：Bundle Adjustment
4. **图像去畸变**：输出去畸变后的图像
5. **生成多时刻结构**：为每个时刻创建独立目录

### 输出结构

```
your_dataset/
├── t1/, t2/, ...           # 原始数据（不变）
├── colmap_workspace/       # COLMAP 工作目录
└── output/                 # ← 用于训练的输出目录
    ├── t0/                 # 对应原始 t1
    │   ├── images/
    │   └── sparse/0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── t1/                 # 对应原始 t2
        ├── images/
        └── sparse/0/
```

> **注意**: 输入使用 `t1, t2, ...`（从 1 开始），输出使用 `t0, t1, ...`（从 0 开始）。

---

## 训练

### 基本用法

```bash
python -m clsplats.train dataset.path=/path/to/your_dataset/output
```

### 训练流程

1. **t0 训练**：从点云初始化，完整训练 3D Gaussian 模型（30000 迭代）
2. **t1, t2, ... 增量训练**：
   - 使用 DINOv2 检测场景变化区域
   - 仅对变化区域进行局部优化（16000 迭代）
   - 保持不变区域的 Gaussian 参数固定

### 常用参数

```bash
python -m clsplats.train \
    dataset.path=/path/to/data/output \
    train.iterations=30000 \
    train.incremental_iterations=16000 \
    output_dir=outputs/my_experiment \
    change_detection.threshold=0.2
```

### 从检查点恢复

```bash
python -m clsplats.train \
    dataset.path=/path/to/data/output \
    resume_from=outputs/checkpoint_t1.pt
```

### 训练输出

```
outputs/
├── checkpoint_t0.pt       # 时刻 0 检查点
├── checkpoint_t1.pt       # 时刻 1 检查点
├── checkpoint_final.pt    # 最终检查点
├── point_cloud_final.ply  # 最终点云（可用于可视化）
├── train.log              # 训练日志
└── visualizations/        # 可视化图像/GIF
    └── t1/
        └── training_view0.gif
```

---

## 配置详解

完整配置文件：`configs/cl-splats.yaml`

### 训练参数

```yaml
train:
  iterations: 30000              # t0 完整训练迭代数
  incremental_iterations: 16000  # t>0 增量训练迭代数
  
  # 学习率
  lr_position: 0.00016
  lr_feature: 0.0025
  lr_opacity: 0.05
  lr_scaling: 0.005
  lr_rotation: 0.001
  
  # 密集化
  densify_from_iter: 500
  densify_until_iter: 15000
  densification_interval: 100
  
  # 损失权重
  lambda_dssim: 0.2
  
  # 可视化（仅 t>0 生效）
  vis_interval: 1000             # 保存可视化间隔（0 禁用）
  vis_generate_gif: true         # 训练结束生成 GIF
```

### 变化检测参数

```yaml
change_detection:
  threshold: 0.2           # 变化检测阈值（0-1，越小越敏感）
  dilate_mask: true        # 是否膨胀掩码
  dilate_kernel_size: 13   # 膨胀核大小
```

### 模型参数

```yaml
model:
  sh_degree: 3                   # 球谐函数阶数
  optimizer_type: sparse_adam    # 稀疏优化器（局部优化）
```

### 剪枝参数

```yaml
pruning:
  min_cluster_size: 1000   # HDBSCAN 最小聚类大小
  quantile: 0.98           # 半径百分位数
  radius_multiplier: 1.1   # 安全边界因子
```

---

## 进阶用法

### 自定义配置文件

```bash
# 复制并编辑配置
cp configs/cl-splats.yaml configs/my-config.yaml

# 使用自定义配置
python -m clsplats.train --config-name=my-config dataset.path=/path/to/data
```

### 仅运行变化检测

```python
from clsplats.change_detection import DinoV2Detector
import omegaconf

cfg = omegaconf.OmegaConf.create({
    "threshold": 0.2, 
    "dilate_mask": True, 
    "dilate_kernel_size": 13
})
detector = DinoV2Detector(cfg)

# 比较两张图片 (H, W, 3) 格式的 torch.Tensor
mask = detector.predict_change_mask(rendered_image, observed_image)
# mask: (H, W) bool tensor, True 表示变化区域
```

### 可视化点云

训练完成后，使用以下工具查看 `.ply` 文件：
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.cloudcompare.org/)
- [Blender](https://www.blender.org/)

---

## 完整工作流程示例

```bash
# 1. 准备数据
mkdir -p my_scene/t1 my_scene/t2
# 将初始场景图片放入 t1/
# 将变化后场景图片放入 t2/

# 2. 预处理
python clsplats/utils/preprocessing.py --input_dir my_scene

# 3. 训练
python -m clsplats.train \
    dataset.path=my_scene/output \
    output_dir=outputs/my_scene

# 4. 查看结果
# 使用 MeshLab 打开 outputs/my_scene/point_cloud_final.ply
```

---

## 常见问题

**Q: 重建失败，只有部分图像注册成功？**
- 增加 t1 时刻的图片数量
- 确保图像之间有足够的视角重叠

**Q: 增量注册 t2 失败？**
- 确保 t2 的图像包含足够的不变区域用于匹配

**Q: Windows 上 COLMAP 命令找不到？**
- 确保 COLMAP 已添加到系统 PATH

**Q: 训练时 CUDA 内存不足？**
- 降低 `dataset.resolution`（如 0.5）
- 减少 `train.iterations`

---

# English

This guide walks you through the complete workflow from data preparation to model training.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Preprocessing](#preprocessing-camera-pose-estimation)
4. [Training](#training-1)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)

---

## Installation

### 1. Create Conda Environment

```bash
# Linux
conda env create -f environment.yml

# Windows
conda env create -f environment_windows.yml

# Activate
conda activate cl-splats
```

### 2. Install COLMAP

COLMAP is used for camera pose estimation during preprocessing.

**Ubuntu:**
```bash
sudo apt install colmap
```

**Windows:**
- Download from [COLMAP Releases](https://github.com/colmap/colmap/releases)
- Add the directory containing `colmap.exe` to system PATH

**Verify installation:**
```bash
colmap help
```

### 3. Compile CUDA Extensions

```bash
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation

cd ../simple-knn
pip install -e . --no-build-isolation
```

> **Note**: Ensure PyTorch CUDA version matches your system CUDA version. Windows users should run in Visual Studio Developer Command Prompt.

---

## Data Preparation

### Input Directory Structure

Organize your images by timestep:

```
your_dataset/
├── t1/                    # Timestep 1 (initial scene)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── t2/                    # Timestep 2 (after scene change)
│   ├── image_101.png
│   └── ...
└── t3/                    # Timestep 3 (optional)
    └── ...
```

### Requirements

| Item | Requirement |
|------|-------------|
| Directory naming | Must be `t1`, `t2`, `t3`, ... format |
| Image format | `.jpg`, `.jpeg`, `.png` |
| Image content | Multi-view captures of the same scene per timestep |
| Camera | Same camera recommended (shared intrinsics) |

### Capture Tips

1. **t1**: Capture complete scene from multiple angles (50-200 images recommended)
2. **Later timesteps**: Focus on changed regions, but include some unchanged areas for registration
3. **Overlap**: Maintain 60%-80% view overlap between adjacent images
4. **Avoid**: Motion blur, over/under exposure, textureless surfaces

---

## Preprocessing (Camera Pose Estimation)

Use COLMAP for incremental camera pose estimation.

### Run Preprocessing Script

```bash
python clsplats/utils/preprocessing.py --input_dir /path/to/your_dataset
```

### Processing Steps

1. **t1 full reconstruction**: Feature extraction → Matching → Sparse reconstruction
2. **t2, t3, ... incremental registration**: Register new images to existing model
3. **Global optimization**: Bundle Adjustment
4. **Image undistortion**: Output undistorted images
5. **Generate multi-timestep structure**: Create separate directories per timestep

### Output Structure

```
your_dataset/
├── t1/, t2/, ...           # Original data (unchanged)
├── colmap_workspace/       # COLMAP workspace
└── output/                 # ← Training output directory
    ├── t0/                 # Corresponds to original t1
    │   ├── images/
    │   └── sparse/0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── t1/                 # Corresponds to original t2
        ├── images/
        └── sparse/0/
```

> **Note**: Input uses `t1, t2, ...` (starting from 1), output uses `t0, t1, ...` (starting from 0).

---

## Training

### Basic Usage

```bash
python -m clsplats.train dataset.path=/path/to/your_dataset/output
```

### Training Flow

1. **t0 training**: Initialize from point cloud, full 3D Gaussian training (30000 iterations)
2. **t1, t2, ... incremental training**:
   - Detect scene changes using DINOv2
   - Local optimization on changed regions only (16000 iterations)
   - Keep unchanged Gaussian parameters frozen

### Common Parameters

```bash
python -m clsplats.train \
    dataset.path=/path/to/data/output \
    train.iterations=30000 \
    train.incremental_iterations=16000 \
    output_dir=outputs/my_experiment \
    change_detection.threshold=0.2
```

### Resume from Checkpoint

```bash
python -m clsplats.train \
    dataset.path=/path/to/data/output \
    resume_from=outputs/checkpoint_t1.pt
```

### Training Output

```
outputs/
├── checkpoint_t0.pt       # Timestep 0 checkpoint
├── checkpoint_t1.pt       # Timestep 1 checkpoint
├── checkpoint_final.pt    # Final checkpoint
├── point_cloud_final.ply  # Final point cloud (for visualization)
├── train.log              # Training log
└── visualizations/        # Visualization images/GIFs
    └── t1/
        └── training_view0.gif
```

---

## Configuration

Full configuration file: `configs/cl-splats.yaml`

### Training Parameters

```yaml
train:
  iterations: 30000              # t0 full training iterations
  incremental_iterations: 16000  # t>0 incremental training iterations
  
  # Learning rates
  lr_position: 0.00016
  lr_feature: 0.0025
  lr_opacity: 0.05
  lr_scaling: 0.005
  lr_rotation: 0.001
  
  # Densification
  densify_from_iter: 500
  densify_until_iter: 15000
  densification_interval: 100
  
  # Loss weights
  lambda_dssim: 0.2
  
  # Visualization (only for t>0)
  vis_interval: 1000             # Save visualization interval (0 to disable)
  vis_generate_gif: true         # Generate GIF at end of each timestep
```

### Change Detection Parameters

```yaml
change_detection:
  threshold: 0.2           # Change detection threshold (0-1, lower = more sensitive)
  dilate_mask: true        # Whether to dilate mask
  dilate_kernel_size: 13   # Dilation kernel size
```

### Model Parameters

```yaml
model:
  sh_degree: 3                   # Spherical harmonics degree
  optimizer_type: sparse_adam    # Sparse optimizer (for local optimization)
```

### Pruning Parameters

```yaml
pruning:
  min_cluster_size: 1000   # HDBSCAN minimum cluster size
  quantile: 0.98           # Radius percentile
  radius_multiplier: 1.1   # Safety margin factor
```

---

## Advanced Usage

### Custom Configuration File

```bash
# Copy and edit config
cp configs/cl-splats.yaml configs/my-config.yaml

# Use custom config
python -m clsplats.train --config-name=my-config dataset.path=/path/to/data
```

### Run Change Detection Only

```python
from clsplats.change_detection import DinoV2Detector
import omegaconf

cfg = omegaconf.OmegaConf.create({
    "threshold": 0.2, 
    "dilate_mask": True, 
    "dilate_kernel_size": 13
})
detector = DinoV2Detector(cfg)

# Compare two images (H, W, 3) torch.Tensor format
mask = detector.predict_change_mask(rendered_image, observed_image)
# mask: (H, W) bool tensor, True indicates changed region
```

### Visualize Point Cloud

After training, view `.ply` files with:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.cloudcompare.org/)
- [Blender](https://www.blender.org/)

---

## Complete Workflow Example

```bash
# 1. Prepare data
mkdir -p my_scene/t1 my_scene/t2
# Put initial scene images in t1/
# Put changed scene images in t2/

# 2. Preprocess
python clsplats/utils/preprocessing.py --input_dir my_scene

# 3. Train
python -m clsplats.train \
    dataset.path=my_scene/output \
    output_dir=outputs/my_scene

# 4. View results
# Open outputs/my_scene/point_cloud_final.ply with MeshLab
```

---

## FAQ

**Q: Reconstruction failed, only some images registered?**
- Increase number of images in t1
- Ensure sufficient view overlap between images

**Q: Incremental registration of t2 failed?**
- Ensure t2 images contain enough unchanged regions for matching

**Q: COLMAP command not found on Windows?**
- Ensure COLMAP is added to system PATH

**Q: CUDA out of memory during training?**
- Lower `dataset.resolution` (e.g., 0.5)
- Reduce `train.iterations`

---

## References

- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP Documentation](https://colmap.github.io/)
- [DINOv2](https://github.com/facebookresearch/dinov2)
