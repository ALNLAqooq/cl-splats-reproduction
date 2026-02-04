# 提升模块 / Lifter Module

> **复现项目说明 / Reproduction Notice**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats
> 
> This is a reproduction of CL-Splats. For the original project, see: https://github.com/jan-ackermann/cl-splats

---

[中文](#中文) | [English](#english)

---

# 中文

2D→3D 提升模块，通过多视角投票将 2D 变化掩码提升为 3D 活跃高斯掩码。

## 概述

Lifter 模块是连接 2D 变化检测和 3D 局部优化的桥梁：

1. 将 3D 点投影到每个相机视角
2. 检查投影是否落在 2D 变化掩码内
3. 通过多数投票确定哪些 3D 点属于变化区域

## 架构

```
lifter/
├── base_lifter.py           # 抽象基类
├── majority_vote_lifter.py  # 多数投票实现
└── __init__.py
```

## 使用方法

```python
from clsplats.lifter import MajorityVoteLifter
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'vote_threshold': 2  # 最少需要在多少个视角中可见
})

lifter = MajorityVoteLifter(cfg)

# 将 2D 掩码提升到 3D
active_mask = lifter.lift(
    points_3d=gaussians_xyz,      # [N, 3] 3D 点坐标
    change_masks_2d=change_masks, # List[Tensor] 每个视角的 2D 掩码
    cameras=cameras,              # List[Camera] 相机列表
    depth_maps=depth_maps         # Optional: 深度图用于遮挡检测
)
# 输出: [N] bool tensor，True 表示该点在变化区域
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vote_threshold` | 2 | 点必须在多少个视角中可见才被认为有效 |

## 工作原理

1. **投影**：将每个 3D 点投影到每个相机的 2D 平面
2. **掩码检查**：检查投影点是否落在变化掩码内
3. **遮挡检测**（可选）：使用深度图过滤被遮挡的点
4. **投票**：统计每个点在多少个视角中被标记为变化
5. **阈值化**：保留投票数 ≥ threshold 的点

## 遮挡处理

当提供深度图时，lifter 会进行遮挡检测：

```python
# 如果点的深度 > 表面深度 + margin，则认为被遮挡
is_occluded = point_depth > (surface_depth + 0.2)
```

这可以防止将被遮挡的点错误地标记为变化。

## 阈值选择

- **threshold=1**：宽松，可能包含遮挡点
- **threshold=2**：平衡，推荐用于大多数场景
- **threshold≥3**：严格，适合有很多视角的场景

---

# English

2D→3D lifting module that lifts 2D change masks to 3D active Gaussian masks through multi-view voting.

## Overview

The Lifter module bridges 2D change detection and 3D local optimization:

1. Project 3D points to each camera view
2. Check if projections fall within 2D change masks
3. Use majority voting to determine which 3D points belong to changed regions

## Architecture

```
lifter/
├── base_lifter.py           # Abstract base class
├── majority_vote_lifter.py  # Majority vote implementation
└── __init__.py
```

## Usage

```python
from clsplats.lifter import MajorityVoteLifter
import omegaconf

cfg = omegaconf.OmegaConf.create({
    'vote_threshold': 2  # Minimum views required
})

lifter = MajorityVoteLifter(cfg)

# Lift 2D masks to 3D
active_mask = lifter.lift(
    points_3d=gaussians_xyz,      # [N, 3] 3D point coordinates
    change_masks_2d=change_masks, # List[Tensor] 2D masks per view
    cameras=cameras,              # List[Camera] camera list
    depth_maps=depth_maps         # Optional: depth maps for occlusion
)
# Output: [N] bool tensor, True indicates point in changed region
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vote_threshold` | 2 | Minimum views a point must be visible in to be valid |

## How It Works

1. **Projection**: Project each 3D point to each camera's 2D plane
2. **Mask Check**: Check if projected point falls within change mask
3. **Occlusion Detection** (optional): Use depth maps to filter occluded points
4. **Voting**: Count how many views mark each point as changed
5. **Thresholding**: Keep points with votes ≥ threshold

## Occlusion Handling

When depth maps are provided, the lifter performs occlusion detection:

```python
# If point depth > surface depth + margin, consider it occluded
is_occluded = point_depth > (surface_depth + 0.2)
```

This prevents incorrectly marking occluded points as changed.

## Threshold Selection

- **threshold=1**: Lenient, may include occluded points
- **threshold=2**: Balanced, recommended for most scenes
- **threshold≥3**: Strict, suitable for scenes with many views
