# CL-Splats 中文文档

> **复现项目说明**
> 
> 本项目是对 CL-Splats 的复现工作。原始项目请参见：https://github.com/jan-ackermann/cl-splats

---

欢迎使用 CL-Splats！这是一个基于 3D Gaussian Splatting 的持续学习框架，能够通过少量新图像高效更新已有的 3DGS 场景表示。

## 📚 文档目录

### 核心模块文档

| 模块 | 说明 | 状态 |
|------|------|------|
| [变化检测](../clsplats/change_detection/README.md) | 基于 DINOv2 的语义变化检测 | ✅ 已完成 |
| [采样模块](../clsplats/sampling/README.md) | 全场景/区域采样策略 | ✅ 已完成 |
| [数据集模块](../clsplats/dataset/README.md) | COLMAP 数据读取与预处理 | ✅ 已完成 |
| [剪枝模块](../clsplats/pruning/README.md) | 球体边界约束剪枝 | ✅ 已完成 |
| [提升模块](../clsplats/lifter/README.md) | 2D→3D 多视角投票提升 | ✅ 已完成 |

### 开发文档

- [使用教程](USAGE_GUIDE.md) - 完整的安装、数据准备、训练流程

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/ALNLAqooq/cl-splats-reproduction.git
cd cl-splats-reproduction

# 创建 conda 环境
conda env create -f environment.yml          # Linux
conda env create -f environment_windows.yml  # Windows

conda activate cl-splats-dev

# 编译 CUDA 扩展
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
cd ../simple-knn
pip install -e . --no-build-isolation
```

### 2. 数据准备

将多时刻图像按以下结构组织：

```
your_dataset/
├── t1/           # 初始场景图像
│   ├── img_001.png
│   └── ...
└── t2/           # 变化后的场景图像
    ├── img_101.png
    └── ...
```

运行预处理脚本获取相机位姿：

```bash
python clsplats/utils/preprocessing.py --input_dir your_dataset
```

### 3. 训练

```bash
python -m clsplats.train dataset.path=your_dataset/output
```

## 🔧 核心流程

CL-Splats 的增量更新流程：

1. **变化检测** - 使用 DINOv2 比较渲染图像与新观测图像，生成 2D 变化掩码
2. **2D→3D 提升** - 通过多视角投票将 2D 掩码提升为 3D 活跃高斯掩码
3. **局部优化** - 仅优化活跃高斯，冻结不变区域
4. **球体剪枝** - 约束活跃高斯在包围球内，防止漂移

## ⚙️ 主要配置

配置文件位于 `configs/cl-splats.yaml`：

```yaml
train:
  iterations: 30000              # t0 完整训练迭代数
  incremental_iterations: 16000  # t>0 增量训练迭代数
  
change_detection:
  threshold: 0.2                 # 变化检测阈值（越小越敏感）
  dilate_kernel_size: 13         # 掩码膨胀核大小

model:
  optimizer_type: sparse_adam    # 使用稀疏优化器实现局部优化
```

## 📊 性能参考

| 指标 | 数值 |
|------|------|
| 变化检测速度 | ~30ms/图像对 |
| 初始重建 (100张图) | ~10分钟 |
| 增量更新 (25张新图) | ~5分钟 |
| 渲染速度 | 220+ FPS |

## 💡 常见问题

**Q: 需要什么硬件？**

最低要求：NVIDIA GPU (推荐 RTX 3090+)，16GB RAM

**Q: 如何提高重建质量？**

- 增加图像数量和视角覆盖
- 调低 `change_detection.threshold`（更敏感）
- 增加训练迭代次数

**Q: 支持哪些场景？**

✅ 室内/室外静态场景、物体级场景  
⚠️ 大规模室外场景（性能可能下降）  
❌ 动态场景

## 📄 引用

```bibtex
@inproceedings{ackermann2025clsplats,
    author={Ackermann, Jan and Kulhanek, Jonas and Cai, Shengqu and Haofei, Xu and Pollefeys, Marc and Wetzstein, Gordon and Guibas, Leonidas and Peng, Songyou},
    title={CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## 🔗 链接

- [原始项目](https://github.com/jan-ackermann/cl-splats)
- [项目主页](https://cl-splats.github.io/)
- [论文](https://arxiv.org/abs/2506.21117)

---

**最后更新**：2026年2月
