# CL-Splats 实现问题报告

## 概述

我基于 CL-Splats 论文和已发布的代码框架实现了完整的增量学习训练流程。在实际测试中遇到了两个主要问题：

1. **灾难性遗忘**：在 t1 时刻增量训练时，t0 时刻的场景区域出现明显退化
2. **重建效率不高**：训练收敛较慢，最终质量不如预期

本文档详细描述我的实现方式、与论文的对比、以及可能的问题原因，希望能得到作者的指导。

---

## 问题 1：灾难性遗忘

### 现象描述

在 t1 时刻进行增量训练时，即使使用了 Local Optimization 机制（只更新 active Gaussians），t0 时刻的场景区域仍然出现明显的质量退化：
- 原本清晰的区域变得模糊
- 颜色出现偏差
- 几何结构发生变形

**Dataset 1:**

| T0 训练完成（正常） | T1 训练后（退化） |
|:--:|:--:|
| ![T0 Fine](../docs/pics/dataset1_t0_fine.png) | ![T1 Bad](../docs/pics/dataset1_t1_bad.png) |

**Dataset 2:**

| T0 训练完成（正常） | T1 训练后（退化） |
|:--:|:--:|
| ![T0 Fine](../docs/pics/dataset2_t0_fine.png) | ![T1 Bad](../docs/pics/dataset2_t1_bad.png) |

### 我的实现方式

#### 1. Change Detection (变化检测)

使用 DINOv2 特征进行变化检测：

```python
# clsplats/change_detection/dinov2_detector.py
class DinoV2Detector:
    def predict_change_mask(self, rendered_image, observation):
        # 1. 提取 DINOv2 特征
        rendered_feats = self.model.get_intermediate_layers(rendered, reshape=True)
        observed_feats = self.model.get_intermediate_layers(observed, reshape=True)
        
        # 2. 计算余弦相似度
        cos_sim = self.cos(rendered_feats, observed_feats)
        
        # 3. 阈值化得到变化 mask
        mask = cos_sim < self.cfg.threshold  # 默认 threshold=0.2
        
        # 4. 膨胀处理
        if self.cfg.dilate_mask:
            mask = self._dilate_mask(mask, kernel_size=13)
```

**配置参数** (`configs/change_detection/dinov2.yaml`):
```yaml
threshold: 0.2
dilate_mask: true
dilate_kernel_size: 13
upsample: true
```

#### 2. 2D→3D Lifting (Majority Vote)

将 2D 变化 mask 提升到 3D 空间：

```python
# clsplats/lifter/majority_vote_lifter.py
class MajorityVoteLifter:
    def lift(self, points_3d, change_masks_2d, cameras, depth_maps):
        vote_counts = torch.zeros(num_points)
        
        for mask_2d, camera in zip(change_masks_2d, cameras):
            # 1. 投影 3D 点到 2D
            projected_2d, point_depths = self._project_points(points_3d, camera)
            
            # 2. 检查是否在 mask 内
            in_mask = mask_2d[y, x]
            
            # 3. 遮挡检测（使用深度图）
            if depth_maps is not None:
                is_occluded = pt_depth > (surface_depth + margin)
                should_vote = in_mask & (~is_occluded)
                vote_counts[valid_idx] += should_vote.int()
            else:
                vote_counts[valid_idx] += in_mask.int()
        
        return vote_counts >= self.threshold  # 默认 threshold=5
```

**配置参数** (`configs/lifter/majority_vote.yaml`):
```yaml
vote_threshold: 5  # 至少在 5 个视角中被标记为变化
```

#### 3. Local Optimization (局部优化)

在训练循环中，只更新 active Gaussians：

```python
# clsplats/trainer.py - _train_step()
def _train_step(self, iteration, camera, gt_image):
    # 1. 渲染（传入 active_gaussian_mask）
    render_result = render(
        camera, self.gaussians, self.bg_color,
        active_gaussian_mask=self.active_gaussians_mask,
    )
    
    # 2. 计算 loss 并反向传播
    loss = combined_loss(rendered_image, gt_image, lambda_dssim)
    loss.backward()
    
    with torch.no_grad():
        # 3. 关键：将 inactive Gaussians 的梯度置零
        if self.active_gaussians_mask is not None and self.timestep > 0:
            self.gaussians.zero_gradients_for_inactive(self.active_gaussians_mask)
        
        # 4. Densification（只对 active Gaussians）
        if iteration < self.training_args.densify_until_iter:
            self.gaussians.add_densification_stats(
                viewspace_points, visibility_filter,
                active_mask=self.active_gaussians_mask if self.timestep > 0 else None
            )
            
            if iteration % densification_interval == 0:
                # 传入 active_mask，只 densify/prune active Gaussians
                updated_mask = self.gaussians.densify_and_prune(
                    ...,
                    active_mask=active_mask_for_prune
                )
        
        # 5. Opacity reset（只对 active Gaussians）
        if iteration % opacity_reset_interval == 0:
            if self.timestep > 0 and self.active_gaussians_mask is not None:
                self.gaussians.reset_opacity(self.active_gaussians_mask)
        
        # 6. 冻结 inactive Gaussians 的 optimizer momentum
        if self.active_gaussians_mask is not None and self.timestep > 0:
            self.gaussians.freeze_inactive_gaussians(self.active_gaussians_mask)
        
        # 7. SparseGaussianAdam 设置 visible mask
        if hasattr(self.gaussians.optimizer, 'set_visible_mask'):
            self.gaussians.optimizer.set_visible_mask(self.active_gaussians_mask)
        
        # 8. Optimizer step
        self.gaussians.optimizer.step()
```

#### 4. GaussianModel 中的保护机制

```python
# clsplats/representation/gaussian_model.py

def zero_gradients_for_inactive(self, active_mask):
    """将 inactive Gaussians 的梯度置零"""
    inactive_mask = ~active_mask
    if self._xyz.grad is not None:
        self._xyz.grad[inactive_mask] = 0.0
    # ... 对所有参数执行相同操作

def freeze_inactive_gaussians(self, active_mask):
    """冻结 inactive Gaussians 的 optimizer momentum"""
    inactive_mask = ~active_mask
    for group in self.optimizer.param_groups:
        stored_state = self.optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state['exp_avg'][inactive_mask] = 0.0
            stored_state['exp_avg_sq'][inactive_mask] = 0.0

def densify_and_clone(self, grads, grad_threshold, scene_extent, active_mask=None):
    """只 clone active Gaussians"""
    if active_mask is not None:
        selected_pts_mask = torch.logical_and(selected_pts_mask, active_mask)
    # ...

def densify_and_split(self, grads, grad_threshold, scene_extent, active_mask=None):
    """只 split active Gaussians"""
    if active_mask is not None:
        selected_pts_mask = torch.logical_and(selected_pts_mask, active_mask)
    # ...

def reset_opacity(self, active_mask=None):
    """只 reset active Gaussians 的 opacity"""
    if active_mask is not None:
        opacities_new = self._opacity.clone()
        opacities_new[active_mask] = reset_value
        # ...
```

### Change Detection 实现差异

对比作者已发布的 `dinov2_detector.py`，发现处理顺序不同：

**作者代码**：
```python
# 先阈值化，再膨胀，最后上采样
cos_sim = self.cos(rendered_feats[0], observed_feats[0])  # 注意：取 [0]
mask = cos_sim < self.cfg.threshold
if dilate_mask:
    mask = self._dilate_mask(mask, ...)
if upsample:
    mask = interpolate(mask, ...)  # nearest 插值
```

**我的代码**：
```python
# 先上采样 cos_sim 值，再阈值化，最后膨胀
cos_sim = self.cos(rendered_feats, observed_feats)  # 不取 [0]
if upsample:
    cos_sim = interpolate(cos_sim, ...)  # bilinear 插值
mask = cos_sim < self.cfg.threshold
if dilate_mask:
    mask = self._dilate_mask(mask, ...)
```

**差异点**：
- 作者对 `rendered_feats[0]` 取索引，我直接使用整个 tensor
- 作者先阈值化再上采样（nearest），我先上采样（bilinear）再阈值化
- 这可能导致 mask 边界精度不同

### 与论文的对比

| 论文描述 | 我的实现 | 可能的差异 |
|---------|---------|-----------|
| Local Optimization: 只更新变化区域的 Gaussians | ✅ 通过 `active_gaussians_mask` 实现 | 实现方式可能不同 |
| 使用 tile mask 加速渲染 | ⚠️ 代码中有 `tile_mask` 参数但未完全实现 | 可能影响梯度计算 |
| Sphere Pruning: 防止 Gaussians 逃逸 | ✅ 实现了 `SpherePruner` | 参数可能需要调整 |
| SparseGaussianAdam 优化器 | ✅ 使用了 `SparseGaussianAdam` | 需要确认 CUDA kernel 是否正确 |

### 可能的原因分析

1. **Change Detection 阈值问题**
   - `threshold=0.2` 可能太低，导致过多区域被标记为变化
   - 膨胀核大小 `kernel_size=13` 可能不够，变化区域边界不准确

2. **Majority Vote 阈值问题**
   - `vote_threshold=5` 可能太高，导致真正变化的 Gaussians 被漏掉
   - 或者太低，导致不该变化的 Gaussians 被错误标记

3. **梯度泄漏**
   - 虽然对 inactive Gaussians 置零了梯度，但渲染过程中可能仍有间接影响
   - `tile_mask` 未完全实现，可能导致 inactive Gaussians 参与了渲染计算

4. **Optimizer Momentum 问题**
   - 每次 timestep 开始时重新 `setup_training()`，但之前的 momentum 可能有残留
   - `freeze_inactive_gaussians()` 可能调用时机不对

5. **Densification 同步问题**
   - `active_gaussians_mask` 在 densification 后需要同步更新
   - 新增的 Gaussians 应该继承 active 状态

---

## 问题 2：重建效率不高

### 现象描述

- 训练收敛较慢
- 最终 PSNR/SSIM 指标不如预期
- 变化区域的重建质量不够好

### 我的实现方式

#### 训练配置

```yaml
# configs/cl-splats.yaml
train:
  iterations: 30000           # T0 全场景训练
  incremental_iterations: 16000  # T1+ 增量训练
  
  # Learning rates
  lr_position: 0.00016
  lr_feature: 0.0025
  lr_opacity: 0.05
  lr_scaling: 0.005
  lr_rotation: 0.001
  
  # Densification
  densification_interval: 100
  densify_from_iter: 500
  densify_until_iter: 15000
  densify_grad_threshold: 0.0002
```

#### Sampling 模块

```python
# clsplats/sampling/gaussian_sampler.py
# 使用 GMM 在变化区域采样新点
class GaussianSampler:
    def sample(self, change_masks, cameras, depth_maps):
        # 1. 从变化区域反投影得到 3D 点
        # 2. 使用 K-Means 聚类
        # 3. 拟合 GMM
        # 4. 从 GMM 采样新点
```

### 可能的原因分析

1. **增量训练迭代次数**
   - `incremental_iterations=16000` 可能不够
   - 论文中是否有推荐的迭代次数？

2. **学习率设置**
   - 增量训练时是否应该使用不同的学习率？
   - Position learning rate 的 decay 策略是否正确？

3. **Sampling 策略**
   - 新采样的点数量是否足够？
   - GMM 参数是否合适？

4. **缺少 History Recovery**
   - 论文中提到的 History Recovery 模块尚未实现
   - 这可能是影响质量的关键因素

---

## 代码结构参考

```
clsplats/
├── change_detection/
│   └── dinov2_detector.py      # DINOv2 变化检测
├── lifter/
│   └── majority_vote_lifter.py # 2D→3D 提升
├── sampling/
│   └── gaussian_sampler.py     # GMM 采样
├── pruning/
│   └── sphere_pruner.py        # 球形边界剪枝
├── representation/
│   └── gaussian_model.py       # Gaussian 模型（含 Local Optimization 逻辑）
├── rendering/
│   └── renderer.py             # 渲染器
└── trainer.py                  # 训练主循环
```

---

## 问题总结

1. **灾难性遗忘**：最关键的问题，需要确认 Local Optimization 的实现是否正确
2. **重建效率**：可能与参数设置、缺少 History Recovery 有关

## 希望得到的帮助

1. Local Optimization 的正确实现方式是什么？特别是：
   - 如何正确使用 `tile_mask`？
   - `SparseGaussianAdam` 的 `set_visible_mask` 应该如何使用？
   - 梯度置零的时机是否正确？

2. 推荐的超参数设置是什么？
   - Change detection threshold
   - Majority vote threshold
   - 增量训练迭代次数

3. History Recovery 模块的实现细节？

---

## 附录：关键代码片段

### A. 训练主循环 (trainer.py)

```python
def train(self):
    # 每个 timestep 重新设置 optimizer
    self.setup_training(iterations=iterations)
    
    for iteration in range(iterations):
        # 随机选择视角
        idx = random.randint(0, num_views - 1)
        camera = self.current_cameras[idx]
        gt_image = self.current_images[idx]
        
        metrics = self._train_step(iteration, camera, gt_image)
```

### B. 渲染调用 (trainer.py)

```python
render_result = render(
    camera, 
    self.gaussians, 
    self.bg_color,
    active_gaussian_mask=self.active_gaussians_mask,  # Local Optimization
    tile_mask=tile_mask,  # 目前为 None
)
```

### C. Active Mask 计算 (trainer.py)

```python
def _compute_active_gaussians_mask(self):
    xyz = self.gaussians.get_xyz.detach()
    self.active_gaussians_mask = self.lifter.lift(
        points_3d=xyz,
        change_masks_2d=self.change_masks,
        cameras=self.current_cameras,
        depth_maps=self.current_depths
    )
```
