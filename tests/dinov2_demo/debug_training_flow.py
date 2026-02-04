"""
Debug script to understand the training flow and change detection.

This script simulates exactly what happens in trainer.py:
1. Load t0 checkpoint
2. Load t1 cameras and images
3. Render from t1 cameras using t0 model
4. Detect changes between rendered and observed
5. Visualize the masks
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# ============================================================
# 配置
# ============================================================
CHECKPOINT_PATH = r"C:\3dgsprojects\cl-splats-main\outputs\checkpoint_t0.pt"
DATASET_PATH = r"C:\3dgsprojects\cl-splats-main\cl-splats-dataset\test1\output"
OUTPUT_DIR = os.path.dirname(__file__)

# ============================================================


def main():
    print("=" * 60)
    print("Debug Training Flow - Change Detection")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load dataset (same as trainer)
    print("\n1. Loading dataset...")
    from clsplats.dataset import CLSplatsDataset
    dataset = CLSplatsDataset(
        path=DATASET_PATH,
        resolution_scale=1.0,
        white_background=False,
        eval_mode=False,
        device="cuda"
    )
    print(f"   Timesteps: {dataset.get_num_timesteps()}")
    
    # 2. Load Gaussian model from checkpoint
    print("\n2. Loading Gaussian model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    ply_path = checkpoint.get("ply_path")
    
    from clsplats.representation.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    print(f"   Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # 3. Get t1 cameras and images (same as trainer._load_timestep_data)
    print("\n3. Loading t1 cameras and images...")
    t1_cameras = dataset.get_cameras(1)
    t1_images = dataset.get_images(1)
    print(f"   {len(t1_cameras)} cameras, {len(t1_images)} images")
    
    # 4. Render from t1 cameras using t0 model (same as trainer._render_from_cameras)
    print("\n4. Rendering from t1 cameras using t0 model...")
    from clsplats.rendering import render
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    rendered_images = []
    
    with torch.no_grad():
        for camera in t1_cameras:
            result = render(camera, gaussians, bg_color)
            rendered_images.append(result["render"])
    
    print(f"   Rendered {len(rendered_images)} images")
    
    # 5. Detect changes (same as trainer._detect_changes)
    print("\n5. Detecting changes...")
    from clsplats.change_detection.dinov2_detector import DinoV2Detector
    
    cfg = OmegaConf.create({
        'threshold': 0.2,
        'dilate_mask': True,
        'dilate_kernel_size': 13,
        'upsample': True
    })
    detector = DinoV2Detector(cfg)
    
    change_masks = []
    for i, (rendered, observed) in enumerate(zip(rendered_images, t1_images)):
        # Images are [C, H, W], detector expects [H, W, C]
        rendered_hwc = rendered.permute(1, 2, 0)
        observed_hwc = observed.permute(1, 2, 0)
        mask = detector.predict_change_mask(rendered_hwc, observed_hwc)
        change_masks.append(mask)
        
        if i < 3:
            changed = mask.sum().item()
            total = mask.numel()
            print(f"   View {i}: {changed}/{total} pixels changed ({100*changed/total:.2f}%)")
    
    # 6. Visualize a few views
    print("\n6. Creating visualizations...")
    
    view_indices = [0, len(t1_cameras) // 2, len(t1_cameras) - 1]
    
    for idx in view_indices:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        rendered_np = rendered_images[idx].permute(1, 2, 0).cpu().numpy()
        observed_np = t1_images[idx].permute(1, 2, 0).cpu().numpy()
        mask_np = change_masks[idx].cpu().numpy().astype(np.float32)
        
        # Resize mask if needed
        if mask_np.shape != observed_np.shape[:2]:
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((observed_np.shape[1], observed_np.shape[0]), Image.NEAREST)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        
        axes[0, 0].imshow(np.clip(rendered_np, 0, 1))
        axes[0, 0].set_title(f'Rendered (t0 model, view {idx})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.clip(observed_np, 0, 1))
        axes[0, 1].set_title(f'Observed (t1, view {idx})')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(mask_np, cmap='hot')
        changed_pct = 100 * mask_np.sum() / mask_np.size
        axes[1, 0].set_title(f'Change Mask ({changed_pct:.1f}% changed)')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = observed_np.copy()
        overlay[mask_np > 0.5] = [1, 0, 0]
        axes[1, 1].imshow(np.clip(overlay, 0, 1))
        axes[1, 1].set_title('Changes (red) on Observed')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'training_flow_view{idx}.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"   Saved: {output_path}")
    
    # 7. Check the mask that would be saved in visualization
    print("\n7. Checking mask format...")
    mask = change_masks[0]
    print(f"   Mask shape: {mask.shape}")
    print(f"   Mask dtype: {mask.dtype}")
    print(f"   Mask device: {mask.device}")
    print(f"   Mask min/max: {mask.min().item()}, {mask.max().item()}")
    print(f"   Mask unique values: {torch.unique(mask)}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
