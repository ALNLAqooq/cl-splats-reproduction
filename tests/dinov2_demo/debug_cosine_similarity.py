"""
Debug script to understand cosine similarity distribution in DINOv2 change detection.
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
import torchvision

# Config
OBSERVATION_IMAGE_PATH = r"C:\3dgsprojects\cl-splats-main\tests\dinov2_demo\day_1_0020.png"
SPARSE_PATH = r"C:\3dgsprojects\cl-splats-main\cl-splats-dataset\test1\output\t1\sparse\0"
CHECKPOINT_PATH = r"C:\3dgsprojects\cl-splats-main\outputs\checkpoint_t0.pt"

def main():
    print("=" * 60)
    print("Debug: Cosine Similarity Distribution")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DINOv2
    print("\n1. Loading DINOv2 model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    model.to(device)
    
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # Load images
    print("\n2. Loading images...")
    
    # Load rendered image from checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    ply_path = checkpoint.get("ply_path")
    
    from clsplats.representation.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    print(f"   Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Build camera and render
    from clsplats.dataset.colmap_reader import (
        read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
    )
    from clsplats.utils.graphics_utils import getWorld2View2, getProjectionMatrix
    
    cameras = read_intrinsics_binary(os.path.join(SPARSE_PATH, "cameras.bin"))
    images = read_extrinsics_binary(os.path.join(SPARSE_PATH, "images.bin"))
    
    image_name = os.path.basename(OBSERVATION_IMAGE_PATH)
    target_image = None
    for img in images.values():
        if img.name == image_name:
            target_image = img
            break
    
    camera_intrinsics = cameras[target_image.camera_id]
    
    # Load actual image to get dimensions
    obs_img = Image.open(OBSERVATION_IMAGE_PATH).convert('RGB')
    width, height = obs_img.size
    
    fx, fy, cx, cy = camera_intrinsics.params
    FoVx = 2 * np.arctan(width / (2 * fx))
    FoVy = 2 * np.arctan(height / (2 * fy))
    
    R = np.transpose(qvec2rotmat(target_image.qvec))
    T = np.array(target_image.tvec)
    
    world_view_transform = torch.tensor(
        getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).cuda().float()
    
    projection_matrix = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy
    ).transpose(0, 1).cuda().float()
    
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)
    
    class SimpleCamera:
        pass
    
    cam = SimpleCamera()
    cam.image_width = width
    cam.image_height = height
    cam.FoVx = FoVx
    cam.FoVy = FoVy
    cam.znear = 0.01
    cam.zfar = 100.0
    cam.world_view_transform = world_view_transform
    cam.projection_matrix = projection_matrix
    cam.full_proj_transform = full_proj_transform
    cam.camera_center = world_view_transform.inverse()[3, :3]
    
    # Render
    from clsplats.rendering import render
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    with torch.no_grad():
        result = render(cam, gaussians, bg_color)
    rendered = result["render"]  # [C, H, W]
    
    # Load observation
    obs_np = np.array(obs_img).astype(np.float32) / 255.0
    observed = torch.from_numpy(obs_np).cuda()  # [H, W, C]
    
    print(f"   Rendered shape: {rendered.shape}, range: [{rendered.min():.3f}, {rendered.max():.3f}]")
    print(f"   Observed shape: {observed.shape}, range: [{observed.min():.3f}, {observed.max():.3f}]")
    
    # Preprocess for DINOv2
    def preprocess(image_hwc):
        """Preprocess image [H, W, C] -> [C, H', W'] normalized"""
        DINO_PATCH_SIZE = 14
        h, w, c = image_hwc.shape
        aligned_h = h - (h % DINO_PATCH_SIZE)
        aligned_w = w - (w % DINO_PATCH_SIZE)
        image_t = image_hwc.permute(2, 0, 1)  # [C, H, W]
        resized = torch.nn.functional.interpolate(
            image_t.unsqueeze(0), size=(aligned_h, aligned_w), 
            mode="bilinear", align_corners=False
        )
        normalized = normalize(resized.squeeze(0))
        return normalized
    
    print("\n3. Preprocessing images for DINOv2...")
    rendered_hwc = rendered.permute(1, 2, 0)  # [H, W, C]
    
    rendered_proc = preprocess(rendered_hwc)
    observed_proc = preprocess(observed)
    
    print(f"   Rendered processed: {rendered_proc.shape}, range: [{rendered_proc.min():.3f}, {rendered_proc.max():.3f}]")
    print(f"   Observed processed: {observed_proc.shape}, range: [{observed_proc.min():.3f}, {observed_proc.max():.3f}]")
    
    # Extract features
    print("\n4. Extracting DINOv2 features...")
    with torch.no_grad():
        rendered_feats = model.get_intermediate_layers(
            rendered_proc.unsqueeze(0), reshape=True
        )
        observed_feats = model.get_intermediate_layers(
            observed_proc.unsqueeze(0), reshape=True
        )
    
    print(f"   get_intermediate_layers returns: {type(rendered_feats)}, len={len(rendered_feats)}")
    print(f"   rendered_feats[0] shape: {rendered_feats[0].shape}")
    print(f"   observed_feats[0] shape: {observed_feats[0].shape}")
    
    # Compute cosine similarity
    print("\n5. Computing cosine similarity...")
    cos = torch.nn.CosineSimilarity(dim=1)
    
    # The shape is [B, C, H, W] after reshape=True
    # We want to compare feature vectors across the channel dimension
    rf = rendered_feats[0]  # [B, C, H, W] = [1, 768, H/14, W/14]
    of = observed_feats[0]
    
    print(f"   Feature shapes: rendered={rf.shape}, observed={of.shape}")
    
    # Squeeze batch dimension for CosineSimilarity
    cos_sim = cos(rf, of)  # [H, W]
    
    print(f"\n6. Cosine similarity statistics:")
    print(f"   Shape: {cos_sim.shape}")
    print(f"   Min:   {cos_sim.min().item():.4f}")
    print(f"   Max:   {cos_sim.max().item():.4f}")
    print(f"   Mean:  {cos_sim.mean().item():.4f}")
    print(f"   Std:   {cos_sim.std().item():.4f}")
    
    # Distribution
    print(f"\n   Distribution:")
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        below = (cos_sim < thresh).float().mean().item() * 100
        print(f"   cos_sim < {thresh:.1f}: {below:.2f}%")
    
    # Check if same image gives high similarity
    print("\n7. Sanity check: comparing rendered with itself...")
    cos_sim_self = cos(rf, rf)
    print(f"   Self-similarity: min={cos_sim_self.min():.4f}, max={cos_sim_self.max():.4f}, mean={cos_sim_self.mean():.4f}")
    
    # Save visualization
    print("\n8. Saving cosine similarity heatmap...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.clip(rendered_hwc.cpu().numpy(), 0, 1))
    axes[0].set_title('Rendered')
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(observed.cpu().numpy(), 0, 1))
    axes[1].set_title('Observed')
    axes[1].axis('off')
    
    im = axes[2].imshow(cos_sim.cpu().numpy(), cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title(f'Cosine Similarity\n(mean={cos_sim.mean():.3f})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'debug_cosine_sim.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\n   Saved to: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
