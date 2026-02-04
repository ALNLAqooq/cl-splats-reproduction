"""
Debug: Visualize cosine similarity heatmap to understand which regions have high/low similarity.
"""

import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

# Config
OBSERVATION_IMAGE_PATH = r"C:\3dgsprojects\cl-splats-main\tests\dinov2_demo\day_1_0020.png"
SPARSE_PATH = r"C:\3dgsprojects\cl-splats-main\cl-splats-dataset\test1\output\t1\sparse\0"
CHECKPOINT_PATH = r"C:\3dgsprojects\cl-splats-main\outputs\checkpoint_t0.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading DINOv2...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    # Load and render
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    ply_path = checkpoint.get("ply_path")
    
    from clsplats.representation.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    
    from clsplats.dataset.colmap_reader import read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
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
    
    from clsplats.rendering import render
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    with torch.no_grad():
        result = render(cam, gaussians, bg_color)
    rendered = result["render"]  # [C, H, W]
    
    obs_np = np.array(obs_img).astype(np.float32) / 255.0
    observed = torch.from_numpy(obs_np).cuda()  # [H, W, C]
    
    # Preprocess
    def preprocess(image_hwc):
        DINO_PATCH_SIZE = 14
        h, w, c = image_hwc.shape
        aligned_h = h - (h % DINO_PATCH_SIZE)
        aligned_w = w - (w % DINO_PATCH_SIZE)
        image_t = image_hwc.permute(2, 0, 1)
        resized = torch.nn.functional.interpolate(
            image_t.unsqueeze(0), size=(aligned_h, aligned_w), 
            mode="bilinear", align_corners=False
        )
        normalized = normalize(resized.squeeze(0))
        return normalized
    
    rendered_hwc = rendered.permute(1, 2, 0)
    rendered_proc = preprocess(rendered_hwc)
    observed_proc = preprocess(observed)
    
    # Features
    with torch.no_grad():
        rendered_feats = model.get_intermediate_layers(rendered_proc.unsqueeze(0), reshape=True)
        observed_feats = model.get_intermediate_layers(observed_proc.unsqueeze(0), reshape=True)
    
    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(rendered_feats[0], observed_feats[0]).squeeze(0)  # [H, W]
    
    # Upsample cos_sim to original size for visualization
    cos_sim_up = torch.nn.functional.interpolate(
        cos_sim.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Images
    axes[0, 0].imshow(np.clip(rendered_hwc.cpu().numpy(), 0, 1))
    axes[0, 0].set_title('Rendered (from t0 model)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(observed.cpu().numpy(), 0, 1))
    axes[0, 1].set_title('Observed (t1)')
    axes[0, 1].axis('off')
    
    # Pixel difference
    diff = np.abs(rendered_hwc.cpu().numpy() - observed.cpu().numpy()).mean(axis=2)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Pixel Difference')
    axes[0, 2].axis('off')
    
    # Row 2: Cosine similarity analysis
    im1 = axes[1, 0].imshow(cos_sim_up.cpu().numpy(), cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Cosine Similarity\n(Green=Same, Red=Different)')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # Show where cos_sim < 0.5 (marked as changed)
    mask_05 = (cos_sim_up < 0.5).cpu().numpy()
    axes[1, 1].imshow(mask_05, cmap='gray')
    pct_05 = mask_05.mean() * 100
    axes[1, 1].set_title(f'Changed (cos_sim < 0.5)\n{pct_05:.1f}% marked as changed')
    axes[1, 1].axis('off')
    
    # Show where cos_sim < 0.4
    mask_04 = (cos_sim_up < 0.4).cpu().numpy()
    axes[1, 2].imshow(mask_04, cmap='gray')
    pct_04 = mask_04.mean() * 100
    axes[1, 2].set_title(f'Changed (cos_sim < 0.4)\n{pct_04:.1f}% marked as changed')
    axes[1, 2].axis('off')
    
    plt.suptitle('DINOv2 Cosine Similarity Analysis\n(Before Dilation)', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'cosine_similarity_heatmap.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")
    
    # Also print stats for regions
    print("\n=== Cosine Similarity Statistics ===")
    print(f"Overall: min={cos_sim.min():.3f}, max={cos_sim.max():.3f}, mean={cos_sim.mean():.3f}")

if __name__ == '__main__':
    main()
