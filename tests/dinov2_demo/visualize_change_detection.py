"""
DINOv2 Change Detection Visualization Demo

流程：
1. 指定一张 t1 观测图路径和对应的 sparse 文件夹
2. 从 sparse 读取该图对应的相机位姿
3. 用 t0 的 Gaussian 模型从该位姿渲染图片
4. 对比渲染图 vs 观测图，检测变化区域
5. 可视化结果
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# ============================================================
# 配置区域 - 修改这里的路径
# ============================================================

# t1 观测图路径
OBSERVATION_IMAGE_PATH = r"C:\3dgsprojects\cl-splats-main\cl-splats-dataset\test2\t2\d2_0087.png"

# t1 的 sparse 文件夹路径（包含 cameras.bin, images.bin）
SPARSE_PATH = r"C:\3dgsprojects\cl-splats-main\cl-splats-dataset\test2\output\t1\sparse\0"

# t0 训练好的 checkpoint 路径
CHECKPOINT_PATH = r"C:\3dgsprojects\cl-splats-main\outputs\checkpoint_t0.pt"

# 输出目录
OUTPUT_DIR = os.path.dirname(__file__)

# 变化检测参数
THRESHOLD = 0.2
DILATE = True
DILATE_KERNEL = 13  # 改小，之前 31 太大了

# ============================================================


def load_colmap_camera_by_image_name(sparse_path: str, image_name: str):
    """
    从 COLMAP sparse 文件夹中读取指定图片的相机信息。
    
    Returns:
        camera_intrinsics: Camera namedtuple (id, model, width, height, params)
        image_extrinsics: Image namedtuple (id, qvec, tvec, camera_id, name, ...)
    """
    from clsplats.dataset.colmap_reader import (
        read_intrinsics_binary, read_intrinsics_text,
        read_extrinsics_binary, read_extrinsics_text
    )
    
    # Try binary first, then text
    cameras_bin = os.path.join(sparse_path, "cameras.bin")
    cameras_txt = os.path.join(sparse_path, "cameras.txt")
    images_bin = os.path.join(sparse_path, "images.bin")
    images_txt = os.path.join(sparse_path, "images.txt")
    
    if os.path.exists(cameras_bin):
        cameras = read_intrinsics_binary(cameras_bin)
    elif os.path.exists(cameras_txt):
        cameras = read_intrinsics_text(cameras_txt)
    else:
        raise FileNotFoundError(f"No cameras file found in {sparse_path}")
    
    if os.path.exists(images_bin):
        images = read_extrinsics_binary(images_bin)
    elif os.path.exists(images_txt):
        images = read_extrinsics_text(images_txt)
    else:
        raise FileNotFoundError(f"No images file found in {sparse_path}")
    
    # Find the image by name
    target_image = None
    for img in images.values():
        if img.name == image_name:
            target_image = img
            break
    
    if target_image is None:
        print(f"Available images: {[img.name for img in images.values()]}")
        raise ValueError(f"Image '{image_name}' not found in {sparse_path}")
    
    camera = cameras[target_image.camera_id]
    
    return camera, target_image


def build_camera_for_rendering(camera_intrinsics, image_extrinsics, image_path: str):
    """
    根据 COLMAP 相机信息构建用于渲染的 Camera 对象。
    
    注意：COLMAP 的旋转矩阵需要转置才能用于渲染！
    """
    from clsplats.dataset.colmap_reader import qvec2rotmat
    from clsplats.utils.graphics_utils import getWorld2View2, getProjectionMatrix
    
    # Load image to get actual dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Get intrinsics
    if camera_intrinsics.model == "PINHOLE":
        fx, fy, cx, cy = camera_intrinsics.params
    elif camera_intrinsics.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera_intrinsics.params
        fx = fy = f
    else:
        raise ValueError(f"Unsupported camera model: {camera_intrinsics.model}")
    
    # Compute FoV
    FoVx = 2 * np.arctan(width / (2 * fx))
    FoVy = 2 * np.arctan(height / (2 * fy))
    
    # Get extrinsics - 关键：需要转置旋转矩阵！
    R = np.transpose(qvec2rotmat(image_extrinsics.qvec))  # 转置！
    T = np.array(image_extrinsics.tvec)
    
    print(f"\n   === Processed for Rendering ===", flush=True)
    print(f"   R (transposed):", flush=True)
    print(f"     {R[0]}", flush=True)
    print(f"     {R[1]}", flush=True)
    print(f"     {R[2]}", flush=True)
    print(f"   T: {T}", flush=True)
    
    # Build transforms
    print(f"   Building world_view_transform...", flush=True)
    world_view_transform = torch.tensor(
        getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).cuda().float()
    
    projection_matrix = getProjectionMatrix(
        znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy
    ).transpose(0, 1).cuda().float()
    
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(
        projection_matrix.unsqueeze(0)
    ).squeeze(0)
    
    # Create a simple camera object
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
    
    return cam


def render_from_camera(gaussians, camera, bg_color):
    """用 Gaussian 模型从指定相机位置渲染图片。"""
    from clsplats.rendering import render
    
    with torch.no_grad():
        result = render(camera, gaussians, bg_color)
    return result["render"]  # [C, H, W]


def main():
    print("=" * 60)
    print("DINOv2 Change Detection Demo")
    print("=" * 60)
    
    # Check paths
    if not os.path.exists(OBSERVATION_IMAGE_PATH):
        print(f"ERROR: Observation image not found: {OBSERVATION_IMAGE_PATH}")
        return
    if not os.path.exists(SPARSE_PATH):
        print(f"ERROR: Sparse path not found: {SPARSE_PATH}")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 获取图片文件名
    image_name = os.path.basename(OBSERVATION_IMAGE_PATH)
    print(f"\n1. Target image: {image_name}")
    
    # 2. 从 sparse 读取相机位姿
    print(f"\n2. Loading camera pose from: {SPARSE_PATH}")
    camera_intrinsics, image_extrinsics = load_colmap_camera_by_image_name(
        SPARSE_PATH, image_name
    )
    
    # 详细输出位姿信息
    print(f"\n   === Camera Intrinsics ===")
    print(f"   Camera ID: {camera_intrinsics.id}")
    print(f"   Model: {camera_intrinsics.model}")
    print(f"   Width x Height: {camera_intrinsics.width} x {camera_intrinsics.height}")
    print(f"   Params: {camera_intrinsics.params}")
    if camera_intrinsics.model == "PINHOLE":
        fx, fy, cx, cy = camera_intrinsics.params
        print(f"     fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    elif camera_intrinsics.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera_intrinsics.params
        print(f"     f={f:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    
    print(f"\n   === Image Extrinsics ===")
    print(f"   Image ID: {image_extrinsics.id}")
    print(f"   Image Name: {image_extrinsics.name}")
    print(f"   Camera ID: {image_extrinsics.camera_id}")
    print(f"   Quaternion (qvec): {image_extrinsics.qvec}")
    print(f"   Translation (tvec): {image_extrinsics.tvec}")
    
    # 计算并输出旋转矩阵
    from clsplats.dataset.colmap_reader import qvec2rotmat
    R = qvec2rotmat(image_extrinsics.qvec)
    print(f"\n   === Rotation Matrix (from qvec) ===")
    print(f"   {R[0]}")
    print(f"   {R[1]}")
    print(f"   {R[2]}")
    
    # 计算相机在世界坐标系中的位置
    camera_position = -R.T @ image_extrinsics.tvec
    print(f"\n   === Camera Position (world coords) ===")
    print(f"   X: {camera_position[0]:.6f}")
    print(f"   Y: {camera_position[1]:.6f}")
    print(f"   Z: {camera_position[2]:.6f}")
    
    # 3. 构建渲染用的相机
    print(f"\n3. Building camera for rendering...", flush=True)
    try:
        camera = build_camera_for_rendering(
            camera_intrinsics, image_extrinsics, OBSERVATION_IMAGE_PATH
        )
        print(f"   FoVx: {np.degrees(camera.FoVx):.2f}°, FoVy: {np.degrees(camera.FoVy):.2f}°", flush=True)
    except Exception as e:
        print(f"   ERROR building camera: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 4. 加载 Gaussian 模型
    print(f"\n4. Loading Gaussian model from checkpoint...", flush=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    ply_path = checkpoint.get("ply_path")
    if not ply_path or not os.path.exists(ply_path):
        print(f"ERROR: PLY file not found: {ply_path}")
        return
    
    from clsplats.representation.gaussian_model import GaussianModel
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(ply_path)
    print(f"   Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # 5. 渲染图片
    print(f"\n5. Rendering from t0 model...")
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    rendered = render_from_camera(gaussians, camera, bg_color)  # [C, H, W]
    print(f"   Rendered image shape: {rendered.shape}")
    
    # 6. 加载观测图
    print(f"\n6. Loading observation image...")
    obs_img = Image.open(OBSERVATION_IMAGE_PATH).convert('RGB')
    obs_np = np.array(obs_img).astype(np.float32) / 255.0
    observed = torch.from_numpy(obs_np).cuda()  # [H, W, C]
    print(f"   Observation shape: {observed.shape}")
    
    # 7. 变化检测 - 直接计算余弦相似度（和debug_heatmap.py一样）
    print(f"\n7. Detecting changes...")
    print(f"   Threshold: {THRESHOLD}, Dilate: {DILATE}, Kernel: {DILATE_KERNEL}")
    
    import torchvision
    
    # Load DINOv2
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    def preprocess_for_dino(image_hwc):
        """Preprocess [H, W, C] image for DINOv2"""
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
    
    # Convert rendered to [H, W, C]
    rendered_hwc = rendered.permute(1, 2, 0)
    
    # Preprocess both images
    rendered_proc = preprocess_for_dino(rendered_hwc)
    observed_proc = preprocess_for_dino(observed)
    
    # Extract DINOv2 features
    with torch.no_grad():
        rendered_feats = model.get_intermediate_layers(rendered_proc.unsqueeze(0), reshape=True)
        observed_feats = model.get_intermediate_layers(observed_proc.unsqueeze(0), reshape=True)
    
    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1)
    cos_sim = cos(rendered_feats[0], observed_feats[0]).squeeze(0)  # [H, W] on feature map
    
    # Upsample cosine similarity values (NOT binary mask) to original resolution
    height, width = observed.shape[:2]
    cos_sim_up = torch.nn.functional.interpolate(
        cos_sim.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    
    # Threshold to get mask
    mask = cos_sim_up < THRESHOLD
    
    # Optional dilation
    if DILATE:
        kernel = torch.ones((1, 1, DILATE_KERNEL, DILATE_KERNEL), dtype=torch.float32, device=device)
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)
        dilated = torch.nn.functional.conv2d(mask_float, kernel, padding=DILATE_KERNEL // 2)
        mask = (dilated > 0).squeeze()
    
    changed_pixels = mask.sum().item()
    total_pixels = mask.numel()
    changed_pct = 100 * changed_pixels / total_pixels
    print(f"   Changed pixels: {changed_pixels}/{total_pixels} ({changed_pct:.2f}%)")
    
    # 8. 可视化 - 显示余弦相似度热力图
    print(f"\n8. Creating visualization...")
    
    rendered_np = rendered.permute(1, 2, 0).cpu().numpy()
    observed_np = observed.cpu().numpy()
    mask_np = mask.cpu().numpy().astype(np.float32)
    cos_sim_np = cos_sim_up.cpu().numpy()
    

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1
    axes[0, 0].imshow(np.clip(rendered_np, 0, 1))
    axes[0, 0].set_title('Rendered (from t0 model)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.clip(observed_np, 0, 1))
    axes[0, 1].set_title('Observed (t1)')
    axes[0, 1].axis('off')
    
    # Pixel difference
    diff = np.abs(rendered_np - observed_np).mean(axis=2)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Pixel Difference')
    axes[0, 2].axis('off')
    
    # Row 2 - 显示余弦相似度热力图和掩码
    im1 = axes[1, 0].imshow(cos_sim_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Cosine Similarity\n(Green=Same, Red=Different)')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # Binary mask
    axes[1, 1].imshow(mask_np, cmap='gray')
    axes[1, 1].set_title(f'Change Mask (< {THRESHOLD})\n({changed_pct:.1f}% changed)')
    axes[1, 1].axis('off')
    
    # Overlay on observed
    overlay_observed = observed_np.copy()
    overlay_observed[mask_np > 0.5] = [1, 0, 0]
    axes[1, 2].imshow(np.clip(overlay_observed, 0, 1))
    axes[1, 2].set_title('Changes on Observed')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Change Detection: {image_name}\n'
                 f'Threshold={THRESHOLD}, Dilate={DILATE}, Kernel={DILATE_KERNEL}',
                 fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'change_detection_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n" + "=" * 60)
    print(f"Done! Saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
