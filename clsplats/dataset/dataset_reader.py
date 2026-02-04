"""
Dataset reader for CL-Splats.

Supports loading COLMAP and NeRF Synthetic datasets with multi-timestep support
for continual learning scenarios.
"""

import os
import sys
import json
from typing import List, Dict, NamedTuple, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from clsplats.dataset.colmap_reader import (
    read_extrinsics_text, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text,
    qvec2rotmat
)
from clsplats.utils.graphics_utils import (
    BasicPointCloud, getWorld2View2, focal2fov, fov2focal
)
from clsplats.utils.sh_utils import SH2RGB
from clsplats.utils.camera_utils import Camera


class CameraInfo(NamedTuple):
    """Camera information container."""
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    depth_path: str = ""
    depth_params: dict = None
    is_test: bool = False


class SceneInfo(NamedTuple):
    """Scene information container."""
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool = False


class CLSplatsDataset:
    """
    Dataset class for CL-Splats with multi-timestep support.
    
    Supports two data organization modes:
    1. Single timestep: Standard COLMAP/NeRF format
    2. Multi-timestep: Separate folders for each timestep (t0, t1, t2, ...)
    """
    
    def __init__(
        self,
        path: str,
        resolution_scale: float = 1.0,
        white_background: bool = False,
        eval_mode: bool = False,
        device: str = "cuda"
    ):
        """
        Initialize the dataset.
        
        Args:
            path: Path to dataset root
            resolution_scale: Scale factor for image resolution
            white_background: Whether to use white background
            eval_mode: Whether to split train/test
            device: Device for tensors
        """
        self.path = path
        self.resolution_scale = resolution_scale
        self.white_background = white_background
        self.eval_mode = eval_mode
        self.device = device
        
        # Detect dataset type and timesteps
        self.timesteps = self._detect_timesteps()
        self.dataset_type = self._detect_dataset_type()
        
        # Cache for loaded data
        self._scene_info_cache: Dict[int, SceneInfo] = {}
        self._cameras_cache: Dict[int, List[Camera]] = {}
        self._images_cache: Dict[int, List[torch.Tensor]] = {}
    
    def _detect_timesteps(self) -> List[int]:
        """Detect available timesteps in the dataset."""
        # Check for multi-timestep structure (t0, t1, t2, ...)
        timesteps = []
        for item in os.listdir(self.path):
            if item.startswith('t') and item[1:].isdigit():
                timesteps.append(int(item[1:]))
        
        if timesteps:
            return sorted(timesteps)
        else:
            # Single timestep dataset
            return [0]
    
    def _detect_dataset_type(self) -> str:
        """Detect dataset type (colmap or blender)."""
        # Check first timestep path
        check_path = self._get_timestep_path(self.timesteps[0])
        
        if os.path.exists(os.path.join(check_path, "sparse")):
            return "colmap"
        elif os.path.exists(os.path.join(check_path, "transforms_train.json")):
            return "blender"
        else:
            raise ValueError(f"Unknown dataset type at {check_path}")
    
    def _get_timestep_path(self, timestep: int) -> str:
        """Get path for a specific timestep."""
        if len(self.timesteps) == 1 and self.timesteps[0] == 0:
            return self.path
        else:
            return os.path.join(self.path, f"t{timestep}")
    
    def get_num_timesteps(self) -> int:
        """Get number of timesteps."""
        return len(self.timesteps)
    
    def get_scene_info(self, timestep: int = 0) -> SceneInfo:
        """Get scene information for a timestep."""
        if timestep not in self._scene_info_cache:
            path = self._get_timestep_path(timestep)
            
            if self.dataset_type == "colmap":
                scene_info = self._read_colmap_scene(path)
            else:
                scene_info = self._read_nerf_synthetic_scene(path)
            
            self._scene_info_cache[timestep] = scene_info
        
        return self._scene_info_cache[timestep]
    
    def get_cameras(self, timestep: int = 0) -> List[Camera]:
        """Get camera objects for a timestep."""
        if timestep not in self._cameras_cache:
            scene_info = self.get_scene_info(timestep)
            cameras = []
            
            cam_infos = scene_info.train_cameras
            if self.eval_mode:
                cam_infos = cam_infos + scene_info.test_cameras
            
            for cam_info in cam_infos:
                camera = self._load_camera(cam_info)
                cameras.append(camera)
            
            self._cameras_cache[timestep] = cameras
        
        return self._cameras_cache[timestep]
    
    def get_images(self, timestep: int = 0) -> List[torch.Tensor]:
        """Get ground truth images for a timestep."""
        if timestep not in self._images_cache:
            cameras = self.get_cameras(timestep)
            images = []
            
            for camera in cameras:
                if camera.original_image is not None:
                    images.append(camera.original_image)
            
            self._images_cache[timestep] = images
        
        return self._images_cache[timestep]
    
    def get_point_cloud(self, timestep: int = 0) -> BasicPointCloud:
        """Get initial point cloud for a timestep."""
        scene_info = self.get_scene_info(timestep)
        return scene_info.point_cloud
    
    def _load_camera(self, cam_info: CameraInfo) -> Camera:
        """Load a Camera object from CameraInfo."""
        # Load image
        image = None
        if os.path.exists(cam_info.image_path):
            pil_image = Image.open(cam_info.image_path)
            
            # Resize if needed
            if self.resolution_scale != 1.0:
                new_w = int(cam_info.width / self.resolution_scale)
                new_h = int(cam_info.height / self.resolution_scale)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            else:
                new_w, new_h = cam_info.width, cam_info.height
            
            # Convert to tensor
            im_data = np.array(pil_image.convert("RGBA"))
            
            # Handle background
            bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            
            image = torch.from_numpy(arr).float().permute(2, 0, 1).to(self.device)
        else:
            new_w, new_h = cam_info.width, cam_info.height
        
        return Camera(
            uid=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image_width=new_w,
            image_height=new_h,
            image_name=cam_info.image_name,
            image=image,
            device=self.device
        )
    
    def _read_colmap_scene(self, path: str) -> SceneInfo:
        """Read COLMAP format scene."""
        # Try binary first, then text
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        # Read cameras
        images_folder = os.path.join(path, "images")
        cam_infos = self._read_colmap_cameras(
            cam_extrinsics, cam_intrinsics, images_folder
        )
        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
        
        # Split train/test
        if self.eval_mode:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
        
        # Compute normalization
        nerf_normalization = self._get_nerf_normalization(train_cam_infos)
        
        # Load point cloud
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply...")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            self._store_ply(ply_path, xyz, rgb)
        
        pcd = self._fetch_ply(ply_path)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
            is_nerf_synthetic=False
        )
    
    def _read_colmap_cameras(
        self,
        cam_extrinsics: dict,
        cam_intrinsics: dict,
        images_folder: str
    ) -> List[CameraInfo]:
        """Read camera information from COLMAP data."""
        cam_infos = []
        
        # Get list of actual images in the folder
        existing_images = set()
        if os.path.exists(images_folder):
            existing_images = set(os.listdir(images_folder))
        
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            
            # Skip if image doesn't exist in this timestep's folder
            if existing_images and extr.name not in existing_images:
                continue
            
            height = intr.height
            width = intr.width
            uid = intr.id
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                raise ValueError(f"Camera model {intr.model} not supported")
            
            image_path = os.path.join(images_folder, extr.name)
            image_name = extr.name
            
            cam_info = CameraInfo(
                uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                image_path=image_path, image_name=image_name,
                width=width, height=height
            )
            cam_infos.append(cam_info)
        
        return cam_infos
    
    def _read_nerf_synthetic_scene(self, path: str) -> SceneInfo:
        """Read NeRF Synthetic format scene."""
        # Read train cameras
        train_cam_infos = self._read_nerf_cameras(
            path, "transforms_train.json", is_test=False
        )
        
        # Read test cameras
        test_cam_infos = []
        if self.eval_mode:
            test_cam_infos = self._read_nerf_cameras(
                path, "transforms_test.json", is_test=True
            )
        
        nerf_normalization = self._get_nerf_normalization(train_cam_infos)
        
        # Generate or load point cloud
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self._store_ply(ply_path, xyz, SH2RGB(shs) * 255)
        else:
            pcd = self._fetch_ply(ply_path)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
            is_nerf_synthetic=True
        )
    
    def _read_nerf_cameras(
        self,
        path: str,
        transforms_file: str,
        is_test: bool,
        extension: str = ".png"
    ) -> List[CameraInfo]:
        """Read cameras from NeRF transforms file."""
        cam_infos = []
        transforms_path = os.path.join(path, transforms_file)
        
        if not os.path.exists(transforms_path):
            return cam_infos
        
        with open(transforms_path) as f:
            contents = json.load(f)
        
        fovx = contents["camera_angle_x"]
        
        for idx, frame in enumerate(contents["frames"]):
            file_path = frame["file_path"]
            # Only add extension if file_path doesn't already have one
            if not file_path.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                file_path = file_path + extension
            cam_name = os.path.join(path, file_path)
            
            # Camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # Convert from OpenGL to COLMAP convention
            c2w[:3, 1:3] *= -1
            
            # World-to-camera
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            # Load image to get dimensions
            if os.path.exists(cam_name):
                image = Image.open(cam_name)
                width, height = image.size
            else:
                width, height = 800, 800  # Default
            
            fovy = focal2fov(fov2focal(fovx, width), height)
            
            cam_info = CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                image_path=cam_name, image_name=Path(cam_name).stem,
                width=width, height=height, is_test=is_test
            )
            cam_infos.append(cam_info)
        
        return cam_infos
    
    def _get_nerf_normalization(self, cam_infos: List[CameraInfo]) -> dict:
        """Compute NeRF++ normalization parameters."""
        cam_centers = []
        
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        radius = diagonal * 1.1
        translate = -center.flatten()
        
        return {"translate": translate, "radius": radius}
    
    def _fetch_ply(self, path: str) -> BasicPointCloud:
        """Load point cloud from PLY file."""
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        
        try:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        except:
            normals = np.zeros_like(positions)
        
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    def _store_ply(self, path: str, xyz: np.ndarray, rgb: np.ndarray):
        """Store point cloud to PLY file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        
        normals = np.zeros_like(xyz)
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        vertex_element = PlyElement.describe(elements, 'vertex')
        PlyData([vertex_element]).write(path)


# Convenience function for backward compatibility
def load_dataset(cfg) -> CLSplatsDataset:
    """Load dataset from configuration."""
    return CLSplatsDataset(
        path=cfg.get("path"),
        resolution_scale=cfg.get("resolution", 1.0),
        white_background=cfg.get("white_background", False),
        eval_mode=cfg.get("eval", False),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
