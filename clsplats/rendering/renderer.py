"""
Gaussian Splatting Renderer.

Wrapper around diff-gaussian-rasterization for rendering Gaussian scenes.
"""

import torch
import math
from typing import Optional

# Try to import the CUDA rasterizer
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer
    )
    RASTERIZER_AVAILABLE = True
except ImportError:
    RASTERIZER_AVAILABLE = False
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    print("Warning: diff_gaussian_rasterization not available. Rendering will not work.")


def render(
    viewpoint_camera,
    gaussians,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    antialiasing: bool = False,
    active_gaussian_mask: Optional[torch.Tensor] = None,
    tile_mask: Optional[torch.Tensor] = None,
):
    """
    Render the scene from a viewpoint.
    
    Args:
        viewpoint_camera: Camera object with view/projection matrices
        gaussians: GaussianModel containing the scene
        bg_color: Background color [3]
        scaling_modifier: Scale modifier for Gaussians
        override_color: Optional override for Gaussian colors
        antialiasing: Whether to use antialiasing
        active_gaussian_mask: Optional mask for local optimization (CL-Splats)
        tile_mask: Optional tile mask for local optimization (CL-Splats)
        
    Returns:
        Dictionary containing:
            - render: Rendered image [3, H, W]
            - viewspace_points: 2D positions of Gaussians
            - visibility_filter: Mask of visible Gaussians
            - radii: 2D radii of Gaussians
    """
    if not RASTERIZER_AVAILABLE:
        raise RuntimeError(
            "diff_gaussian_rasterization is not available. "
            "Please install it from submodules/diff-gaussian-rasterization"
        )
    
    # Create rasterization settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=antialiasing,
        active_gaussian_mask=active_gaussian_mask,
        tile_mask=tile_mask,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Get Gaussian properties
    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    
    try:
        means2D.retain_grad()
    except:
        pass
    
    opacity = gaussians.get_opacity
    
    # Get scales and rotations
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    # Get colors (SH or override)
    if override_color is None:
        shs = gaussians.get_features
    else:
        shs = None
    
    # Rasterize - returns (color, radii, invdepths)
    rendered_image, radii, invdepths = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    # Visibility filter
    visibility_filter = radii > 0
    
    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "invdepths": invdepths,
    }


def render_simple(
    viewpoint_camera,
    means3D: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    bg_color: torch.Tensor,
    sh_degree: int = 0,
    antialiasing: bool = False,
    active_gaussian_mask: Optional[torch.Tensor] = None,
    tile_mask: Optional[torch.Tensor] = None,
):
    """
    Simplified render function with explicit parameters.
    
    Useful for rendering without a full GaussianModel.
    """
    if not RASTERIZER_AVAILABLE:
        raise RuntimeError("diff_gaussian_rasterization is not available.")
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=antialiasing,
        active_gaussian_mask=active_gaussian_mask,
        tile_mask=tile_mask,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    
    rendered_image, radii, invdepths = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    return rendered_image, radii
