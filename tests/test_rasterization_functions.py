"""
验证 diff_gaussian_rasterization 新增功能的测试脚本
"""
import torch
import math

print("="*60)
print("验证 diff_gaussian_rasterization 新增功能")
print("="*60)

# 1. 测试导入
print("\n[1] 测试模块导入...")
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        SparseGaussianAdam,
        compute_tile_mask_from_active_gaussians,
    )
    print("   ✓ 所有模块导入成功!")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    exit(1)

# 2. 测试 GaussianRasterizationSettings 的新参数
print("\n[2] 测试 GaussianRasterizationSettings 新参数...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 创建基本设置
    settings = GaussianRasterizationSettings(
        image_height=512,
        image_width=512,
        tanfovx=math.tan(0.5 * 1.0),
        tanfovy=math.tan(0.5 * 1.0),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
        active_gaussian_mask=None,  # 新参数
        tile_mask=None,             # 新参数
    )
    print("   ✓ GaussianRasterizationSettings 创建成功 (使用新参数)")
    
    # 验证参数存在
    assert hasattr(settings, 'active_gaussian_mask'), "缺少 active_gaussian_mask"
    assert hasattr(settings, 'tile_mask'), "缺少 tile_mask"
    print("   ✓ 新参数 active_gaussian_mask 和 tile_mask 已验证")
except Exception as e:
    print(f"   ✗ 测试失败: {e}")

# 3. 测试 SparseGaussianAdam
print("\n[3] 测试 SparseGaussianAdam 优化器...")
try:
    # 创建测试参数
    test_param = torch.randn(100, 3, device=device, requires_grad=True)
    optimizer = SparseGaussianAdam([test_param], lr=0.001)
    
    print(f"   ✓ SparseGaussianAdam 创建成功")
    
    # 测试 set_visible_mask 方法
    mask = torch.zeros(100, dtype=torch.bool, device=device)
    mask[:50] = True  # 前50个点是可见的
    optimizer.set_visible_mask(mask)
    print(f"   ✓ set_visible_mask 方法可用")
    
    # 测试优化步骤
    loss = test_param.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"   ✓ 稀疏优化步骤执行成功")
    
except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试 compute_tile_mask_from_active_gaussians
print("\n[4] 测试 compute_tile_mask_from_active_gaussians...")
try:
    # 创建测试数据
    num_gaussians = 100
    active_gaussian_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
    active_gaussian_mask[:30] = True  # 前30个是活跃的
    
    # 创建假的 2D 均值和半径
    means2D = torch.rand(num_gaussians, 2, device=device) * 512  # 随机位置
    radii = torch.ones(num_gaussians, dtype=torch.int32, device=device) * 10  # 固定半径
    
    tile_mask = compute_tile_mask_from_active_gaussians(
        active_gaussian_mask=active_gaussian_mask,
        means2D=means2D,
        radii=radii,
        image_height=512,
        image_width=512,
        block_x=16,
        block_y=16
    )
    
    print(f"   ✓ compute_tile_mask_from_active_gaussians 执行成功")
    print(f"   - 输入: {num_gaussians} Gaussians, {active_gaussian_mask.sum().item()} 活跃")
    print(f"   - 输出 tile_mask 形状: {tile_mask.shape}")
    print(f"   - 活跃 tiles: {tile_mask.sum().item()}/{tile_mask.numel()}")
    
except Exception as e:
    print(f"   ✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试完整的带 mask 的光栅化流程 (如果有 CUDA)
print("\n[5] 测试带 mask 参数的光栅化...")
if device.type == "cuda":
    try:
        # 创建测试 Gaussian 数据
        num_gaussians = 50
        means3D = torch.randn(num_gaussians, 3, device=device)
        means2D = torch.zeros(num_gaussians, 2, device=device, requires_grad=True)
        opacities = torch.sigmoid(torch.randn(num_gaussians, 1, device=device))
        scales = torch.abs(torch.randn(num_gaussians, 3, device=device)) * 0.01
        rotations = torch.randn(num_gaussians, 4, device=device)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        shs = torch.randn(num_gaussians, 16, 3, device=device)
        
        # 创建 active_gaussian_mask
        active_mask = torch.ones(num_gaussians, dtype=torch.bool, device=device)
        
        # 创建相机参数
        fov = 1.0
        viewmatrix = torch.eye(4, device=device)
        viewmatrix[2, 3] = 3.0  # 相机后退
        projmatrix = torch.eye(4, device=device)
        
        settings = GaussianRasterizationSettings(
            image_height=256,
            image_width=256,
            tanfovx=math.tan(0.5 * fov),
            tanfovy=math.tan(0.5 * fov),
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=torch.tensor([0, 0, 3], device=device, dtype=torch.float),
            prefiltered=False,
            debug=False,
            antialiasing=False,
            active_gaussian_mask=active_mask,
            tile_mask=None,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=settings)
        
        # 调用光栅化
        rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacities,
            shs=shs,
            scales=scales,
            rotations=rotations,
        )
        
        print(f"   ✓ 带 mask 的光栅化执行成功")
        print(f"   - 渲染图像形状: {rendered_image.shape}")
        print(f"   - 深度图形状: {depth.shape}")
        print(f"   - 可见 Gaussians: {(radii > 0).sum().item()}/{num_gaussians}")
        
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   ⚠ 跳过：需要 CUDA 设备")

# 总结
print("\n" + "="*60)
print("验证完成!")
print("="*60)
