"""
Test script for the sampling module.

This script demonstrates how to use the GaussianSampler to generate
new points in changed regions.
"""

import torch
import numpy as np
import omegaconf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clsplats.sampling.gaussian_sampler import GaussianSampler


class MockCamera:
    """Mock camera for testing."""
    
    def __init__(self, image_width=640, image_height=480):
        self.image_width = image_width
        self.image_height = image_height
        
        # Create simple projection matrices
        self.world_view_transform = torch.eye(4, device='cuda')
        self.projection_matrix = torch.eye(4, device='cuda')
        
        # Simple perspective projection
        fov = np.pi / 3  # 60 degrees
        aspect = image_width / image_height
        near, far = 0.1, 100.0
        
        f = 1.0 / np.tan(fov / 2)
        self.projection_matrix[0, 0] = f / aspect
        self.projection_matrix[1, 1] = f
        self.projection_matrix[2, 2] = (far + near) / (near - far)
        self.projection_matrix[2, 3] = (2 * far * near) / (near - far)
        self.projection_matrix[3, 2] = -1.0
        self.projection_matrix[3, 3] = 0.0


def create_test_data():
    """Create synthetic test data."""
    
    # Create existing Gaussians (a simple point cloud)
    num_existing = 1000
    existing_gaussians = torch.randn(num_existing, 3, device='cuda') * 2.0
    
    # Create mock cameras
    num_views = 5
    cameras = [MockCamera() for _ in range(num_views)]
    
    # Create change masks (circular region in the center)
    change_masks = []
    for _ in range(num_views):
        mask = torch.zeros(480, 640, dtype=torch.bool, device='cuda')
        # Create a circular mask in the center
        y, x = torch.meshgrid(
            torch.arange(480, device='cuda'),
            torch.arange(640, device='cuda'),
            indexing='ij'
        )
        center_x, center_y = 320, 240
        radius = 100
        dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask[dist < radius] = True
        change_masks.append(mask)
    
    return existing_gaussians, cameras, change_masks


def test_full_scene_sampling():
    """Test full-scene sampling strategy."""
    print("\n" + "="*60)
    print("Testing Full-Scene Sampling (Algorithm 2)")
    print("="*60)
    
    # Create configuration
    cfg = omegaconf.DictConfig({
        'num_clusters': 10,
        'samples_per_round': 100,
        'max_rounds': 10,
        'min_points_for_region_sampling': 5000,  # Set high to force full-scene sampling
        'majority_vote_threshold': 2
    })
    
    # Initialize sampler
    sampler = GaussianSampler(cfg)
    
    # Create test data
    existing_gaussians, cameras, change_masks = create_test_data()
    
    print(f"Existing Gaussians: {existing_gaussians.shape[0]}")
    print(f"Number of views: {len(cameras)}")
    print(f"Change mask size: {change_masks[0].shape}")
    print(f"Changed pixels per view: {[mask.sum().item() for mask in change_masks]}")
    
    # Sample new points
    num_samples = 500
    print(f"\nSampling {num_samples} new points...")
    
    sampled_points, valid_mask = sampler.sample_new_points(
        existing_gaussians, change_masks, cameras, num_samples
    )
    
    print(f"Total sampled points: {sampled_points.shape[0]}")
    print(f"Valid points: {valid_mask.sum().item()}")
    print(f"Valid ratio: {valid_mask.sum().item() / sampled_points.shape[0]:.2%}")
    
    # Get only valid samples
    valid_samples = sampler.get_valid_samples(sampled_points, valid_mask, num_samples)
    print(f"Final valid samples: {valid_samples.shape[0]}")
    
    return valid_samples


def test_region_sampling():
    """Test region-based sampling strategy."""
    print("\n" + "="*60)
    print("Testing Region-Based Sampling (Algorithm 3)")
    print("="*60)
    
    # Create configuration
    cfg = omegaconf.DictConfig({
        'num_clusters': 10,
        'samples_per_round': 100,
        'max_rounds': 10,
        'min_points_for_region_sampling': 50,  # Set low to force region sampling
        'majority_vote_threshold': 2
    })
    
    # Initialize sampler
    sampler = GaussianSampler(cfg)
    
    # Create test data with more points in changed region
    existing_gaussians, cameras, change_masks = create_test_data()
    
    # Add more points in the changed region (center of scene)
    changed_region_points = torch.randn(200, 3, device='cuda') * 0.5
    existing_gaussians = torch.cat([existing_gaussians, changed_region_points], dim=0)
    
    print(f"Existing Gaussians: {existing_gaussians.shape[0]}")
    print(f"Number of views: {len(cameras)}")
    
    # Sample new points
    num_samples = 500
    print(f"\nSampling {num_samples} new points...")
    
    sampled_points, valid_mask = sampler.sample_new_points(
        existing_gaussians, change_masks, cameras, num_samples
    )
    
    print(f"Total sampled points: {sampled_points.shape[0]}")
    print(f"Valid points: {valid_mask.sum().item()}")
    print(f"Valid ratio: {valid_mask.sum().item() / sampled_points.shape[0]:.2%}")
    
    # Get only valid samples
    valid_samples = sampler.get_valid_samples(sampled_points, valid_mask, num_samples)
    print(f"Final valid samples: {valid_samples.shape[0]}")
    
    return valid_samples


def test_majority_voting():
    """Test majority voting mechanism."""
    print("\n" + "="*60)
    print("Testing Majority Voting")
    print("="*60)
    
    cfg = omegaconf.DictConfig({
        'num_clusters': 10,
        'samples_per_round': 100,
        'max_rounds': 10,
        'min_points_for_region_sampling': 50,
        'majority_vote_threshold': 3  # Require 3 views
    })
    
    sampler = GaussianSampler(cfg)
    
    # Create test points
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],  # Center point
        [5.0, 5.0, 5.0],  # Far away point
        [0.5, 0.5, 0.5],  # Near center
    ], device='cuda')
    
    existing_gaussians, cameras, change_masks = create_test_data()
    
    # Test with different thresholds
    for threshold in [1, 2, 3]:
        valid_mask = sampler.majority_vote(
            test_points, change_masks, cameras, threshold=threshold
        )
        print(f"Threshold {threshold}: {valid_mask.sum().item()}/{len(test_points)} points valid")
        print(f"  Valid points: {valid_mask.cpu().numpy()}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CL-Splats Sampling Module Tests")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, tests may fail or be slow")
        return
    
    try:
        # Test majority voting
        test_majority_voting()
        
        # Test full-scene sampling
        valid_samples_full = test_full_scene_sampling()
        
        # Test region-based sampling
        valid_samples_region = test_region_sampling()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
