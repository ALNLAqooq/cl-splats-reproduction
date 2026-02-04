"""
Test script for the pruning module.

This script demonstrates how to use the SpherePruner to constrain
Gaussian optimization within detected change regions.
"""

import torch
import numpy as np
import omegaconf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clsplats.pruning.sphere_pruner import SpherePruner


def create_test_gaussians():
    """Create synthetic test Gaussians in multiple clusters."""
    
    # Create three distinct clusters
    cluster1 = torch.randn(500, 3, device='cuda') * 0.5 + torch.tensor([0, 0, 0], device='cuda')
    cluster2 = torch.randn(500, 3, device='cuda') * 0.5 + torch.tensor([5, 0, 0], device='cuda')
    cluster3 = torch.randn(500, 3, device='cuda') * 0.5 + torch.tensor([0, 5, 0], device='cuda')
    
    gaussians = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    return gaussians


def test_sphere_fitting():
    """Test sphere fitting on clustered data."""
    print("\n" + "="*60)
    print("Testing Sphere Fitting")
    print("="*60)
    
    # Create configuration
    cfg = omegaconf.DictConfig({
        'min_cluster_size': 100,
        'quantile': 0.98,
        'radius_multiplier': 1.1
    })
    
    # Initialize pruner
    pruner = SpherePruner(cfg)
    
    # Create test data
    gaussians = create_test_gaussians()
    print(f"Created {gaussians.shape[0]} test Gaussians in 3 clusters")
    
    # Fit spheres
    pruner.fit(gaussians)
    
    print(f"Fitted {pruner.get_num_spheres()} bounding spheres")
    
    # Display sphere info
    sphere_info = pruner.get_sphere_info()
    for i, info in enumerate(sphere_info):
        print(f"\nSphere {i+1}:")
        print(f"  Center: [{info['center'][0]:.2f}, {info['center'][1]:.2f}, {info['center'][2]:.2f}]")
        print(f"  Radius: {info['radius']:.2f}")
    
    return pruner, gaussians


def test_pruning_logic():
    """Test pruning logic with points inside and outside spheres."""
    print("\n" + "="*60)
    print("Testing Pruning Logic")
    print("="*60)
    
    pruner, original_gaussians = test_sphere_fitting()
    
    # Test with original points (should all be inside)
    inside_mask = pruner.is_inside(original_gaussians)
    print(f"\nOriginal Gaussians inside spheres: {inside_mask.sum().item()}/{len(original_gaussians)}")
    
    # Create some points far outside
    outside_points = torch.tensor([
        [100, 100, 100],
        [-100, -100, -100],
        [50, 50, 50]
    ], device='cuda', dtype=torch.float32)
    
    outside_mask = pruner.is_inside(outside_points)
    print(f"Far-away points inside spheres: {outside_mask.sum().item()}/{len(outside_points)}")
    
    # Test pruning decision
    all_points = torch.cat([original_gaussians[:10], outside_points], dim=0)
    should_prune = pruner.should_prune(all_points)
    print(f"\nPoints to prune: {should_prune.sum().item()}/{len(all_points)}")
    print(f"Pruning mask: {should_prune.cpu().numpy()}")
    
    return pruner


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)
    
    cfg = omegaconf.DictConfig({
        'min_cluster_size': 100,
        'quantile': 0.98,
        'radius_multiplier': 1.1
    })
    
    pruner = SpherePruner(cfg)
    
    # Test 1: Empty input
    print("\nTest 1: Empty input")
    empty_gaussians = torch.empty(0, 3, device='cuda')
    pruner.fit(empty_gaussians)
    print(f"  Spheres fitted: {pruner.get_num_spheres()}")
    
    # Test 2: Single cluster
    print("\nTest 2: Single small cluster")
    small_cluster = torch.randn(50, 3, device='cuda')
    pruner.fit(small_cluster)
    print(f"  Spheres fitted: {pruner.get_num_spheres()}")
    
    # Test 3: Very sparse points
    print("\nTest 3: Very sparse points")
    sparse_points = torch.randn(10, 3, device='cuda') * 10
    pruner.fit(sparse_points)
    print(f"  Spheres fitted: {pruner.get_num_spheres()}")
    
    # Test 4: Collinear points
    print("\nTest 4: Collinear points")
    t = torch.linspace(0, 10, 100, device='cuda')
    collinear = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=1)
    pruner.fit(collinear)
    print(f"  Spheres fitted: {pruner.get_num_spheres()}")


def test_optimization_simulation():
    """Simulate Gaussian movement during optimization."""
    print("\n" + "="*60)
    print("Simulating Optimization with Pruning")
    print("="*60)
    
    cfg = omegaconf.DictConfig({
        'min_cluster_size': 100,
        'quantile': 0.98,
        'radius_multiplier': 1.1,
        'prune_every': 15
    })
    
    pruner = SpherePruner(cfg)
    
    # Initial Gaussians
    gaussians = create_test_gaussians()
    pruner.fit(gaussians)
    
    print(f"Initial Gaussians: {gaussians.shape[0]}")
    print(f"Bounding spheres: {pruner.get_num_spheres()}")
    
    # Simulate optimization iterations
    num_iterations = 50
    prune_every = cfg.prune_every
    
    active_gaussians = gaussians.clone()
    pruned_count = 0
    
    for iteration in range(1, num_iterations + 1):
        # Simulate random movement (gradient updates)
        noise = torch.randn_like(active_gaussians) * 0.1
        active_gaussians = active_gaussians + noise
        
        # Prune every N iterations
        if iteration % prune_every == 0:
            should_prune = pruner.should_prune(active_gaussians)
            num_to_prune = should_prune.sum().item()
            
            if num_to_prune > 0:
                active_gaussians = active_gaussians[~should_prune]
                pruned_count += num_to_prune
                
                print(f"Iteration {iteration}: Pruned {num_to_prune} Gaussians "
                      f"(Total pruned: {pruned_count}, Remaining: {active_gaussians.shape[0]})")
    
    print(f"\nFinal statistics:")
    print(f"  Started with: {gaussians.shape[0]} Gaussians")
    print(f"  Ended with: {active_gaussians.shape[0]} Gaussians")
    print(f"  Total pruned: {pruned_count} Gaussians")
    print(f"  Retention rate: {active_gaussians.shape[0] / gaussians.shape[0] * 100:.1f}%")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CL-Splats Pruning Module Tests")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, tests may fail or be slow")
        return
    
    try:
        # Test sphere fitting
        test_sphere_fitting()
        
        # Test pruning logic
        test_pruning_logic()
        
        # Test edge cases
        test_edge_cases()
        
        # Test optimization simulation
        test_optimization_simulation()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
