"""
CL-Splats Training Entry Point.

This script provides the main entry point for training CL-Splats models
using Hydra for configuration management.
"""

import os
import sys

import hydra
from loguru import logger
import omegaconf
import wandb
import torch

from clsplats.trainer import CLSplatsTrainer
from clsplats.dataset import CLSplatsDataset


def setup_logging(cfg: omegaconf.DictConfig) -> None:
    """Configure logging based on config."""
    log_level = cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    log_dir = cfg.get("output_dir", "outputs")
    if cfg.get("log_to_file", True):
        os.makedirs(log_dir, exist_ok=True)
        logger.add(
            os.path.join(log_dir, "train.log"),
            level=log_level,
            rotation="100 MB"
        )


def setup_wandb(cfg: omegaconf.DictConfig) -> None:
    """Initialize Weights & Biases logging."""
    wandb_mode = cfg.get("wandb_mode", "disabled")
    
    if wandb_mode == "disabled":
        return
    
    wandb.init(
        project=cfg.get("wandb_project", "cl-splats"),
        name=cfg.get("wandb_run_name", None),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_mode,
    )
    logger.info(f"Initialized wandb in {wandb_mode} mode")


def load_data(cfg: omegaconf.DictConfig) -> CLSplatsDataset:
    """Load dataset based on configuration."""
    dataset_cfg = cfg.get("dataset", {})
    data_path = dataset_cfg.get("path", None)
    
    if data_path is None:
        logger.error("No dataset path specified! Set dataset.path in config or command line.")
        return None
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset path does not exist: {data_path}")
        return None
    
    logger.info(f"Loading dataset from {data_path}")
    
    dataset = CLSplatsDataset(
        path=data_path,
        resolution_scale=dataset_cfg.get("resolution", 1.0),
        white_background=dataset_cfg.get("white_background", False),
        eval_mode=dataset_cfg.get("eval", False),
    )
    
    logger.info(f"Dataset loaded: {dataset.get_num_timesteps()} timesteps, type={dataset.dataset_type}")
    return dataset


@hydra.main(version_base=None, config_path="../configs", config_name="cl-splats")
def main(cfg: omegaconf.DictConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging(cfg)
    setup_wandb(cfg)
    
    logger.info("=" * 60)
    logger.info("CL-Splats Training")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type != "cuda":
        logger.warning("CUDA not available! Training will be very slow or may not work.")
    
    # Load dataset
    dataset = load_data(cfg)
    if dataset is None:
        return
    
    # Initialize trainer
    trainer = CLSplatsTrainer(cfg)
    
    # Load checkpoint if resuming
    checkpoint_path = cfg.get("resume_from", None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        # Initialize Gaussians from first timestep point cloud
        scene_info = dataset.get_scene_info(0)
        if scene_info.point_cloud is not None:
            cameras = dataset.get_cameras(0)
            spatial_lr_scale = scene_info.nerf_normalization["radius"]
            trainer.initialize_from_point_cloud(
                scene_info.point_cloud, 
                cameras, 
                spatial_lr_scale
            )
        else:
            logger.error("No point cloud found in dataset!")
            return
    
    # Training loop over timesteps
    train_cfg = cfg.get("train", {})
    start_time = train_cfg.get("start_time", 0)
    num_times = min(train_cfg.get("num_times", 10), dataset.get_num_timesteps())
    
    # If resumed, start from next timestep
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_time = max(start_time, trainer.timestep + 1)
    
    for timestep in range(start_time, num_times):
        logger.info(f"\n{'='*60}")
        logger.info(f"Timestep {timestep}/{num_times-1}")
        logger.info(f"{'='*60}")
        
        # Prepare data for this timestep
        trainer.prepare_timestep(timestep, dataset)
        
        # Train
        stats = trainer.train()
        
        # Log history
        trainer.log_history()
        
        # Save checkpoint
        save_interval = cfg.get("save_interval", 1)
        output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        if (timestep + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_t{timestep}.pt")
            trainer.save_checkpoint(ckpt_path)
        
        logger.info(f"Timestep {timestep} complete: loss={stats['avg_loss']:.6f}, gaussians={stats['num_gaussians']}")
    
    # Save final checkpoint and PLY
    output_dir = cfg.get("output_dir", "outputs")
    final_ckpt_path = os.path.join(output_dir, "checkpoint_final.pt")
    trainer.save_checkpoint(final_ckpt_path)
    
    final_ply_path = os.path.join(output_dir, "point_cloud_final.ply")
    trainer.gaussians.save_ply(final_ply_path)
    logger.info(f"Saved final point cloud to {final_ply_path}")
    
    # Cleanup
    if wandb.run is not None:
        wandb.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
