"""
CL-Splats Dataset Downloader / 数据集下载脚本

Downloads the CL-Splats dataset from Hugging Face Hub.
从 Hugging Face Hub 下载 CL-Splats 数据集。

Usage / 使用方法:
    python download_data.py

Requirements / 依赖:
    pip install huggingface_hub

Available datasets / 可用数据集:
    - Real-World: Real captured scenes / 真实拍摄场景
    - Blender-Levels: Synthetic Blender scenes / Blender 合成场景
"""

from huggingface_hub import snapshot_download

# Download dataset from Hugging Face / 从 Hugging Face 下载数据集
snapshot_download(
    repo_id="ackermannj/cl-splats-dataset",  # Repository ID / 仓库 ID
    repo_type="dataset",                      # Repository type / 仓库类型
    local_dir="./cl-splats-dataset",          # Local save path / 本地保存路径
    
    # Specify folders to download (use "*" for all) / 指定下载的文件夹（使用 "*" 下载全部）
    allow_patterns="Real-World/*",
    
    # Patterns to exclude / 排除的文件模式
    ignore_patterns=[
        "*.pth",  # Exclude checkpoint files / 排除检查点文件
    ],
    
    local_dir_use_symlinks=False  # Don't use symlinks on Windows / Windows 上不使用符号链接
)

print("Download complete! / 下载完成！")
print("Dataset saved to: ./cl-splats-dataset")
