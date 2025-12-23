import os
import requests
import tarfile
import zipfile
import json
from tqdm import tqdm

def download_file(url, save_path):
    """带进度条的下载工具"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(save_path))
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def setup_mini_imagenet():
    # 1. 定义路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_data_dir = os.path.join(base_dir, "dataset", "mini_imagenet")
    json_path = os.path.join(base_dir, "imagenet_class_index.json")
    
    # 创建数据集目录
    os.makedirs(target_data_dir, exist_ok=True)

    # 2. 下载 JSON 映射表 (存放在根目录)
    json_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    if not os.path.exists(json_path):
        print(f"--- 正在下载类别映射 JSON ---")
        download_file(json_url, json_path)
    else:
        print(f"JSON 映射表已存在: {json_path}")

    # 3. 下载 Mini-ImageNet (此处使用常用的学术镜像链接)
    # 注意：如果此链接失效，建议手动从 Kaggle 下载并将 zip 放入 .\dataset\
    dataset_url = "https://github.com/renmengye/few-shot-ssl-public/raw/master/data/mini-imagenet.tar.gz" # 示例链接
    # 备用链接 (Kaggle 版本通常是 zip): https://raw.githubusercontent.com/yaoyao-liu/mini-imagenet-tools/master/data/mini-imagenet.zip
    
    archive_save_path = os.path.join(base_dir, "dataset", "mini-imagenet.tar.gz")

    if not os.listdir(target_data_dir):  # 如果目录为空则下载
        print(f"\n--- 正在下载 Mini-ImageNet 数据集 (约 1GB+) ---")
        try:
            download_file(dataset_url, archive_save_path)
            
            # 4. 解压
            print(f"正在解压到 {target_data_dir}...")
            if archive_save_path.endswith(".tar.gz"):
                with tarfile.open(archive_save_path, "r:gz") as tar:
                    tar.extractall(path=target_data_dir)
            elif archive_save_path.endswith(".zip"):
                with zipfile.ZipFile(archive_save_path, 'r') as zip_ref:
                    zip_ref.extractall(target_data_dir)
            
            # 清理压缩包
            os.remove(archive_save_path)
            print("解压完成并已清理压缩包。")
        except Exception as e:
            print(f"下载过程中出错: {e}")
            print("建议从 Kaggle 手动下载 mini-imagenet.zip 并放入 ./dataset/ 目录后手动解压。")
    else:
        print(f"数据集已存在于: {target_data_dir}")

if __name__ == "__main__":
    setup_mini_imagenet()