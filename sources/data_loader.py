import os
import random
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import clip

class CLIPDatasetWrapper(Dataset):
    def __init__(self, subset, class_to_name, transform=None, all_classes=None):
        self.subset = subset
        self.class_to_name = class_to_name
        self.transform = transform
        # 直接使用传入的 all_classes，不再从 subset 中读取
        self.all_classes = all_classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label_idx = self.subset[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # 使用传入的 all_classes 列表来获取文件夹 ID
        folder_name = self.all_classes[label_idx]
        
        # 映射逻辑：ID -> Human Label
        human_name = self.class_to_name.get(folder_name, folder_name).replace('_', ' ')
        
        text = clip.tokenize(f"a photo of a {human_name}")[0]
        return image, text, label_idx

def prepare_dataloaders(root_dir, batch_size=12, num_classes=50):
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    
    # 使用 ImageFolder 加载 (假设子文件夹名是 n02108915 这种 ID)
    full_raw_ds = ImageFolder(root=root_dir)
    
    # 加载 ID 到 英文名的映射 (ImageNet 标准映射)
    # 如果你没有这个 json，可以手动创建一个简单的字典测试
    label_map = {}
    try:
        # 示例：尝试从本地加载映射表
        with open("imagenet_class_index.json", "r") as f:
            raw_map = json.load(f)
            label_map = {v[0]: v[1] for k, v in raw_map.items()}
    except:
        print("警告: 未找到映射表，将使用文件夹名作为标签")

    # 随机选择 50 个类别
    all_classes = full_raw_ds.classes
    selected_classes = random.sample(all_classes, min(num_classes, len(all_classes)))
    selected_indices = [i for i, cls in enumerate(all_classes) if cls in selected_classes]
    
    # 筛选样本
    samples_indices = [i for i, (_, label) in enumerate(full_raw_ds.samples) if label in selected_indices]
    subset_ds = torch.utils.data.Subset(full_raw_ds, samples_indices)
    
    # 8:1:1 划分
    total = len(subset_ds)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    train_split, val_split, test_split = random_split(subset_ds, [train_size, val_size, test_size])
    
    # 在包装时传入 all_classes
    train_loader = DataLoader(
        CLIPDatasetWrapper(train_split, label_map, preprocess, all_classes=all_classes), 
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        CLIPDatasetWrapper(val_split, label_map, preprocess, all_classes=all_classes), 
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        CLIPDatasetWrapper(test_split, label_map, preprocess, all_classes=all_classes), 
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader, selected_classes
