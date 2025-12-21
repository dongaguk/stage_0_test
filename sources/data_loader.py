import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import clip

class ImageNetSubset(Dataset):
    def __init__(self, root_dir, num_classes=50, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. 筛选50个标签
        all_classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if len(all_classes) < num_classes:
            raise ValueError(f"数据集类别不足 {num_classes}")
        
        self.selected_classes = random.sample(all_classes, num_classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.selected_classes)}
        
        # 2. 收集样本
        self.samples = []
        for cls_name in self.selected_classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # CLIP Tokenizer: 将类别名转为文本特征输入
        class_name = self.selected_classes[label_idx].replace('_', ' ')
        text = clip.tokenize(f"a photo of a {class_name}")[0]
        
        return image, text, label_idx

def prepare_dataloaders(root_dir, batch_size=12):
    # 加载CLIP自带的预处理（包含Resize, CenterCrop, Normalize）
    _, preprocess = clip.load("ViT-B/32", device="cpu")
    
    full_dataset = ImageNetSubset(root_dir, num_classes=50, transform=preprocess)
    
    # 3. 8:1:1 划分
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, full_dataset.selected_classes