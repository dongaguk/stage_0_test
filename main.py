import torch
import torch.optim as optim
from sources.data_loader import prepare_dataloaders
from sources.model import get_clip_model
from sources.train import run_epoch

# --- 配置 ---
# 注意：Mini-ImageNet 路径通常指向包含 'train' 文件夹的目录
DATA_ROOT = "./dataset/mini_imagenet/" 
BATCH_SIZE = 12
LR = 1e-5  # ImageNet 数据更复杂，稍微提升一点学习率
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"开始 Mini-ImageNet 训练。设备: {DEVICE}")
    
    # 1. 加载数据 (选50个类)
    train_loader, val_loader, test_loader, _ = prepare_dataloaders(DATA_ROOT, BATCH_SIZE, num_classes=50)
    
    # 2. 获取模型 (之前已修改 model.py 加入 .float())
    model = get_clip_model("ViT-B/32", DEVICE)
    
    # 3. 优化器 (只更新 requires_grad=True 的参数)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=(DEVICE != "cpu"))
    
    best_acc = 0
    for epoch in range(EPOCHS):
        # 训练
        t_loss, t_acc = run_epoch(model, train_loader, optimizer, scaler, DEVICE, is_train=True)
        # 验证
        v_loss, v_acc = run_epoch(model, val_loader, None, None, DEVICE, is_train=False)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.4f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "mini_imagenet_best.pt")
            print("保存当前最佳模型。")

    # 4. 最终测试
    model.load_state_dict(torch.load("mini_imagenet_best.pt", weights_only=True))
    _, final_acc = run_epoch(model, test_loader, None, None, DEVICE, is_train=False)
    print(f"\n测试集最终准确率: {final_acc:.4f}")

if __name__ == "__main__":
    main()

