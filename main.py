import torch
from sources.data_loader import prepare_dataloaders
from sources.model import get_clip_model
from sources.train import run_epoch

# 配置
DATA_ROOT = "./dataset/imageNet/" # 修改为你真实的ImageNet路径
BATCH_SIZE = 12
LR = 5e-6
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Step 1: 准备数据
    train_loader, val_loader, test_loader, class_names = prepare_dataloaders(DATA_ROOT, BATCH_SIZE)
    print(f"成功加载50个类别。训练集大小: {len(train_loader.dataset)}")

    # Step 2: 准备模型
    model = get_clip_model("ViT-B/32", DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0

    # Step 4: 训练/推理循环
    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, scaler, DEVICE, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, None, DEVICE, is_train=False)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_clip_model.pt")

    # Step 4e: 在Best Epoch上计算测试集
    print("\n--- 正在加载最优模型并在测试集上进行最终评估 ---")
    model.load_state_dict(torch.load("best_clip_model.pt"))
    test_loss, test_acc = run_epoch(model, test_loader, None, None, DEVICE, is_train=False)
    print(f"测试集最终准确率 (Test Acc): {test_acc:.4f}")

if __name__ == "__main__":
    main()