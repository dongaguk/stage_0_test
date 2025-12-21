import torch
import torch.nn as nn

def run_epoch(model, loader, optimizer, scaler, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    correct = 0
    total_samples = 0
    
    # 对比损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    for images, texts, _ in loader:
        images, texts = images.to(device), texts.to(device)
        batch_size = images.size(0)
        
        # 对应标签：在batch内，第i张图对应第i个文本
        ground_truth = torch.arange(batch_size, device=device)
        
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast():
                logits_per_image, logits_per_text = model(images, texts)
                loss = (loss_fn(logits_per_image, ground_truth) + 
                        loss_fn(logits_per_text, ground_truth)) / 2
            
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_loss += loss.item() * batch_size
        # 计算准确率 (图像匹配到正确文本的概率)
        preds = logits_per_image.argmax(dim=-1)
        correct += (preds == ground_truth).sum().item()
        total_samples += batch_size
        
    return total_loss / total_samples, correct / total_samples