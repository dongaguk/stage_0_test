import torch
import torch.nn as nn

def run_epoch(model, loader, optimizer, scaler, device, is_train=True):
    model.train() if is_train else model.eval()
    
    total_loss = 0
    correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()
    
    for images, texts, _ in loader:
        images, texts = images.to(device), texts.to(device)
        batch_size = images.size(0)
        
        # Ground Truth 是对角阵 (1, 2, 3...)
        ground_truth = torch.arange(batch_size, device=device)
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.split(':')[0], enabled=(device!="cpu")):
                logits_per_image, logits_per_text = model(images, texts)
                
                loss = (loss_fn(logits_per_image, ground_truth) + 
                        loss_fn(logits_per_text, ground_truth)) / 2
            
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_loss += loss.item() * batch_size
        preds = logits_per_image.argmax(dim=-1)
        correct += (preds == ground_truth).sum().item()
        total_samples += batch_size
        
    return total_loss / total_samples, correct / total_samples