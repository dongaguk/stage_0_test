import clip
import torch

def get_clip_model(model_name="ViT-B/32", device="cuda"):
    model, _ = clip.load(model_name, device=device, jit=False)
    
    # 策略：冻结大部分参数，只更新 Vision Encoder 的最后 2 个 Transformer 层
    # 以及 Text Encoder 的最后 1 层。这可以保留预训练特征同时适应新领域。
    for param in model.parameters():
        param.requires_grad = False
        
    # 开启 Vision Transformer 后部
    for param in model.visual.transformer.resblocks[-2:].parameters():
        param.requires_grad = True
        
    # 开启 Text Transformer 后部
    for param in model.transformer.resblocks[-1:].parameters():
        param.requires_grad = True
        
    return model.to(device)