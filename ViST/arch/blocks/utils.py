import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import numpy as np
from PIL import Image
from torchvision.transforms import Resize

#================================utils================================

def print_trainable_parameters(model, detail=False, nn=""):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if detail:
            print(f"layer name: {name}, shape: {param.shape}, numel: {param.numel()}, requires_grad: {param.requires_grad}")
    if all_param > 0:
        print(f"{nn} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    else:
        print(nn,"zero params")
    
    return trainable_params, all_param

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)

def check_numerical_stability(self, tensor, name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or Inf values")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean()}, Std: {tensor.std()}")
        print(f"Min: {tensor.min()}, Max: {tensor.max()}")
        return False
    return True

def test(tensor):
    print("shape:",tensor.shape)
    print("avg:",tensor.mean())
    print("std:",tensor.std())
    print("min:",tensor.min())
    print("max",tensor.max())
    print("NaN?",torch.isnan(tensor).any())
    print("Inf?",torch.isinf(tensor).any())
    print("grad:",tensor.grad)

def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)

def Normalization(x, norm_const=1.):
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev /= norm_const
    x = x / stdev
    return (x - means) / stdev, means, stdev

def Denormalization(y, means, std, padding=0):
    y = y * (std.repeat(1, padding, 1))
    y = y + (means.repeat(1, padding, 1))
    return y

def reshape_to_image(embedding, channels=3, height=16, width=16):
    B, D = embedding.shape
    assert D == channels * height * width, "embedding dimension does not match image size"
    images = embedding.view(B, channels, height, width)
    return images

def reshape_from_image(images):
    B, C, H, W = images.shape
    embedding = images.view(B, C * H * W)
    return embedding

def calculate_lags(x_enc, top_k):
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags

def generate_prompt(x, description, time_step, top_k=5):
    """
    Generate a concise prompt with statistical properties of node features
    
    Args:
        x: Tensor of shape [N, D] - batch, nodes, features
        description: Dataset description string
        time_step: Current time step index
        top_k: Number of top nodes to consider
    
    Returns:
        List of prompts with statistical information for each batch
    """
    # Handle input dimensions - ensure we have 3D tensor [B, N, D]
    if x.dim() == 4:  # If shape is [B, 1, N, D]
        x = x.squeeze(1)
    
    batch_size, num_nodes, feature_dim = x.shape
    
    # Calculate key statistics across nodes (dim=1)
    min_values = torch.min(x, dim=1)[0]        # [B, D]
    max_values = torch.max(x, dim=1)[0]        # [B, D]
    mean_values = torch.mean(x, dim=1)         # [B, D]
    std_values = torch.std(x, dim=1)           # [B, D]
    
    # Calculate median (more robust than mean)
    sorted_values, _ = torch.sort(x, dim=1)
    median_idx = num_nodes // 2
    median_values = sorted_values[:, median_idx, :]  # [B, D]
    
    # Identify top nodes with highest average values
    node_avg_values = torch.mean(x, dim=2)  # [B, N]
    top_values, top_indices = torch.topk(node_avg_values, k=min(top_k, num_nodes), dim=1)  # [B, top_k]
    
    # Generate concise prompts for each batch
    prompts = []
    for b in range(batch_size):
        # Format statistics as concise strings (limit decimal places)
        stats = {
            "min": [f"{v:.2f}" for v in min_values[b].tolist()],
            "max": [f"{v:.2f}" for v in max_values[b].tolist()],
            "mean": [f"{v:.2f}" for v in mean_values[b].tolist()],
            "median": [f"{v:.2f}" for v in median_values[b].tolist()],
            "std": [f"{v:.2f}" for v in std_values[b].tolist()]
        }
        
        # Identify outliers (simplified)
        outlier_threshold = mean_values[b] + 2 * std_values[b]
        outlier_count = torch.sum(x[b] > outlier_threshold.unsqueeze(0)).item()
        
        # Format top nodes info (concisely)
        top_nodes = ", ".join([f"N{top_indices[b, k].item()}:{top_values[b, k].item():.2f}" 
                              for k in range(min(3, top_k, num_nodes))])
        
        # Build a concise prompt
        prompt = (
            f"<|prompt|> Dataset: {description[:30]}... "
            f"Time: {time_step}: "
            f"min={stats['min'][0]}, max={stats['max'][0]}, "
            f"mean={stats['mean'][0]}, median={stats['median'][0]}, "
            f"std={stats['std'][0]}, "
            f"outliers={outlier_count}, "
            f"top=[{top_nodes}] <|/prompt|>"
        )
        if feature_dim > 1:
            feature_summary = f"Features: {feature_dim}, "
            prompt = prompt.replace("<|prompt|> ", f"<|prompt|> {feature_summary}")
        prompts.append(prompt)
    return prompts

def save_images(save_dir, images, batch_idx):
    for i, img_tensor in enumerate(images):
        img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
        img_tensor = img_tensor.astype(np.uint8)
        img = Image.fromarray(img_tensor)
        img.save(os.path.join(save_dir, f"image_{batch_idx}_{i}.png"))

def load_domain_text(domain):
    with open(f"./prompts/{domain}.txt", "r") as f:
        return f.read()