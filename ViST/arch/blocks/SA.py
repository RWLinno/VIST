import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    """Self-attention mechanism to highlight important regions"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qkv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(1, 1, kernel_size=1)
    
    def forward(self, x):
        """Apply self-attention"""
        b, c, h, w = x.shape
        
        # Get query, key, value
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention calculation
        q = q.view(b, 1, -1).transpose(-2, -1)  # [b, h*w, 1]
        k = k.view(b, 1, -1)  # [b, 1, h*w]
        v = v.view(b, 1, -1).transpose(-2, -1)  # [b, h*w, 1]
        
        # Calculate attention scores
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to value
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).view(b, 1, h, w)
        
        # Final projection
        out = self.to_out(out)
        return out
