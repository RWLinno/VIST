'''
Prompts for Coding:
1. Don't modify any English annotation, translate the Chinese annotation into English.
2. Don't delete any annotation for printing tensor shape

### 3.1 Multi-view Vision Transformation
将时空数据转换为结构化的视觉模态表示，保留其中的时间依赖性、空间依赖性和特征关系。
x: [B, T, N, D] -> visual_representation: [B, 3, H, W]

或许对于其他性质也能concat进去，然后使用图像模态做性质挖掘 -> : [B, 3->X, H, W]

### 3.2 prompts empowered Spatial-Temporal Alignment
使用大语言模型编码隐藏统计信息，生成上下文感知的时空表示。
x: [B, T, N, D] -> temporal_representation: [B, tem_dim, N, 1]
x: [B, T, N, D] -> spatial_representation: [B, spa_dim, N, 1]
x: [B, T, N, D] -> correlation_representation: [B, cor_dim, N, 1]
+ x: [B, T, N, D] -> text_representation: [B, horizon, llm_dim]
-> Context_Aware_ST_representation: [B, tem_dim + spa_dim + cor_dim, N, 1]

### 3.3 Cross-modal Attention Fusion Mechanism
使用跨模态注意力机制将多元时序模态，图像模态，文本模态进行融合，得到最终的预测结果。
visual_representation: [B, 3, H, W]
Multivariate_ST_representation: [B, tem_dim + spa_dim + cor_dim, N, 1]
text_representation: [B, horizon, llm_dim]
-> fused_representation: [B, horizon, N, output_dim]
'''

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft, irfft
from torchvision.transforms import Resize
from PIL import Image
import einops
from contextlib import contextmanager
from easydict import EasyDict
from types import SimpleNamespace
from .blocks.MLP import MultiLayerPerceptron
from .blocks.utils import *
from .blocks.Text_Encoder import *
from .blocks.MultimodalFusion import *
from scipy import signal
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
import hashlib

def normalize_minmax(x, eps=1e-8):
    """Min-max normalization to scale values to [0,1] range"""
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < eps:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)

class MultiPerspectiveVisualEncoder(nn.Module):
    """
    Transforms spatio-temporal data into structured multi-perspective visual representations
    
    This encoder creates visual representations that capture various properties:
    - Temporal correlations
    - Spatial dependencies
    - Feature relationships
    """
    def __init__(self, configs):
        super().__init__()
        # Configuration parameters
        self.image_size = configs.get('image_size', 64)
        self.periodicity = configs.get('periodicity', 24)
        self.interpolation = configs.get('interpolation', 'bilinear')
        self.save_path = configs.get('save_path', "images_output")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        
        # Set up resize operation with proper interpolation
        interp_map = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }
        self.resize_fn = safe_resize((self.image_size, self.image_size), 
                                    interpolation=interp_map[self.interpolation])
        
    def forward(self, hidden_state, adj_mx=None, save_images=False):
        """
        Transform hidden state to visual representation

        Args:
            hidden_state: Tensor of shape [B, emb_dim, N, 1]
            adj_mx: Optional adjacency matrix for spatial relationships
            save_images: Whether to save visualizations as image files

        Returns:
            Tensor of shape [B, 3, image_size, image_size] (image modality)
        """
        # Generate three different views as RGB channels
        temporal_channel = self.generate_temporal_view(hidden_state, adj_mx)
        spatial_channel = self.generate_spatial_view(hidden_state, adj_mx)
        correlation_channel = self.generate_correlation_view(hidden_state, adj_mx)
        
        # Combine channels into an RGB image
        visual_repr = torch.cat([temporal_channel, spatial_channel, correlation_channel], dim=1)
        
        # Ensure output has the right dimensions
        if visual_repr.shape[2] != self.image_size or visual_repr.shape[3] != self.image_size:
            visual_repr = F.interpolate(
                visual_repr,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Save visualizations if requested
        if save_images:
            self.save_visualization(visual_repr[0], "combined_view")
        
        return visual_repr
    
    def generate_temporal_view(self, x, adj_mx=None):
        """Generates visualization focused on temporal patterns"""
        B, emb_dim, N, _ = x.shape
        device = x.device
        
        # Direct approach: Create a 2D representation by reshaping the temporal dimension
        # First, extract and transpose the data to [B, N, emb_dim]
        x_reshaped = x.squeeze(-1).permute(0, 2, 1)
        
        # Calculate dimensions for creating a 2D grid
        # Use direct interpolation instead of trying to map exact dimensions
        temporal_view = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        for b in range(B):
            # Average across nodes to get temporal patterns
            time_series = x_reshaped[b].mean(dim=0)  # [emb_dim]
            
            # Create a square grid by reshaping
            grid_size = int(math.ceil(math.sqrt(emb_dim)))
            grid = torch.zeros(grid_size, grid_size, device=device)
            
            # Fill the grid with temporal values
            for i in range(min(emb_dim, grid_size * grid_size)):
                row, col = i // grid_size, i % grid_size
                grid[row, col] = time_series[i]
            
            # Resize to target dimensions
            temporal_view[b, 0] = F.interpolate(
                grid.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        return normalize_minmax(temporal_view)
    
    def generate_spatial_view(self, x, adj_mx=None):
        """Generates visualization focused on spatial relationships"""
        B, emb_dim, N, _ = x.shape
        device = x.device
        
        # Create spatial relationship visualization
        spatial_view = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        for b in range(B):
            # Create node feature matrix by averaging across temporal dimension
            node_features = x[b].mean(dim=0).squeeze(-1)  # [N]
            
            # Create a spatial layout based on node features
            grid_size = int(math.ceil(math.sqrt(N)))
            grid = torch.zeros(grid_size, grid_size, device=device)
            
            # Place each node on the grid
            for i in range(min(N, grid_size*grid_size)):
                row, col = i // grid_size, i % grid_size
                grid[row, col] = node_features[i]
            
            # Resize to target dimensions
            spatial_view[b, 0] = F.interpolate(
                grid.unsqueeze(0).unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        return normalize_minmax(spatial_view)
    
    def generate_correlation_view(self, x, adj_mx=None):
        """Generates visualization of correlation patterns between features"""
        B, emb_dim, N, _ = x.shape
        device = x.device
        
        # Create correlation view
        correlation_view = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        for b in range(B):
            # Extract data for this batch
            x_b = x[b].squeeze(-1)  # [emb_dim, N]
            
            # Calculate temporal correlation matrix
            x_mean = x_b.mean(dim=1, keepdim=True)
            x_centered = x_b - x_mean
            x_norm = torch.norm(x_centered, dim=1, keepdim=True) + 1e-8
            x_normalized = x_centered / x_norm
            
            # Compute correlation matrix
            correlation = torch.mm(x_normalized, x_normalized.t())  # [emb_dim, emb_dim]
            
            # Resize to target dimensions
            correlation_view[b, 0] = F.interpolate(
                correlation.unsqueeze(0).unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        return normalize_minmax(correlation_view)
    
    def save_visualization(self, image_tensor, name):
        """Save visualization as an image file"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            
        # Convert tensor to numpy array
        image_np = image_tensor.detach().cpu().numpy()
        
        # Format based on channels
        if image_np.shape[0] == 3:  # RGB format
            image_np = np.transpose(image_np, (1, 2, 0))
        else:  # Grayscale format
            image_np = image_np[0]
            
        # Scale to 0-255 for image format
        image_np = (image_np * 255).astype(np.uint8)
        
        # Create and save image
        if len(image_np.shape) == 3:
            img = Image.fromarray(image_np, mode='RGB')
        else:
            img = Image.fromarray(image_np, mode='L')
            
        img.save(os.path.join(self.save_path, f"{name}.png"))

class WaveletTransform(nn.Module):
    """ Performs multi-level wavelet decomposition for multi-scale analysis"""
    def __init__(self, levels=3, wavelet='db4'):
        super().__init__()
        self.levels = levels
        self.wavelet = wavelet
    
    def forward(self, x):
        """Apply wavelet transform to create multi-scale representation"""
        B, C, H, W = x.shape
        
        # Processing on CPU as PyWavelets doesn't support GPU
        x_np = x.detach().cpu().numpy()
        result = np.zeros_like(x_np)
        
        for b in range(B):
            for c in range(C):
                # Apply wavelet decomposition
                coeffs = self._decompose(x_np[b, c])
                # Reconstruct with enhanced details
                result[b, c] = self._reconstruct(coeffs)
        
        # Back to tensor
        return torch.from_numpy(result).to(x.device)
    
    def _decompose(self, img):
        """Simplified wavelet decomposition"""
        # Simple approximation without PyWavelets dependency
        result = img.copy()
        h, w = img.shape
        
        # Low-pass filter approximation
        low_pass = np.array([[0.25, 0.5, 0.25]])
        
        # High-pass filter approximation
        high_pass = np.array([[-0.25, 0.5, -0.25]])
        
        for level in range(self.levels):
            scale_h = h // (2 ** level)
            scale_w = w // (2 ** level)
            
            if min(scale_h, scale_w) <= 2:
                break
                
            # Apply filters
            # This is a very simplified version of wavelet transform
            temp = result[:scale_h, :scale_w].copy()
            
            # Low-pass in both directions (approximation)
            approx = signal.convolve2d(temp, low_pass.T @ low_pass, mode='same')
            
            # Store back
            result[:scale_h//2, :scale_w//2] = approx[::2, ::2]
        
        return result
    
    def _reconstruct(self, coeffs):
        """Reconstruct with enhanced details"""
        # For simplicity, just return the coefficients
        # In a real implementation, this would properly reconstruct
        return coeffs

class SpectralTransform(nn.Module):
    """Transforms data to frequency domain for spectral analysis"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Apply spectral transform"""
        # Get shape
        b, c, h, w = x.shape
        
        # Apply 2D FFT
        x_complex = torch.fft.fft2(x)
        
        # Get amplitude spectrum
        amplitude = torch.abs(x_complex)
        
        # Shift zero frequency to center
        amplitude = torch.fft.fftshift(amplitude, dim=(-2, -1))
        
        # Normalize to [0, 1]
        amplitude = normalize_minmax(amplitude)
        
        return amplitude

class EfficientSpatioTemporalVisionEncoder(nn.Module):
    """
    Encodes spatio-temporal data into multiple visual representations
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        self.periodicity = config['periodicity']
        self.interpolation = config['interpolation']
        self.save_images = getattr(config, 'save_images', True)
        
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[self.interpolation]
        self.input_resize = safe_resize((self.image_size, self.image_size), 
                                        interpolation=interpolation)

    def spatio_temporal_segmentation(self, x):
        """
        Transforms spatio-temporal data into a segmented image representation
        Args:
            x: Tensor of shape [B, T, N, D] where B is batch size, T is time steps,
               N is number of nodes, D is feature dimension
        Returns:
            Tensor of shape [B, 1, image_size, image_size]
        """
        B, T, N, D = x.shape
        
        # Reshape to batch each node separately
        x_reshaped = x.reshape(B * N, T, D)
        x_reshaped = einops.rearrange(x_reshaped, 'bn t d -> bn d t')
        
        # Pad if needed
        pad_left = 0
        if T % self.periodicity != 0:
            pad_left = self.periodicity - T % self.periodicity
        x_pad = F.pad(x_reshaped, (pad_left, 0), mode='replicate')
        
        # Reshape to 2D grid
        x_2d = einops.rearrange(
            x_pad,
            'bn d (p f) -> bn d f p',
            p=x_pad.size(-1) // self.periodicity,
            f=self.periodicity
        )
        
        # Resize to target image size
        x_resize = F.interpolate(
            x_2d,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize each channel separately
        x_channels = []
        for i in range(D):
            channel = x_resize[:, i:i+1]
            channel = normalize_minmax(channel)
            x_channels.append(channel)
        
        x_combined = torch.stack(x_channels, dim=1).mean(dim=1)
        
        # Reshape back to batch dimension
        x_final = x_combined.reshape(B, N, 1, self.image_size, self.image_size)
        
        # Average across nodes to get one image per batch
        # Alternative: could create a spatial grid layout preserving node relationships
        x_final = x_final.mean(dim=1)
        
        # Add grid lines for visual clarity
        grid_size = self.image_size // 8
        grid = torch.ones_like(x_final)
        grid[:, :, ::grid_size] = 0.95 
        grid[:, :, :, ::grid_size] = 0.95 
        x_final = x_final * grid
        
        return x_final

    def gramian_angular_field(self, x):
        """
        Creates Gramian Angular Field representation of spatio-temporal data
        Args:
            x: Tensor of shape [B, T, N, D]
        Returns:
            Tensor of shape [B, 1, image_size, image_size]
        """
        B, T, N, D = x.shape
        
        # Flatten spatial and feature dimensions
        x_flat = x.reshape(B, T, -1)  # [B, T, N*D]
        
        # Normalize to [-1, 1]
        x_norm = normalize_minmax(x_flat) * 2 - 1
        
        # Calculate angle
        theta = torch.arccos(x_norm.clamp(-1 + 1e-6, 1 - 1e-6))
        
        # Create GAF for each batch
        gaf = torch.zeros(B, 1, T, T, device=x.device)
        for b in range(B):
            # 修改这里：确保维度匹配
            theta_b = theta[b]  # [T, N*D]
            cos_sum = torch.zeros(T, T, device=x.device)
            
            # 计算每个时间步之间的余弦和
            for i in range(T):
                for j in range(T):
                    cos_sum[i, j] = torch.cos(theta_b[i] + theta_b[j]).mean()
            
            gaf[b, 0] = normalize_minmax(cos_sum)
        
        # Resize to target dimensions
        gaf = F.interpolate(gaf, size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False)
        
        return gaf

    def recurrence_plot(self, x):
        """
        Creates Recurrence Plot representation of spatio-temporal data
        Args:
            x: Tensor of shape [B, T, N, D]
        Returns:
            Tensor of shape [B, 1, image_size, image_size]
        """
        B, T, N, D = x.shape
        
        # Flatten spatial and feature dimensions
        x_flat = x.reshape(B, T, -1)  # [B, T, N*D]
        
        rp = torch.zeros(B, 1, T, T, device=x.device)
        
        for b in range(B):
            x_i = x_flat[b].unsqueeze(1)  # [T, 1, N*D]
            x_j = x_flat[b].unsqueeze(0)  # [1, T, N*D]
            distances = torch.norm(x_i - x_j, dim=2)
            rp[b, 0] = torch.exp(-distances**2 / 2)
        
        rp = normalize_minmax(rp)
        rp = F.interpolate(rp, size=(self.image_size, self.image_size),
                           mode='bilinear', align_corners=False)
        
        return rp
    
    def spatial_adjacency_image(self, x, adj_mx=None):
        """
        Creates an image representation that captures spatial relationships
        Args:
            x: Tensor of shape [B, T, N, D]
            adj_mx: Optional adjacency matrix of shape [N, N]
        Returns:
            Tensor of shape [B, 1, image_size, image_size]
        """
        B, T, N, D = x.shape
        
        if adj_mx is None:
            # If no adjacency matrix provided, create a simple one based on feature similarity
            # Aggregate features across time
            x_agg = x.mean(dim=1)  # [B, N, D]
            
            # Calculate pairwise similarities
            adj_mx = torch.zeros(B, N, N, device=x.device)
            for b in range(B):
                for i in range(N):
                    for j in range(N):
                        adj_mx[b, i, j] = F.cosine_similarity(
                            x_agg[b, i].unsqueeze(0), 
                            x_agg[b, j].unsqueeze(0), 
                            dim=1
                        )
        else:
            # If adjacency matrix provided, expand to batch dimension
            adj_mx = adj_mx.unsqueeze(0).repeat(B, 1, 1)
        
        # Normalize and reshape
        adj_mx = normalize_minmax(adj_mx).unsqueeze(1)  # [B, 1, N, N]
        
        # Resize to target image size
        adj_mx = F.interpolate(adj_mx, size=(self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        
        return adj_mx

    def normalize(self, x):
        """Min-max normalization to [0,1] range"""
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x

    @torch.no_grad()
    def save_images(self, images, method, batch_idx):
        save_dir = "image_visualization"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy()
            if img_tensor.shape[0] == 1:  # grayscale
                img_tensor = img_tensor[0]
            else:  # RGB
                img_tensor = img_tensor.transpose(1, 2, 0)
            img_tensor = (img_tensor * 255).clip(0, 255).astype(np.uint8)
            if len(img_tensor.shape) == 2:  # grayscale
                img = Image.fromarray(img_tensor, mode='L')
            else:  # RGB
                img = Image.fromarray(img_tensor, mode='RGB')
            img.save(os.path.join(save_dir, f"image_{method}_{batch_idx}_{i}.png"))

    def forward(self, x, adj_mx=None, method='efficient', save_images=False):
        """
        Forward pass of the spatio-temporal pixel encoder
        Args:
            x: Input tensor of shape [B, T, N, D]
            adj_mx: Optional adjacency matrix
            method: Encoding method ('seg', 'gaf', 'rp', 'spatial', 'efficient')
            save_images: Whether to save debug images
        Returns:
            Tensor of shape [B, C, H, W]
        """
        # Normalize input
        x = self.normalize(x)
        
        if method == 'seg':
            #print("Spatio-temporal segmentation")
            output = self.spatio_temporal_segmentation(x)
        elif method == 'gaf':
            #print("Gramian Angular Field")
            output = self.gramian_angular_field(x)
        elif method == 'rp':
            #print("Recurrence Plot")
            output = self.recurrence_plot(x)
        elif method == 'spatial':
            #print("Spatial Adjacency Image")
            output = self.spatial_adjacency_image(x, adj_mx)
        elif method == 'efficient':
            #print("Efficient spatio-temporal embedding")
            output = self.efficient_spatio_temporal_embedding(x, adj_mx)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if save_images:
            self.save_images(output, method, 0)
        
        return output

    def efficient_spatio_temporal_embedding(self, x, adj_mx=None):
        """
        Creates an efficient spatio-temporal embedding by transforming 4D tensor [B,T,N,D] 
        into a 2D image representation with 3 channels [B, 3, H, W].
        
        This method is inspired by TimesNet's approach to transform 1D time series into 2D spaces,
        but optimized for spatio-temporal data with graph structure.
        
        Args:
            x: Input tensor of shape [B, T, N, D] (batch, time, nodes, features)
            adj_mx: Optional adjacency matrix for spatial relationships
            
        Returns:
            Tensor of shape [B, 3, image_size, image_size]
        """
        B, T, N, D = x.shape
        device = x.device
        
        # Channel 1: Temporal patterns representation
        # Identify dominant periods using FFT for efficient period detection
        x_temporal = x.reshape(B, T, -1)  # [B, T, N*D]
        
        # Apply FFT to find dominant frequencies
        xf = torch.fft.rfft(x_temporal, dim=1)
        frequency_magnitudes = torch.abs(xf)
        
        # Get top-k periods based on magnitude
        top_k = min(4, T//2)  # Limit number of periods to consider
        _, top_indices = torch.topk(frequency_magnitudes.mean(dim=2), k=top_k, dim=1)
        
        # Create first channel using period-based embedding (inspired by TimesNet)
        channel1 = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        for b in range(B):
            # Take the most dominant period (or use T if no clear periodicity)
            period = min(T-1, max(2, T // (top_indices[b, 0].item() + 1)))
            
            # Reshape based on periodicity for 2D representation
            num_segments = T // period
            if num_segments > 0:
                # Reshape time series as 2D using dominant period
                segment = x_temporal[b, :num_segments*period].reshape(num_segments, period, -1)
                
                # Create 2D embedding via mean across features
                embed_2d = segment.mean(dim=2)  # [num_segments, period]
                
                # Normalize for visualization
                embed_2d = (embed_2d - embed_2d.min()) / (embed_2d.max() - embed_2d.min() + 1e-8)
                
                # Resize to target dimensions
                embed_2d = F.interpolate(
                    embed_2d.unsqueeze(0).unsqueeze(0), 
                    size=(self.image_size, self.image_size),
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                channel1[b, 0] = embed_2d
        
        # Channel 2: Spatial patterns representation using graph structure
        channel2 = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        # Efficient spatial representation using adjacency matrix
        if adj_mx is not None:
            # Ensure adjacency matrix is on the correct device
            if not isinstance(adj_mx, torch.Tensor):
                adj_mx = torch.tensor(adj_mx, device=device)
            
            # Normalize adjacency matrix if needed
            if adj_mx.shape[0] > 1:  # Batch of adjacency matrices
                norm_adj = adj_mx.float()
            else:  # Single adjacency matrix for all batches
                norm_adj = adj_mx.float().expand(B, -1, -1)
            
            # Create spatial embeddings efficiently
            for b in range(B):
                # Node feature aggregation using adjacency matrix
                node_features = x[b].mean(dim=0)  # [N, D]
                node_embedding = node_features.mean(dim=1)  # [N]
                
                # Use adjacency to create 2D spatial relationships
                spatial_embed = torch.matmul(norm_adj[b], node_embedding.unsqueeze(-1)).squeeze()
                
                # Normalize and reshape into grid
                spatial_embed = normalize_max(spatial_embed)
                
                # Calculate grid dimensions
                grid_size = int(math.ceil(math.sqrt(N)))
                
                # Create 2D grid representation
                spatial_grid = torch.zeros(grid_size, grid_size, device=device)
                for i in range(min(N, grid_size*grid_size)):
                    row, col = i // grid_size, i % grid_size
                    spatial_grid[row, col] = spatial_embed[i]
                
                # Resize to desired dimensions
                channel2[b, 0] = F.interpolate(
                    spatial_grid.unsqueeze(0).unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
        
        # Channel 3: Spatio-temporal interaction patterns
        channel3 = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
        
        for b in range(B):
            # Extract temporal dynamics
            x_mean_space = x[b].mean(dim=1)  # Average across nodes [T, D]
            x_mean_features = x_mean_space.mean(dim=1)  # Average across features [T]
            
            # Extract spatial dynamics
            x_mean_time = x[b].mean(dim=0)  # Average across time [N, D]
            x_mean_time_features = x_mean_time.mean(dim=1)  # Average across features [N]
            
            # Create interaction matrix
            temp_length = len(x_mean_features)
            space_length = len(x_mean_time_features)
            
            # Create simplified interaction map (correlation-like)
            interaction = torch.zeros(min(self.image_size, temp_length), 
                                     min(self.image_size, space_length),
                                     device=device)
            
            # Calculate temporal-spatial correlation efficiently
            temp_norm = normalize_minmax(x_mean_features)
            space_norm = normalize_minmax(x_mean_time_features)
            
            # Create correlation matrix using outer product
            min_t = min(self.image_size, temp_length)
            min_s = min(self.image_size, space_length)
            interaction = torch.outer(temp_norm[:min_t], space_norm[:min_s])
            
            # Resize to target dimensions
            channel3[b, 0] = F.interpolate(
                interaction.unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Combine channels
        return torch.cat([channel1, channel2, channel3], dim=1)

class SpatialTemporalAttention(nn.Module):
    """Multi-head attention mechanism for capturing spatio-temporal dependencies"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SpatialTemporalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: Query tensor [B, L, D]
            key: Key tensor [B, S, D]
            value: Value tensor [B, S, D]
            attn_mask: Optional attention mask
        Returns:
            Output tensor and attention weights
        """
        output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )
        return output, attn_weights

class CrossModalFusionLayer(nn.Module):
    """Fuses information from different modalities (visual, text, frequency)"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossModalFusionLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.self_attn = SpatialTemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.cross_attn = SpatialTemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_vis, Cond_Temp=None, Cond_Text=None):
        """
        x_vis: [B, 3, H, W]
        Cond_Temp: [B, D_temp, N, D]
        Cond_Text: [B, D_text, N]
        通过对TT和CT的交叉注意力，将TT和CT的信息融合到x中
        """
        # Self attention
        residual = x_vis
        x_vis = self.norm1(x_vis)
        x_vis_self, _ = self.self_attn(x_vis, x_vis, x_vis)
        x = residual + x_vis_self
        
        # Cross attention
        residual = x
        x = self.norm2(x)
        x_cross, _ = self.cross_attn(x, Cond_Temp, Cond_Temp)
        x = residual + x_cross
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff_network(x)
        
        return x

class GatingMechanism(nn.Module):
    """
    Gating mechanism for conditional fusion of features
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        """
        Initialize gating mechanism
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden representations
            dropout: Dropout probability
        """
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.projection = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x1, x2):
        """
        Apply gating mechanism to fuse two feature sets
        
        Args:
            x1: First feature set [B, T, N, D]
            x2: Second feature set [B, D']
        Returns:
            Fused features [B, T, N, D]
        """
        B, T, N, D = x1.shape
        
        # Project second features if needed
        if x2.shape[-1] != D:
            x2 = self.projection(x2)
        
        # Reshape for concatenation
        x2_expanded = x2.view(B, 1, 1, -1).expand(-1, T, N, -1)
        
        # Concatenate features
        combined = torch.cat([x1, x2_expanded], dim=-1)
        
        # Compute gating weights
        gate = self.gate_net(combined)
        
        # Apply gate
        return x1 * gate + (1 - gate) * x2_expanded

class SpatioTemporalOutputHead(nn.Module):
    """
    Component that processes text features to generate spatio-temporal predictions.
    Incorporates hidden state information when available to enhance predictions.
    """
    def __init__(self, config):
        super().__init__()
        # Configuration parameters
        self.horizon = config['horizon']
        self.num_nodes = config['num_nodes']
        self.output_dim = config['output_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        self.embed_dim = config.get('embed_dim', 128)
        self.dropout = config.get('dropout', 0.1)
        
        # Text feature processor
        self.text_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Node embeddings
        self.register_buffer('position_ids', torch.arange(self.num_nodes).expand((1, -1)))
        self.node_embeddings = nn.Embedding(self.num_nodes, self.hidden_dim)
        
        # Hidden state processor
        self.hidden_state_processor = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU()
        )
        
        # Feature fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        # Caching mechanism
        self._output_cache = {}
        self._cache_size_limit = 50
    
    def forward(self, text_features, hidden_state=None, cache_key=None):
        """
        Integrates text features and hidden state to generate predictions
        
        Args:
            text_features: Text features [batch_size, hidden_dim]
            hidden_state: Optional hidden state [batch_size, embed_dim, num_nodes, output_dim]
            cache_key: Optional cache key for faster inference
            
        Returns:
            Predictions [batch_size, horizon, num_nodes, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Check cache - only when not using hidden state
        if cache_key and not self.training and hidden_state is None:
            try:
                if cache_key in self._output_cache:
                    return self._output_cache[cache_key].to(device)
            except:
                pass
        
        # 1. Process text features
        text_features = self.text_processor(text_features)
        
        # 2. Get node embeddings
        node_ids = self.position_ids.expand(batch_size, -1)
        node_embeddings = self.node_embeddings(node_ids)  # [batch_size, num_nodes, hidden_dim]
        
        # 3. Process hidden state if provided
        if hidden_state is not None:
            # Transform hidden state: [batch_size, embed_dim, num_nodes, output_dim]
            
            # Efficient batch processing using convolution
            hidden_features = self.hidden_state_processor(hidden_state)  # [batch_size, hidden_dim, num_nodes, output_dim]
            
            # Pool across output dimension to get [batch_size, hidden_dim, num_nodes, 1]
            hidden_features = hidden_features.mean(dim=-1, keepdim=True)
            
            # Reshape to [batch_size, num_nodes, hidden_dim]
            hidden_features = hidden_features.squeeze(-1).permute(0, 2, 1)
            
            # Fuse node embeddings with hidden features using gating
            combined_features = torch.cat([node_embeddings, hidden_features], dim=-1)
            gate = self.fusion_gate(combined_features)
            node_embeddings = gate * node_embeddings + (1 - gate) * hidden_features
        
        # 4. Expand dimensions
        # Add time dimension: [batch_size, horizon, num_nodes, hidden_dim]
        node_expanded = node_embeddings.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        
        # Expand text features: [batch_size, horizon, num_nodes, hidden_dim]
        text_expanded = text_features.unsqueeze(1).unsqueeze(2).expand(
            -1, self.horizon, self.num_nodes, -1)
        
        # 5. Combine features and predict
        combined = torch.cat([node_expanded, text_expanded], dim=-1)
        output = self.output_projection(combined)  # [batch_size, horizon, num_nodes, output_dim]
        
        # Handle caching
        if cache_key and not self.training and hidden_state is None:
            try:
                if len(self._output_cache) >= self._cache_size_limit:
                    self._output_cache = {}
                self._output_cache[cache_key] = output.detach().cpu()
            except:
                pass
        
        return output

class TemporalMapping(nn.Module):
    """
    Maps spatio-temporal data from input dimensions to target temporal embedding dimensions
    Modified based on STID's approach
    """
    def __init__(self, seq_len, input_dim, tem_dim, output_dim, num_nodes, 
                 node_dim=32, temp_dim_tid=32, temp_dim_diw=32, 
                 time_of_day_size=288, day_of_week_size=7):
        super().__init__()
        # Basic dimensions
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = tem_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
        # Temporal embedding dimensions
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        # Spatial embeddings
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
            
        # Temporal embeddings
        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
            
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Time series embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=input_dim * seq_len, 
            out_channels=self.embed_dim, 
            kernel_size=(1, 1), 
            bias=True
        )
        
        # Calculate total hidden dimension
        self.hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_tid + self.temp_dim_diw
        
        # Projection to output dimension
        self.output_projection = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=tem_dim * output_dim,
            kernel_size=(1, 1),
            bias=True
        )
    
    def forward(self, x):
        """
        Transform input tensor to target dimensions
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            
        Returns:
            Tensor of shape [batch_size, tem_dim, num_nodes, output_dim]
        """
        # Prepare data - assuming first channels are the main features
        batch_size, seq_len, num_nodes, _ = x.shape
        input_data = x[..., :self.input_dim]
        
        # Time series embedding
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)  # [B, embed_dim, N, 1]
        
        # Node embeddings - spatial embedding
        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        
        # Temporal embeddings
        tem_emb = []
        t_i_d_data = x[..., 1]  # Assuming time-of-day is channel 1
        time_in_day_emb = self.time_in_day_emb[
        (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
            
        d_i_w_data = x[..., 2]  # Assuming day-of-week is channel 2
        day_in_week_emb = self.day_in_week_emb[
            (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # Concatenate all embeddings
        hidden_state = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # [B, hidden_dim, N, 1]
        
        # Project to output dimensions
        output = self.output_projection(hidden_state)  # [B, tem_dim*output_dim, N, 1]
        output = output.squeeze(-1).view(batch_size, self.embed_dim, self.output_dim, num_nodes)
        output = output.permute(0, 1, 3, 2)  # [B, tem_dim, N, output_dim]
        
        return output

class ViST(nn.Module):
    """
    Vision-enhanced Spatio-Temporal forecasting model
    
    Transforms spatio-temporal data into multiple visual and textual 
    representations, then uses a cross-modal framework to generate predictions
    """
    def __init__(self, **model_args):
        super().__init__()
        
        # Unified configuration with defaults
        self.config = {
            # Core dimensions
            'num_nodes': model_args.get('num_nodes', 207),
            'input_dim': model_args.get('input_dim', 1),
            'output_dim': model_args.get('output_dim', 1),
            'input_len': model_args.get('input_len', 12),
            'horizon': model_args.get('output_len', 12),
            'embed_dim': model_args.get('embed_dim', 128),
            'hidden_dim': model_args.get('hidden_dim', 64),
            
            # Visual parameters
            'image_size': model_args.get('image_size', 64),
            'periodicity': model_args.get('periodicity', 24),
            'interpolation': model_args.get('interpolation', 'bilinear'),
            'save_images': model_args.get('save_images', False),
            'd_vision': model_args.get('d_vision', 64),
            
            # Text parameters
            'vocab_size': model_args.get('vocab_size', 10000),
            'llm_dim': model_args.get('llm_dim', 768),
            'llm_model': model_args.get('llm_model', 'bert-base-uncased'),
            'd_ff': model_args.get('d_ff', 32),
            'n_heads': model_args.get('n_heads', 8),
            'top_k': model_args.get('top_k', 3),
            
            # Temporal parameters
            'd_temporal': model_args.get('d_temporal', 64),
            'node_dim': model_args.get('node_dim', 32),
            'temp_dim_tid': model_args.get('temp_dim_tid', 32),
            'temp_dim_diw': model_args.get('temp_dim_diw', 32),
            'time_of_day_size': model_args.get('time_of_day_size', 288),
            'day_of_week_size': model_args.get('day_of_week_size', 7),
            
            # Output parameters
            'output_channels': model_args.get('output_channels', [32, 64, 128]),
            
            # Training and fusion parameters
            'dropout': model_args.get('dropout', 0.1),
            'encoder_layers': model_args.get('encoder_layers', 3),
            'use_conditioning': model_args.get('use_conditioning', False),
            'condition_dim': model_args.get('condition_dim', 64),
            'fusion_hidden_dim': model_args.get('fusion_hidden_dim', 64),
            'd_fusion': model_args.get('d_fusion', 64),
            'd_hidden': model_args.get('d_hidden', 64),
            
            # Model type
            'output_type': model_args.get('output_type', "full")
        }
        
        # Store working variables
        self.output_type = self.config['output_type']
        self.seq_len = self.config['input_len']
        self.horizon = self.config['horizon']
        self.num_nodes = self.config['num_nodes']
        self.d_ff = self.config['d_ff']
        
        # Get dataset description
        self.domain = model_args.get('data_description', "SD")
        try:
            self.data_description = load_domain_text(self.domain)
        except:
            self.data_description = ""
            
        self.config['description'] = self.data_description
        
        # Initialize components with unified config
        self.vision_encoder = MultiPerspectiveVisualEncoder(self.config)
        self.text_encoder = TextEncoder(self.config)
        self.textual_output_head = TextualOutputHead(self.config)
        
        # Temporal mapping and encoder
        self.temporal_mapping = TemporalMapping(
            self.config['input_len'], 
            self.config['input_dim'], 
            self.config['embed_dim'],
            self.config['output_dim'], 
            self.config['num_nodes'],
            node_dim=self.config['node_dim'],
            temp_dim_tid=self.config['temp_dim_tid'],
            temp_dim_diw=self.config['temp_dim_diw'],
            time_of_day_size=self.config['time_of_day_size'],
            day_of_week_size=self.config['day_of_week_size']
        )
        
        self.temporal_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.config['embed_dim'], self.config['embed_dim']) 
              for _ in range(self.config['encoder_layers'])]
        )

        self.temporal_output_head = nn.Conv2d(
            in_channels=self.config['embed_dim'], 
            out_channels=self.config['horizon'], 
            kernel_size=(1, 1), 
            bias=True
        )

        # Visual output head
        self.visual_output_head = SpatioTemporalOutputHead(self.config) 
        # Initialize fusion components
        if self.config['use_conditioning']:
            self.condition_fusion = GatingMechanism(
                input_dim=self.config['output_dim'],
                hidden_dim=self.config['fusion_hidden_dim'],
                dropout=self.config['dropout']
            )
            self.condition_projector = nn.Linear(self.config['condition_dim'], self.config['fusion_hidden_dim'])
        
        # Attention networks
        self.temporal_attn_net = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len // 2),
            nn.ReLU(),
            nn.Linear(self.seq_len // 2, self.seq_len)
        )
        
        self.spatial_attn_net = nn.Sequential(
            nn.Conv2d(self.horizon, self.horizon, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.horizon, self.horizon, kernel_size=1)
        )
        
        # Confidence networks for adaptive fusion
        self.temp_confidence_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(1, self.config['d_hidden']),
            nn.ReLU(),
            nn.Linear(self.config['d_hidden'], 1),
            nn.Sigmoid()
        )
        
        self.vis_confidence_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(1, self.config['d_hidden']),
            nn.ReLU(),
            nn.Linear(self.config['d_hidden'], 1),
            nn.Sigmoid()
        )
        
        # Visual fusion parameter
        self.visual_fusion_param = nn.Parameter(torch.tensor(0.5))
        
        # Text to visual projection
        self.text_to_visual_proj = nn.Linear(self.config['llm_dim'], self.config['d_vision'])
        
        # Cross-modal fusion
        # self.cross_modal_fusion = CrossModalFusionLayer(
        #     embed_dim=self.config['embed_dim'],
        #     num_heads=self.config['n_heads'],
        #     dropout=self.config['dropout']
        # )
        self.cross_modal_fusion = MultiModalTimeSpacePredictor(self.config)         
        # Gating mechanism for modality fusion
        self.gating = GatingMechanism(
            input_dim=self.config['output_dim'],
            hidden_dim=self.config['d_fusion'],
            dropout=self.config['dropout']
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def create_visual_representations(self, x, adj_mx=None):
        """
        Transform spatiotemporal data [B,T,N,D] into image representation [B,3,H,W]
        Args:
            x: Input data, shape [B,T,N,D]
            adj_mx: Spatial adjacency matrix
            
        Returns:
            Image representation
        """
        B, T, N, D = x.shape
        target_H, target_W = self.config['image_size'], self.config['image_size']  # Target image size
        
        # 1. First, reshape time and node information into a 2D grid
        # Find an integer close to sqrt(N) to arrange nodes into a grid
        n_row = int(math.sqrt(N))
        n_col = math.ceil(N / n_row)
        
        # 2. Reshape to [B, T, n_row, n_col, D]
        x_padded = F.pad(x, (0, 0, 0, n_row * n_col - N))
        x_grid = x_padded.view(B, T, n_row, n_col, D)
        if D >= 3:
            # Use the first three featureses directly as RGB
            x_rgb = x_grid[..., :3]
        else:
            # Use linear projection to 3 channels
            x_flat = x_grid.view(B, T, n_row * n_col, D)
            x_rgb_flat = self.rgb_projection(x_flat)  # [B, T, n_row*n_col, 3]
            x_rgb = x_rgb_flat.view(B, T, n_row, n_col, 3)
        
        # 4. Create periodic image representation, map time steps as different phases of time period
        # Method 1: Directly map T time steps to a specific region of the target image
        images = []
        for t in range(T):
            img_t = x_rgb[:, t]  # [B, n_row, n_col, 3]
            # Convert to standard image format [B, 3, n_row, n_col]
            img_t = img_t.permute(0, 3, 1, 2)
            # Bilinear interpolation to target size
            img_t_resized = F.interpolate(img_t, size=(target_H//4, target_W//4), mode='bilinear', align_corners=False)
            images.append(img_t_resized)
        
        # Concatenate T images into a large image (arranged in grid form)
        t_row = int(math.sqrt(T))
        t_col = math.ceil(T / t_row)
        
        grid_rows = []
        for i in range(t_row):
            if (i+1)*t_col <= T:
                row_images = images[i*t_col:(i+1)*t_col]
            else:
                row_images = images[i*t_col:T]
                # Pad the insufficient part
                padding = [torch.zeros_like(row_images[0]) for _ in range((i+1)*t_col - T)]
                row_images = row_images + padding
            grid_row = torch.cat(row_images, dim=3)  # Concatenate in width direction
            grid_rows.append(grid_row)
        
        grid_image = torch.cat(grid_rows, dim=2)  # Concatenate in height direction
        
        # Adjust to final target size
        final_image = F.interpolate(grid_image, size=(target_H, target_W), mode='bilinear', align_corners=False)
        
        return final_image  # [B, 3, H, W]
    
    def adaptive_gating(self, temporal_prediction, visual_prediction):
        """
        Adaptive gating mechanism with dynamic weighting
        
        Args:
            temporal_prediction: Prediction from temporal encoder
            visual_prediction: Prediction from visual modality
            
        Returns:
            Combined prediction
        """
        # Extract batch size and other dimensions
        B = temporal_prediction.shape[0]
        
        # Calculate confidence scores for each modality
        temp_conf = self.temp_confidence_net(temporal_prediction)  # [B, 1]
        vis_conf = self.vis_confidence_net(visual_prediction)      # [B, 1]
        
        # Normalize confidence scores
        conf_sum = temp_conf + vis_conf + 1e-10
        temp_weight = temp_conf / conf_sum
        vis_weight = vis_conf / conf_sum
        
        # Apply dynamic weighted fusion
        temp_weight = temp_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vis_weight = vis_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        output = temp_weight * temporal_prediction + vis_weight * visual_prediction
        
        # Optional residual connection
        output = output + self.gating(temporal_prediction, visual_prediction)
        
        return output

    def forward(self, history_data, future_data=None, adj_mx=None, text_tokens=None, batch_seen=None, epoch=None, train=True, **kwargs):
        """
        Forward pass of the ViST model with separated text processing
        
        Args:
            history_data: Input historical data of shape [B, T, N, D]
            future_data: Future data for teacher forcing of shape [B, T', N, D]
            adj_mx: Adjacency matrix for spatial relationships
            text_tokens: Text tokens for text modality
            batch_seen: Current batch index
            epoch: Current epoch
            train: Whether in training mode
            
        Returns:
            Predicted future values
        """
        B, T, N, D = history_data.shape
        hidden_state = self.temporal_mapping(history_data)  # [batch_size, embed_dim, num_nodes, output_dim]
            
        # Process text representation with clear separation between steps
        if self.output_type == "only_visual":
            # Note: Here we directly output results using the vision-based output head
            visual_representation = self.vision_encoder(hidden_state, adj_mx) # [batch_size, 3, image_size, image_size]
            output = self.visual_output_head(visual_representation) # [batch_size, horizon, num_nodes, output_dim]
            return output
            
        elif self.output_type == "only_temporal":
            # Note: STID-liked temporal output head
            temporal_representation = self.temporal_encoder(hidden_state) # [batch_size, embed_dim, num_nodes, output_dim]
            output = self.temporal_output_head(temporal_representation)
            return output
            
        elif self.output_type == "only_textual":
            prompts = self.text_encoder.generate_prompts(history_data, self.data_description) # len(batch_size)
            text_representation = self.text_encoder.get_text_embeddings(prompts, history_data.device) # [batch_size, d_ff]
            text_features = self.text_encoder.get_text_features(text_representation)
            #print("text_features:", text_features.shape) # [batch_size, d_ff]
            cache_key = hashlib.md5(str(prompts).encode()).hexdigest() if not train else None
            output = self.textual_output_head(text_features, hidden_state, cache_key)
            return output
        
        output = self.cross_modal_fusion(hidden_state, temporal_representation, text_representation, visual_representation)
        return output

        # Generate predictions from each modality
        visual_output = self.visual_output_head(visual_representation)
        temporal_output = self.temporal_output_head(temporal_representation)
        text_output = self.textual_output_head(text_features, hidden_state)
        
        # Adaptive fusion of modalities
        output = self.adaptive_gating(temporal_output, visual_output)
        
        # Optional conditioning with text features
        if self.config['use_conditioning']:
            text_condition = self.condition_projector(text_features)
            output = self.condition_fusion(output, text_condition)
    
        return output