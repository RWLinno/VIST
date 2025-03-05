import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft
from torchvision.transforms import Resize
from PIL import Image
import einops
import inspect
from contextlib import contextmanager
from easydict import EasyDict

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)

def safe_resize(size, interpolation=Image.BILINEAR):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation=interpolation, antialias=False)
    else:
        return Resize(size, interpolation=interpolation)

def normalize_minmax(x, eps=1e-8):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < eps:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min + eps)

class SpatioTemporalPixelEncoder(nn.Module):
    """
    Encodes spatio-temporal data into multiple visual representations
    """
    def __init__(self, config):
        super(SpatioTemporalPixelEncoder, self).__init__()
        self.image_size = config.image_size
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation
        self.save_debug_images = getattr(config, 'save_debug_images', False)
        
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

    def forward(self, x, adj_mx=None, method='seg', save_images=False):
        """
        Convert spatio-temporal data to image representation
        Args:
            x: Tensor of shape [B, T, N, D] where B is batch size, T is time steps,
               N is number of nodes, D is feature dimension
            adj_mx: Optional adjacency matrix of shape [N, N]
            method: Representation type ('seg', 'gaf', 'rp', or 'spatial')
            save_images: Whether to save debug images
        Returns:
            Tensor of shape [B, 1, image_size, image_size]
        """
        if method == 'seg':
            output = self.spatio_temporal_segmentation(x)
        elif method == 'gaf':
            output = self.gramian_angular_field(x)
        elif method == 'rp':
            output = self.recurrence_plot(x)
        elif method == 'spatial':
            output = self.spatial_adjacency_image(x, adj_mx)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        output = self.normalize(output)

        if save_images:
            self.save_images(output, method, x.size(0))
        return output

class FFTTransformer(nn.Module):
    """Extract frequency domain features from spatio-temporal data"""
    def __init__(self, config=None):
        super(FFTTransformer, self).__init__()
        self.config = config
        self.register_buffer('window', None)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, N, D] or flattened
        Returns:
            Frequency domain representation
        """
        B, T, N, D = x.shape
        # Flatten spatial and feature dimensions
        x = x.reshape(B, T, -1)  # [B, T, N*D]
        
        if self.window is None or self.window.size(-1) != x.size(1):
            self.window = torch.hann_window(x.size(1), device=x.device)
        
        # Apply window function
        x = x * self.window.unsqueeze(0).unsqueeze(-1)
        
        # Apply FFT along time dimension
        x_fft = rfft(x, dim=1)
        
        # Concatenate real and imaginary parts
        return torch.cat([x_fft.real, x_fft.imag], dim=-1)

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
    
    def forward(self, x, context):
        """
        Args:
            x: Main modality tensor [B, L, D]
            context: Context modality tensor [B, S, D]
        Returns:
            Fused representation
        """
        # Self attention
        residual = x
        x = self.norm1(x)
        x_self, _ = self.self_attn(x, x, x)
        x = residual + x_self
        
        # Cross attention
        residual = x
        x = self.norm2(x)
        x_cross, _ = self.cross_attn(x, context, context)
        x = residual + x_cross
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff_network(x)
        
        return x

class GatingMechanism(nn.Module):
    """Dynamic gating mechanism to combine different predictors"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        """
        Args:
            x1: First predictor output
            x2: Second predictor output
        Returns:
            Gated combination of the two predictors
        """
        combined = torch.cat([x1, x2], dim=-1)
        weights = self.gate(combined)
        return weights[:, :, 0:1] * x1 + weights[:, :, 1:2] * x2

class SpatioTemporalOutputHead(nn.Module):
    """Converts visual features to spatio-temporal predictions"""
    def __init__(self, config):
        super(SpatioTemporalOutputHead, self).__init__()
        self.horizon = config.horizon  # prediction horizon
        self.num_nodes = config.num_nodes  # number of spatial locations
        self.out_dim = config.c_out  # output dimension
        
        # Convolutional layers to process visual representation
        self.conv1 = nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, self.horizon, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(self.horizon)
        
        self.relu = nn.LeakyReLU(0.1)
        
        # Fully connected layer to map to final output dimensions
        flattened_size = config.image_size * config.image_size
        self.fc = nn.Linear(flattened_size, self.num_nodes * self.out_dim)
        
    def forward(self, x):
        """
        Args:
            x: Visual representation [B, C, H, W]
        Returns:
            Spatio-temporal predictions [B, horizon, num_nodes, out_dim]
        """
        batch_size = x.size(0)
        
        # Convolutional processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Each channel now represents one time step in the prediction horizon
        # [B, horizon, H, W]
        
        # For each time step, map the visual features to node predictions
        output = []
        for t in range(self.horizon):
            time_slice = x[:, t]  # [B, H, W]
            flattened = time_slice.reshape(batch_size, -1)  # [B, H*W]
            node_predictions = self.fc(flattened)  # [B, num_nodes*out_dim]
            node_predictions = node_predictions.reshape(
                batch_size, self.num_nodes, self.out_dim)  # [B, num_nodes, out_dim]
            output.append(node_predictions)
        
        # Stack along time dimension
        output = torch.stack(output, dim=1)  # [B, horizon, num_nodes, out_dim]
        
        return output

class TemporalEncoder(nn.Module):
    """Encoder for capturing temporal dependencies in spatio-temporal data"""
    def __init__(self, config):
        super(TemporalEncoder, self).__init__()
        self.d_model = config.d_model
        self.num_nodes = config.num_nodes
        self.seq_len = config.seq_len
        self.horizon = config.horizon
        self.out_dim = config.c_out
        
        # Position encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.d_model)
        )
        nn.init.xavier_uniform_(self.pos_encoding)
        
        # Input projection
        self.input_projection = nn.Linear(config.c_in, self.d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.e_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            self.d_model * self.seq_len, 
            self.horizon * self.out_dim
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, N, D] where B is batch size, T is sequence length,
               N is number of nodes, D is feature dimension
        Returns:
            Temporal prediction [B, horizon, N, out_dim]
        """
        batch_size, seq_len, num_nodes, feat_dim = x.shape
        
        # Process each node separately
        outputs = []
        for node in range(num_nodes):
            # Extract node data
            node_data = x[:, :, node, :]  # [B, T, D]
            
            # Project to model dimension
            node_data = self.input_projection(node_data)  # [B, T, d_model]
            
            # Add positional encoding
            node_data = node_data + self.pos_encoding[:, :seq_len, :]
            
            # Apply transformer encoder
            node_encoded = self.transformer_encoder(node_data)  # [B, T, d_model]
            
            # Flatten and project to output
            node_flattened = node_encoded.reshape(batch_size, -1)  # [B, T*d_model]
            node_output = self.output_projection(node_flattened)  # [B, horizon*out_dim]
            
            # Reshape to [B, horizon, out_dim]
            node_output = node_output.reshape(batch_size, self.horizon, self.out_dim)
            
            outputs.append(node_output)
        
        # Stack all node outputs
        outputs = torch.stack(outputs, dim=2)  # [B, horizon, N, out_dim]
        
        return outputs

class ViST(nn.Module):
    """
    Vision-enhanced Spatio-Temporal forecasting model
    
    Transforms spatio-temporal data into multiple visual and textual 
    representations, then uses a cross-modal conditioned framework to generate predictions
    """
    def __init__(self, **model_args):
        super(ViST, self).__init__()
        
        # 保存配置
        self.config = model_args
        
        # 基本维度参数
        self.num_nodes = model_args['num_nodes'] 
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        
        # 视觉编码器配置
        vision_config = EasyDict({
            'image_size': model_args.get('image_size', 32),
            'periodicity': model_args.get('periodicity', 24),
            'interpolation': model_args.get('interpolation', 'bilinear'),
            'save_debug_images': model_args.get('save_debug_images', False)
        })
        self.vision_encoder = SpatioTemporalPixelEncoder(vision_config)
        
        # 频率域编码器配置
        freq_config = EasyDict({
            'freq_embed_dim': model_args['freq_embed_dim'],
            'd_model': model_args['d_model']
        })
        self.frequency_encoder = FFTTransformer(freq_config)
        self.freq_proj = nn.Linear(model_args['freq_embed_dim'], model_args['d_model'])
        
        # 文本编码器配置
        text_config = EasyDict({
            'vocab_size': model_args['vocab_size'],
            'd_model': model_args['d_model'],
            'd_ff': model_args['d_ff'],
            'dropout': model_args['dropout']
        })
        self.text_embedding = nn.Embedding(model_args['vocab_size'], model_args['d_model'])
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_args['d_model'],
                nhead=model_args['n_heads'],
                dim_feedforward=model_args['d_ff'],
                dropout=model_args['dropout'],
                activation='gelu',
                batch_first=True
            ),
            num_layers=model_args['encoder_layers']
        )
        
        # 时间编码器配置
        temporal_config = EasyDict({
            'd_model': model_args['d_model'],
            'num_nodes': model_args['num_nodes'],
            'seq_len': model_args['input_len'],
            'horizon': model_args['output_len'],
            'c_in': model_args['input_dim'],
            'c_out': model_args['output_dim'],
            'd_ff': model_args['d_ff'],
            'dropout': model_args['dropout'],
            'e_layers': model_args['encoder_layers']
        })
        self.temporal_encoder = TemporalEncoder(temporal_config)
        
        # 视觉输出头配置
        output_config = EasyDict({
            'horizon': model_args['output_len'],
            'num_nodes': model_args['num_nodes'],
            'c_out': model_args['output_dim'],
            'in_channels': 4,  # seg, gaf, rp, spatial
            'image_size': model_args.get('image_size', 64)
        })
        self.visual_output_head = SpatioTemporalOutputHead(output_config)
        
        # 跨模态融合层配置
        fusion_config = EasyDict({
            'embed_dim': model_args['d_model'],
            'num_heads': model_args['n_heads'],
            'dropout': model_args['dropout']
        })
        self.fusion_layer = CrossModalFusionLayer(**fusion_config)
        
        # 门控机制配置
        gating_config = EasyDict({
            'input_dim': model_args['output_dim'],
            'hidden_dim': model_args.get('d_fusion', model_args['d_model']),
            'dropout': model_args['dropout']
        })
        self.gating = GatingMechanism(**gating_config)
        
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
    
    def generate_text_embedding(self, text_tokens):
        """Generate embeddings from text description"""
        embedded = self.text_embedding(text_tokens)
        encoded = self.text_encoder(embedded)
        # Return averaged embedding across tokens
        return encoded.mean(dim=1)  # [B, d_model]
    
    def process_frequency_domain(self, x):
        """Process data in frequency domain"""
        freq_features = self.frequency_encoder(x)
        return self.freq_proj(freq_features)
    
    def create_visual_representations(self, x, adj_mx=None):
        """Create multiple visual representations of the data"""
        seg_images = self.vision_encoder(x, adj_mx, method='seg')
        gaf_images = self.vision_encoder(x, adj_mx, method='gaf')
        rp_images = self.vision_encoder(x, adj_mx, method='rp')
        spatial_images = self.vision_encoder(x, adj_mx, method='spatial')
        
        print("seg_images.shape",seg_images.shape)
        print("gaf_images.shape",gaf_images.shape)
        print("rp_images.shape",rp_images.shape)
        print("spatial_images.shape",spatial_images.shape)

        # Stack channels for a multi-view representation
        visual_representation = torch.cat([seg_images, gaf_images, rp_images, spatial_images], dim=1)
        
        return visual_representation
    
    def forward(self, history_data, future_data=None, adj_mx=None, text_tokens=None, batch_seen=None, epoch=None, train=True, **kwargs):
        """
        Forward pass of the ViST model
        
        Args:
            history_data: Input tensor of shape [B, T, N, D] where B is batch size, 
                          T is sequence length, N is number of nodes, D is feature dimension
            future_data: Optional ground truth future data for teacher forcing
            adj_mx: Optional adjacency matrix describing spatial relationships
            text_tokens: Optional text tokens describing the dataset
            batch_seen: Optional batch counter for scheduled sampling
            epoch: Optional epoch counter
            train: Whether in training mode
            
        Returns:
            Predictions tensor of shape [B, horizon, N, D]
        """
        B, T, N, D = history_data.shape
        print("history_data.shape",history_data.shape) # [32, 12, 716, 3]
        assert T == self.seq_len, f"Expected sequence length {self.seq_len}, got {T}"
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes, got {N}"
        
        # Normalize the input data (store statistics for denormalization)
        data_mean = history_data.mean(dim=1, keepdim=True)
        data_std = history_data.std(dim=1, keepdim=True) + 1e-5
        x_norm = (history_data - data_mean) / data_std
        
        # 1. Create visual representations of the data
        visual_representation = self.create_visual_representations(x_norm, adj_mx)
        print("visual_representation.shape",visual_representation.shape)
        
        # 2. Process frequency domain features
        freq_features = self.process_frequency_domain(x_norm)
        print("freq_features.shape",freq_features.shape)
        # 3. Process text descriptions if provided
        if text_tokens is not None:
            text_features = self.generate_text_embedding(text_tokens)
            print("text_features.shape",text_features.shape)
        else:
            # Default text embedding if none provided
            text_features = torch.zeros(
                B, self.config.d_model, device=history_data.device)
        
        # 4. Generate temporal predictions
        temporal_prediction = self.temporal_encoder(x_norm)
        print("temporal_prediction.shape",temporal_prediction.shape)

        # 5. Generate visual predictions
        visual_prediction = self.visual_output_head(visual_representation)
        print("visual_prediction.shape",visual_prediction.shape)

        # 6. Combine predictions with gating mechanism
        output = self.gating(temporal_prediction, visual_prediction)
        print("output.shape",output.shape)
        
        # 7. Denormalize to original scale
        output = output * data_std.repeat(1, self.horizon, 1, 1)
        output = output + data_mean.repeat(1, self.horizon, 1, 1)
        print("output.shape",output.shape)
        return output 