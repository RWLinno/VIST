import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalTimeSpacePredictor(nn.Module):
    """
    Advanced multi-modal spatiotemporal predictor that integrates spatial, temporal,
    textual, and visual representations to forecast future spatiotemporal states.
    
    Handles optional modalities gracefully and employs advanced fusion techniques
    for optimal performance.
    """
    def __init__(self, config, d_ff=512, 
                 hidden_dim=256, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.num_nodes = config['num_nodes']
        self.output_dim = config['output_dim']
        self.horizon = config['horizon']
        self.hidden_dim = config['hidden_dim']
        self.embed_dim = config['embed_dim']
        self.d_ff = d_ff
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        # Visual representation processing
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        
        # Text representation processing
        self.text_encoder = nn.Sequential(
            nn.Linear(self.d_ff, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Spatial representation processing with residual connections
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Temporal representation processing
        self.temporal_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Cross-modal attention for feature fusion
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True, dropout=self.dropout)
        
        # Additional graph attention for spatial dependencies
        self.graph_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Temporal attention for capturing long-range dependencies
        self.temporal_attention = nn.MultiheadAttention(self.hidden_dim, 8, batch_first=True, dropout=self.dropout)
        
        # Advanced predictor with temporal convolution for horizon forecasting
        self.predictor = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim * 2, self.horizon * self.output_dim, kernel_size=1)
        )
        
        # Layer normalization for feature stabilization
        if self.use_layer_norm:
            self.feature_norm = nn.LayerNorm(self.hidden_dim)
            self.output_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, hidden_state, temporal_representation=None, text_representation=None, visual_representation=None):
        """
        Forward pass integrating available modalities for spatiotemporal prediction.
        
        Args:
            hidden_state: Spatiotemporal representation [batch_size, embed_dim, num_nodes, output_dim]
            temporal_representation: Optional temporal representation [batch_size, embed_dim, num_nodes, output_dim]
            text_representation: Optional text conditioning [batch_size, d_ff]
            visual_representation: Optional visual representation [batch_size, 3, image_size, image_size]
            
        Returns:
            Predicted future states [batch_size, horizon, num_nodes, output_dim]
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        
        # Process spatial representation
        spatial_feat = self.spatial_encoder(hidden_state)  # [B, hidden_dim, num_nodes, output_dim]
        
        # Process temporal representation (or use zeros if not provided)
        if temporal_representation is not None:
            temporal_feat = self.temporal_encoder(temporal_representation)
        else:
            temporal_feat = torch.zeros_like(spatial_feat)
        
        # Reshape for attention operations
        spatial_feat = spatial_feat.permute(0, 2, 3, 1)  # [B, num_nodes, output_dim, hidden_dim]
        spatial_feat = spatial_feat.reshape(batch_size, self.num_nodes * self.output_dim, self.hidden_dim)
        
        temporal_feat = temporal_feat.permute(0, 2, 3, 1)
        temporal_feat = temporal_feat.reshape(batch_size, self.num_nodes * self.output_dim, self.hidden_dim)
        
        # Apply temporal self-attention
        temporal_context, _ = self.temporal_attention(
            temporal_feat, temporal_feat, temporal_feat
        )
        
        # Process text representation
        if text_representation is not None:
            text_feat = self.text_encoder(text_representation).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            text_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        # Process visual representation
        if visual_representation is not None:
            visual_feat = self.visual_encoder(visual_representation).unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            visual_feat = torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        # Combine external context
        combined_context = torch.cat([text_feat, visual_feat], dim=1)  # [B, 2, hidden_dim]
        
        # Cross-modal attention
        enhanced_spatial, _ = self.cross_attention(
            spatial_feat, combined_context, combined_context
        )
        
        # Fusion with residual connections
        fused_feat = enhanced_spatial + temporal_context + spatial_feat
        
        if self.use_layer_norm:
            fused_feat = self.feature_norm(fused_feat)
        
        # Graph attention for spatial dependencies
        node_features = fused_feat.reshape(batch_size * self.num_nodes, self.output_dim, self.hidden_dim)
        node_features = self.graph_attention(node_features)
        node_features = node_features.reshape(batch_size, self.num_nodes * self.output_dim, self.hidden_dim)
        
        # Final feature representation
        final_feat = fused_feat + node_features
        
        if self.use_layer_norm:
            final_feat = self.output_norm(final_feat)
        
        # Reshape for prediction
        final_feat = final_feat.permute(0, 2, 1)  # [B, hidden_dim, num_nodes*output_dim]
        
        # Predict future steps
        predictions = self.predictor(final_feat)  # [B, horizon*output_dim, num_nodes*output_dim]
        
        # Important: Reshape to match expected output format [B, L, N, C]
        predictions = predictions.reshape(batch_size, self.horizon, self.output_dim, self.num_nodes, self.output_dim)
        predictions = predictions.sum(dim=-1)  # Merge the last dimension features
        predictions = predictions.permute(0, 1, 3, 2)  # [B, horizon, num_nodes, output_dim]
        
        return predictions