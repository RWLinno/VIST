import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import faiss
import numpy as np
import json
import os
import glob
import traceback
from .blocks.Embed import *
from .blocks.Pretrain import *
from .blocks.Text_Encoder import *
from .blocks.RetrievalStore import *

class RAST(nn.Module):
    """
    Retrieval-Augmented Spatio-Temporal Model (RAST)
    
    A neural network architecture that combines:
    1. Pre-trained transformer encoder for spatio-temporal feature extraction
    2. Retrieval-based augmentation using domain-specific prompts
    3. Cross-attention mechanism for feature fusion
    """
    
    def __init__(self, **model_args):
        """
        Initialize RAST model
        
        Args:
            model_args: Dictionary containing model parameters including:
                - num_nodes: Number of spatial nodes
                - input_dim: Input feature dimension
                - output_dim: Output prediction dimension
                - embed_dim: Embedding dimension
                - input_len: Input sequence length
                - output_len: Prediction horizon
                - encoder_layers: Number of transformer encoder layers
                - decoder_layers: Number of decoder layers
                - patch_size: Size of patches for embedding
                - prompt_domain: Domain for prompt generation
                - top_k: Number of nearest neighbors for retrieval
                - batch_size: Batch size for prompt processing (default: 32)
        """
        super().__init__()
        # Model parameters
        self.num_nodes = model_args['num_nodes'] 
        self.input_dim = model_args['input_dim']
        self.output_dim = model_args['output_dim']
        self.embed_dim = model_args['embed_dim']
        self.retrieval_dim = model_args['retrieval_dim']
        self.temporal_dim = model_args.get('temporal_dim', 64)
        self.spatial_dim = model_args.get('spatial_dim', 32)
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        self.encoder_layers = model_args['encoder_layers']
        self.decoder_layers = model_args['decoder_layers']
        self.patch_size = model_args['patch_size']
        self.domain = model_args['prompt_domain']
        self.top_k = model_args['top_k']
        self.dropout = model_args['dropout']
        self.prompt_batch_size = model_args.get('batch_size', 32)
        
        self.debug_prompts = 1
        self.update_interval = 5
        self.output_type = model_args.get('output_type', 'full')  # 'full', 'only_data_embed', or 'only_retrieval_embed'
        
        # LLM configuration
        self.llm_model = model_args.get('llm_model', "bert-base-uncased")
        self.llm_dim = model_args.get('llm_dim', 768)  # Default BERT dimension
        
        # Pre-training configuration
        self.from_scratch = model_args.get('from_scratch', True)
        self.pre_train_path = model_args.get('pre_train_path', None)
        self.database_path = model_args.get('database_path', './database')
        
        # Ensure database directory exists
        os.makedirs(self.database_path, exist_ok=True)
        print(f"Database path: {self.database_path}")
        
        # Initialize components
        self._init_components()
        self._init_llm_encoder()
        self._freeze_components()
        
        # Create ablation-specific components
        self._init_ablation_components()
        
        # Training settings
        self.prompt_cache = {}  # Cache for prompt generation
        
    def _init_components(self):
        """Initialize model components with proper weight initialization and regularization"""
        # Pre-trained encoder
        self.pre_train_encoder = self._load_weight()
        print(f"Pre-trained encoder initialized with {self.encoder_layers} layers")
        
        # Retrieval component with proper initialization
        self.retrieval_store = RetrievalStore(
            self.retrieval_dim, 
            doc_dir=self.database_path,
            max_files=5,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len
        )
        
        # Projection layers with orthogonal initialization for better gradient flow
        self.query_proj = nn.Linear(self.embed_dim, self.retrieval_dim)
        self.key_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        self.value_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        
        # Initialize projection layers with orthogonal weights
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.value_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        
        # Attention mechanism with dropout for regularization
        self.cross_attention = nn.MultiheadAttention(
            self.retrieval_dim, num_heads=8, batch_first=True, dropout=self.dropout
        )
        
        # Combined dimension
        self.output_dim_combined = self.embed_dim + self.retrieval_dim
        
        # Output projection with layer normalization for stability
        self.out_proj = nn.Sequential(
            nn.Linear(self.output_dim_combined, self.output_dim_combined),
            nn.LayerNorm(self.output_dim_combined),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim_combined, self.horizon * self.output_dim) # [ 24 + 8 => 12 * 1]
        )
        
        # Initialize output projection layers
        for m in self.out_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Spatio-temporal feature processing
        # 1. Define fusion dimension
        self.fusion_dim = self.temporal_dim + self.spatial_dim  
        
        # 2. Node embeddings with Xavier initialization
        self.spatial_node_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.spatial_dim))
        nn.init.xavier_uniform_(self.spatial_node_embeddings)
        
        # 3. Temporal series encoding with proper initialization
        self.temporal_series_encoder = nn.Conv2d(
            in_channels=self.input_dim * self.seq_len,
            out_channels=self.temporal_dim,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.temporal_series_encoder.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.temporal_series_encoder.bias)
        
        # 4. Feature encoding network with residual connections for better gradient flow
        self.feature_encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for _ in range(self.encoder_layers)
        ])
        
        # Initialize feature encoder layers
        for layer in self.feature_encoder_layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # 5. Regression layer with proper initialization
        self.regression_layer = nn.Linear(self.fusion_dim, self.horizon)
        nn.init.xavier_normal_(self.regression_layer.weight)
        nn.init.zeros_(self.regression_layer.bias)
        
        # 6. Prediction generator with proper initialization
        self.prediction_generator = nn.Conv2d(
            in_channels=self.fusion_dim,
            out_channels=self.horizon,
            kernel_size=(1, 1),
            bias=True
        )
        nn.init.kaiming_normal_(self.prediction_generator.weight, mode='fan_out', nonlinearity='linear')
        nn.init.zeros_(self.prediction_generator.bias)
        
        # 7. Output adapter with proper initialization
        self.output_projector = nn.Linear(self.output_dim, self.output_dim)
        nn.init.xavier_normal_(self.output_projector.weight)
        nn.init.zeros_(self.output_projector.bias)
        
        # Hidden to embedding projection with proper initialization
        self.hidden_to_embed_proj = nn.Linear(self.fusion_dim, self.embed_dim)
        nn.init.xavier_normal_(self.hidden_to_embed_proj.weight)
        nn.init.zeros_(self.hidden_to_embed_proj.bias)

    def _init_ablation_components(self):
        """Initialize components for different ablation modes with proper regularization"""
        # Use original dimensions (matching pre-trained weights)
        self.data_embed_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 2, self.output_dim_combined)
        )
        
        # For only_retrieval_embed mode - use original dimensions
        self.retrieval_embed_mlp = nn.Sequential(
            nn.Linear(self.retrieval_dim, self.retrieval_dim * 2),
            nn.LayerNorm(self.retrieval_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.retrieval_dim * 2, self.output_dim_combined)
        )
        
        # For full mode - use original dimensions
        self.combined_embed_mlp = nn.Sequential(
            nn.Linear(self.output_dim_combined, self.output_dim_combined),
            nn.LayerNorm(self.output_dim_combined),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim_combined, self.output_dim_combined)
        )
        
        # Initialize all MLP layers with proper weight initialization
        for module in [self.data_embed_mlp, self.retrieval_embed_mlp, self.combined_embed_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_llm_encoder(self):
        """Initialize LLM encoder"""
        try:
            print(f"Initializing LLM Encoder with model: {self.llm_model}")
            self.llm_encoder = LLMEncoder(model_name=self.llm_model)
            self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
            print("LLM Encoder initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM Encoder: {e}")
            print("Using a simplified encoder instead")
            self.llm_encoder = nn.Sequential(
                nn.Linear(self.retrieval_dim, self.llm_dim),
                nn.ReLU(),
                nn.Linear(self.llm_dim, self.llm_dim)
            )
            self.llm_proj = nn.Linear(self.llm_dim, self.retrieval_dim)
    
    def _freeze_components(self):
        """Freeze pre-trained components"""
        # Freeze pre-trained encoder
        for param in self.pre_train_encoder.parameters():
            param.requires_grad = False
        
        # Freeze LLM encoder
        for param in self.llm_encoder.parameters():
            param.requires_grad = False
            
        print("Pre-trained components frozen")

    def _load_weight(self):
        """Load pre-trained weights or initialize new encoder"""
        encoder = nn.ModuleList([
            TransformerEncoder(self.embed_dim, num_heads=8)
            for _ in range(self.encoder_layers)
        ])
        
        if not self.from_scratch and self.pre_train_path:
            try:
                state_dict = torch.load(self.pre_train_path)
                encoder.load_state_dict(state_dict)
                print(f"Successfully loaded pre-trained weights from {self.pre_train_path}")
            except Exception as e:
                print(f"Failed to load pre-trained weights: {e}")
                print("Training from scratch instead")
        
        return encoder
            
    def _generate_prompt(self, domain: str, data: torch.Tensor) -> Tuple[List[str], List[str]]:
        """Generate prompts for temporal and spatial dimensions
        
        Args:
            domain: Domain identifier
            data: Input tensor of shape [B, T, N, D]
            
        Returns:
            Tuple of (temporal_prompts, spatial_prompts):
            - temporal_prompts: List[str] of length T, prompts for each timestep
            - spatial_prompts: List[str] of length N, prompts for each node
        """
        B, T, N, D = data.shape
        device = data.device
        
        # Average across batch dimension to get [T, N, D]
        data_mean = torch.mean(data, dim=0)
        
        # 1. Generate temporal prompts
        temporal_prompts = []
        for t in range(T):
            # Get data for current timestep [N, D]
            time_data = data_mean[t]
            
            # Calculate statistics for current timestep
            t_mean = torch.mean(time_data, dim=0)  # [D]
            t_std = torch.std(time_data, dim=0)    # [D]
            
            # Find max/min nodes for each feature dimension
            max_nodes, _ = torch.max(time_data, dim=0)  # [D]
            min_nodes, _ = torch.min(time_data, dim=0)  # [D]
            
            # Find most significant nodes based on variance
            node_variance = torch.var(time_data, dim=1)  # [N]
            _, top_k_nodes = torch.topk(node_variance, min(self.top_k, N))
            
            # Generate temporal prompt
            prompt = (
                f"<|domain|>{domain} prediction task<|domain_end|>\n"
                f"<|temporal_info|>\n"
                f"Timestep {t}: Global pattern\n"
                f"Average values: {t_mean.tolist()}\n"
                f"Pattern variation: {t_std.tolist()}\n"
                f"Max feature values: {max_nodes.tolist()}\n"
                f"Min feature values: {min_nodes.tolist()}\n"
                f"Most active nodes: {top_k_nodes.tolist()}\n"
                f"<|temporal_info_end|>\n"
            )
            temporal_prompts.append(prompt)
        
        # 2. Generate spatial prompts
        spatial_prompts = []
        # Transpose to [N, T, D] for processing each node's time series
        node_data_mean = data_mean.transpose(0, 1)
        
        for n in range(N):
            # Get time series data for current node [T, D]
            node_data = node_data_mean[n]
            
            # Calculate statistics for current node
            n_mean = torch.mean(node_data, dim=0)  # [D]
            n_std = torch.std(node_data, dim=0)    # [D]
            
            # Find peak times for each feature dimension
            max_times, _ = torch.max(node_data, dim=0)  # [D]
            min_times, _ = torch.min(node_data, dim=0)  # [D]
            
            # Find most active time periods
            time_variance = torch.var(node_data, dim=1)  # [T]
            _, top_k_times = torch.topk(time_variance, min(self.top_k, T))
            
            # Generate spatial prompt
            prompt = (
                f"<|domain|>{domain} prediction task<|domain_end|>\n"
                f"<|spatial_info|>\n"
                f"Node {n}: Temporal pattern\n"
                f"Average values: {n_mean.tolist()}\n"
                f"Pattern variation: {n_std.tolist()}\n"
                f"Max feature times: {max_times.tolist()}\n"
                f"Min feature times: {min_times.tolist()}\n"
                f"Most active times: {top_k_times.tolist()}\n"
                f"<|spatial_info_end|>\n"
            )
            spatial_prompts.append(prompt)
        
        # 添加缓存机制
        cache_key = f"{domain}_{data.shape}_{torch.mean(data).item():.4f}"
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        
        # 存入缓存
        self.prompt_cache[cache_key] = (temporal_prompts, spatial_prompts)
        # 限制缓存大小
        if len(self.prompt_cache) > 100:  # 保留最近的100个结果
            self.prompt_cache.pop(next(iter(self.prompt_cache)))
        
        return temporal_prompts, spatial_prompts
        
    @torch.no_grad()
    def update_retrieval_store(self, values: torch.Tensor, epoch: int):
        """Update retrieval store with new values
        
        Args:
            values: Input tensor of shape [B, T, N, D]
            epoch: Current training epoch
        """
        try:
            # Check if vectors already exist in the store
            has_existing_vectors = len(self.retrieval_store.temporal_vectors) > 0 and len(self.retrieval_store.spatial_vectors) > 0
            
            # Generate prompts for temporal and spatial dimensions
            temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, values)
            
            # Optimize batch size for processing
            batch_size = 128  # Increased batch size for better efficiency
            device = next(self.parameters()).device
            
            # Process all prompts in a single loop to reduce redundancy
            all_prompts = temporal_prompts + spatial_prompts
            all_embeddings = []
            
            # Use mixed precision computation for better performance
            with torch.cuda.amp.autocast():
                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i:i+batch_size]
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    # Project to retrieval dimension
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    all_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all embeddings
            all_embeddings = np.vstack(all_embeddings)
            
            # Separate temporal and spatial embeddings
            temporal_embeddings = all_embeddings[:len(temporal_prompts)]
            spatial_embeddings = all_embeddings[len(temporal_prompts):]
            
            # Update retrieval store using incremental update if available
            if has_existing_vectors and hasattr(self.retrieval_store, 'incremental_update'):
                # Use incremental update method
                self.retrieval_store.incremental_update(
                    temporal_embeddings, 
                    spatial_embeddings,
                    max_vectors=1000  # Limit vector count to control memory usage
                )
            else:
                # Full update of retrieval store and rebuild indices
                self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
                self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
                self.retrieval_store._rebuild_indices()
            
            # Save documents for future use
            self.retrieval_store.save_documents(self.domain, epoch, temporal_prompts, spatial_prompts)
            
            # Log statistics
            print(f"Successfully updated retrieval store at epoch {epoch}")
            print(f"Statistics: {self.retrieval_store.get_statistics()}")
            
        except Exception as e:
            print(f"Error in update_retrieval_store: {e}")
            traceback.print_exc()

    def retrieve(self, query_embed: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant content from spatio-temporal information store with optimized processing"""
        B, L, N, E = query_embed.shape
        device = query_embed.device
        
        # 1. Check if retrieval store is initialized
        if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
            if not getattr(self.retrieval_store, 'warning_shown', False):
                print("Warning: Initializing retrieval store")
                self.retrieval_store.warning_shown = True
                success = self.retrieval_store.load_documents(self.domain)
                
                if not success:
                    print("Failed to load documents, using empty store")
        
        # 2. If still empty, return zero tensor
        if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
            return torch.zeros(B, L, N, self.retrieval_dim).to(device)

        try:
            # Check vector dimensions
            if len(self.retrieval_store.temporal_vectors) > 0:
                actual_dim = len(self.retrieval_store.temporal_vectors[0])
                if actual_dim != self.retrieval_dim:
                    print(f"Warning: Dimension mismatch in retrieval store. Expected {self.retrieval_dim}, got {actual_dim}.")
                    print("Reinitializing retrieval store with correct dimensions...")
                    # Clear existing vectors
                    self.retrieval_store.temporal_vectors = []
                    self.retrieval_store.spatial_vectors = []
                    # Initialize with current data
                    self._initialize_retrieval_store(history_data, 0)
                    # If still empty after reinitialization, return zeros
                    if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
                        return torch.zeros(B, L, N, self.retrieval_dim).to(device)
            
            # 3. Optimize index structure (if not already optimized)
            if not hasattr(self.retrieval_store, 'indices_optimized'):
                self.retrieval_store.optimize_indices()
                self.retrieval_store.indices_optimized = True
            
            # 4. Use provided query vectors
            query = query_embed
            
            # 5. Convert to numpy array for retrieval - use more efficient reshaping
            query_flat = query.reshape(-1, self.retrieval_dim)
            query_np = query_flat.detach().cpu().numpy().astype(np.float32)
            
            # 6. Determine retrieval quantity
            k_temporal = min(self.top_k, len(self.retrieval_store.temporal_vectors))
            k_spatial = min(self.top_k, len(self.retrieval_store.spatial_vectors))
            
            # 7. Execute retrieval - use parallel processing
            import concurrent.futures
            
            def search_temporal():
                return self.retrieval_store.search(query_np, k=k_temporal, temporal=True)
                
            def search_spatial():
                return self.retrieval_store.search(query_np, k=k_spatial, temporal=False)
            
            # Execute searches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_temporal = executor.submit(search_temporal)
                future_spatial = executor.submit(search_spatial)
                
                temporal_distances, temporal_indices = future_temporal.result()
                spatial_distances, spatial_indices = future_spatial.result()
            
            # 8. Process retrieval results - use vectorized operations
            k_total = k_temporal + k_spatial
            total_queries = query_np.shape[0]
            
            # Pre-allocate memory for efficiency
            all_vectors = np.zeros((total_queries, k_total, self.retrieval_dim), dtype=np.float32)
            
            # Get temporal vectors (vectorized operation)
            if k_temporal > 0:
                # Ensure indices are within valid range
                valid_indices = np.clip(temporal_indices, 0, len(self.retrieval_store.temporal_vectors)-1)
                temporal_vectors = np.array([self.retrieval_store.temporal_vectors[i] for i in valid_indices.flatten()])
                
                # Verify dimensions
                vector_dim = temporal_vectors.shape[-1]
                if vector_dim != self.retrieval_dim:
                    # Handle dimension mismatch by padding or truncating
                    if vector_dim < self.retrieval_dim:
                        # Pad with zeros
                        padded_vectors = np.zeros((temporal_vectors.shape[0], self.retrieval_dim), dtype=np.float32)
                        padded_vectors[:, :vector_dim] = temporal_vectors
                        temporal_vectors = padded_vectors
                    else:
                        # Truncate
                        temporal_vectors = temporal_vectors[:, :self.retrieval_dim]
                
                temporal_vectors = temporal_vectors.reshape(total_queries, k_temporal, -1)
                all_vectors[:, :k_temporal, :] = temporal_vectors
            
            # Get spatial vectors (vectorized operation)
            if k_spatial > 0:
                # Ensure indices are within valid range
                valid_indices = np.clip(spatial_indices, 0, len(self.retrieval_store.spatial_vectors)-1)
                spatial_vectors = np.array([self.retrieval_store.spatial_vectors[i] for i in valid_indices.flatten()])
                
                # Verify dimensions
                vector_dim = spatial_vectors.shape[-1]
                if vector_dim != self.retrieval_dim:
                    # Handle dimension mismatch by padding or truncating
                    if vector_dim < self.retrieval_dim:
                        # Pad with zeros
                        padded_vectors = np.zeros((spatial_vectors.shape[0], self.retrieval_dim), dtype=np.float32)
                        padded_vectors[:, :vector_dim] = spatial_vectors
                        spatial_vectors = padded_vectors
                    else:
                        # Truncate
                        spatial_vectors = spatial_vectors[:, :self.retrieval_dim]
                
                spatial_vectors = spatial_vectors.reshape(total_queries, k_spatial, -1)
                all_vectors[:, k_temporal:, :] = spatial_vectors
            
            # 9. Convert to tensor
            retrieved_vectors = torch.tensor(all_vectors, dtype=torch.float32, device=device)
            retrieved_vectors = retrieved_vectors.reshape(B, L, N, k_total, self.retrieval_dim)
            
            # 10. Apply attention mechanism
            retrieved_keys = self.key_proj(retrieved_vectors)
            retrieved_values = self.value_proj(retrieved_vectors)
            
            # 11. Reshape tensors for attention calculation
            query_reshaped = query.reshape(B*L*N, 1, self.retrieval_dim)
            keys_reshaped = retrieved_keys.reshape(B*L*N, k_total, self.retrieval_dim)
            values_reshaped = retrieved_values.reshape(B*L*N, k_total, self.retrieval_dim)
            
            # 12. Execute attention - use batching for efficiency
            batch_size = 1024
            if B*L*N > batch_size:
                # Batch attention
                retrieved_embed_list = []
                for i in range(0, B*L*N, batch_size):
                    end_idx = min(i + batch_size, B*L*N)
                    batch_query = query_reshaped[i:end_idx]
                    batch_keys = keys_reshaped[i:end_idx]
                    batch_values = values_reshaped[i:end_idx]
                    
                    batch_embed, _ = self.cross_attention(batch_query, batch_keys, batch_values)
                    retrieved_embed_list.append(batch_embed)
                
                retrieved_embed = torch.cat(retrieved_embed_list, dim=0)
            else:
                retrieved_embed, _ = self.cross_attention(query_reshaped, keys_reshaped, values_reshaped)
            
            # 13. Reshape result
            retrieved_embed = retrieved_embed.reshape(B, L, N, self.retrieval_dim)
            
            return retrieved_embed
            
        except Exception as e:
            print(f"Error during retrieval process: {e}")
            traceback.print_exc()
            return torch.zeros(B, L, N, self.retrieval_dim).to(device)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> dict:
        """
        Forward pass of the model, handling different output modes
        
        Args:
            history_data: Historical input data [B, L, N, C]
            future_data: Future data (for teacher forcing)
            batch_seen: Number of batches processed
            epoch: Current training epoch
            train: Whether in training mode
            
        Returns:
            Dictionary containing prediction results [B, horizon, N, D]
        """
        B, L, N, D = history_data.shape
        device = history_data.device
        
        # 1. Basic data processing - shared across all modes
        input_data = history_data[..., range(self.input_dim)]
        
        # 2. Temporal feature processing
        # Transform shape to fit convolutional layer [B, C*L, N, 1]
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(B, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        data_embed = self.temporal_series_encoder(input_data)
        
        # 3. Node embedding processing
        node_emb = self.spatial_node_embeddings.unsqueeze(0).expand(
            B, -1, -1).transpose(1, 2).unsqueeze(-1)
        
        # 4. Feature fusion
        hidden = torch.cat([data_embed, node_emb], dim=1)
        
        # 5. Feature encoding with residual connections
        hidden_permuted = hidden.squeeze(-1).transpose(1, 2)  # [B, N, fusion_dim]
        
        # Apply feature encoding layers with residual connections
        for layer in self.feature_encoder_layers:
            residual = hidden_permuted
            hidden_permuted = layer(hidden_permuted)
            hidden_permuted = hidden_permuted + residual  # Add residual connection
        # print("hidden_permuted.shape",hidden_permuted.shape)
        
        # Process based on output mode
        if self.output_type == "only_data_embed":
            # 6a. Data embedding only mode
            # Generate predictions [B, N, horizon]
            prediction = self.regression_layer(hidden_permuted)
            
            # Adjust output dimensions
            prediction = prediction.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)
            prediction = self.output_projector(prediction.reshape(-1, self.output_dim))
            prediction = prediction.reshape(B, self.num_nodes, self.horizon, self.output_dim)
            
            # Convert to expected output format [B, horizon, N, output_dim]
            prediction = prediction.permute(0, 2, 1, 3)
            
            return {'prediction': prediction}
        
        else:  # "full" or "only_retrieval_embed" mode
            # 6b. Retrieval-augmented mode
            
            # Project hidden features to embedding dimension
            query_embed = self.hidden_to_embed_proj(hidden_permuted)
            
            # Expand to required shape [B, L, N, embed_dim]
            query_embed = query_embed.unsqueeze(1).expand(-1, L, -1, -1)
            
            # Initialize retrieval embeddings
            retrieval_embed = torch.zeros(B, L, N, self.retrieval_dim).to(device)
            
            # Retrieval processing
            if self.output_type in ["full", "only_retrieval_embed"]:
                # Check if dimensions have changed
                if len(self.retrieval_store.temporal_vectors) > 0:
                    actual_dim = len(self.retrieval_store.temporal_vectors[0])
                    if actual_dim != self.retrieval_dim:
                        print(f"Dimension mismatch detected. Clearing retrieval store. Expected: {self.retrieval_dim}, Got: {actual_dim}")
                        # Clear existing vectors
                        self.retrieval_store.temporal_vectors = []
                        self.retrieval_store.spatial_vectors = []
                        # Reset flags
                        if hasattr(self, 'last_update_epoch'):
                            delattr(self, 'last_update_epoch')
                        if hasattr(self, 'eval_loaded'):
                            delattr(self, 'eval_loaded')
                
                # Initialize or load retrieval store
                if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
                    # First try to load from file
                    loaded = self.retrieval_store.load_documents(self.domain)
                    if not loaded:
                        # If loading fails, initialize
                        self._initialize_retrieval_store(history_data, epoch)
                        # Record last update epoch
                        self.last_update_epoch = epoch
                
                # Update retrieval store - only in training mode and at update intervals
                if self.output_type == "full" and train:
                    # Check if update is needed (based on epoch interval and batch start)
                    if not hasattr(self, 'last_update_epoch') or \
                       (epoch % self.update_interval == 0 and 
                        epoch != getattr(self, 'last_update_epoch', -1) and 
                        batch_seen == 0):
                        
                        print(f"Updating retrieval store at epoch {epoch}")
                        self.update_retrieval_store(history_data, epoch)
                        # Record last update epoch
                        self.last_update_epoch = epoch
                        # Set flag indicating update
                        self.store_updated = True
                elif not train and batch_seen == 0 and not hasattr(self, 'eval_loaded'):
                    # In evaluation mode, only load latest documents on first batch
                    print("Loading latest retrieval documents for evaluation")
                    self.retrieval_store.load_documents(self.domain)
                    # Set flag indicating loaded
                    self.eval_loaded = True

                # Retrieve relevant embeddings
                query = self.query_proj(query_embed)
                retrieval_embed = self.retrieve(query, history_data)
                #print("retrieval_embed.shape",retrieval_embed.shape)

            # Process embeddings based on mode
            if self.output_type == "only_retrieval_embed":
                # Only use retrieval embeddings
                final_embed = self.retrieval_embed_mlp(retrieval_embed)
            else:  # "full" mode
                # Combine both embeddings
                combined = torch.cat([query_embed, retrieval_embed], dim=-1)
                final_embed = self.combined_embed_mlp(combined)
            
            #print("final_embed.shape",final_embed.shape)
            # Generate predictions
            B_new, L_new, N_new, E_new = final_embed.shape
            
            # Project to output space
            final_embed_flat = final_embed.reshape(-1, E_new)
            out_flat = self.out_proj(final_embed_flat)
            
            # Reshape output
            out = out_flat.reshape(B_new, L_new, N_new, self.horizon * self.output_dim)
            out = out.mean(dim=1)  # Aggregate time dimension
            out = out.reshape(B_new, N_new, self.horizon, self.output_dim)
            out = out.permute(0, 2, 1, 3)  # [B, horizon, N, output_dim]
            
            #print("out.shape",out.shape)
            if train and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    return {'prediction': out}
            else:
                return {'prediction': out}

    def _initialize_retrieval_store(self, history_data: torch.Tensor, epoch: int):
        """Initialize retrieval store with current batch data
        
        This method creates initial temporal and spatial embeddings for the retrieval store
        using the current batch data. It's called when the retrieval store is empty.
        
        Args:
            history_data: Historical input data [B, L, N, C]
            epoch: Current training epoch
        """
        try:
            # Generate temporal and spatial prompts
            temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, history_data)
            
            # Initialize empty lists for embeddings
            temporal_embeddings = []
            spatial_embeddings = []
            
            # Process prompts in batches for efficiency
            batch_size = 128  # Increased batch size for better efficiency
            device = next(self.parameters()).device
            
            # Process temporal prompts
            for i in range(0, len(temporal_prompts), batch_size):
                batch_prompts = temporal_prompts[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    temporal_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Process spatial prompts
            for i in range(0, len(spatial_prompts), batch_size):
                batch_prompts = spatial_prompts[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    batch_embeddings = self.llm_proj(batch_embeddings)
                    spatial_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Concatenate all embeddings
            if temporal_embeddings:
                temporal_embeddings = np.vstack(temporal_embeddings)
            else:
                temporal_embeddings = np.zeros((0, self.retrieval_dim))
            
            if spatial_embeddings:
                spatial_embeddings = np.vstack(spatial_embeddings)
            else:
                spatial_embeddings = np.zeros((0, self.retrieval_dim))
            
            # Update retrieval store
            self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
            self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
            
            # Update temporal and spatial values using the update_patterns method
            self.retrieval_store.update_patterns(history_data)
            
            # Rebuild indices
            self.retrieval_store._rebuild_indices()
            
            # Save documents
            self.retrieval_store.save_documents(
                self.domain, 
                epoch, 
                temporal_prompts, 
                spatial_prompts
            )
            
            print(f"Successfully initialized retrieval store at epoch {epoch}")
            print(f"Statistics: {self.retrieval_store.get_statistics()}")
            
        except Exception as e:
            print(f"Error initializing retrieval store: {e}")
            traceback.print_exc()
            # Create empty store as fallback
            self.retrieval_store.temporal_vectors = []
            self.retrieval_store.spatial_vectors = []
            self.retrieval_store.temporal_values = []
            self.retrieval_store.spatial_values = []