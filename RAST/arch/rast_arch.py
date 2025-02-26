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
        self.seq_len = model_args['input_len']
        self.horizon = model_args['output_len']
        self.encoder_layers = model_args['encoder_layers']
        self.decoder_layers = model_args['decoder_layers']
        self.patch_size = model_args['patch_size']
        self.domain = model_args['prompt_domain']
        self.top_k = model_args['top_k']
        self.prompt_batch_size = model_args.get('batch_size', 32)  # Add prompt processing batch size
        
        self.debug_prompts = 1
        self.update_interval = 1
        self.output_type = 'full'

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
        
        # Training settings
        self.prompt_cache = {}  # Cache for prompt generation
        
    def _init_components(self):
        """Initialize model components"""
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(self.input_dim, self.embed_dim, self.patch_size)
        
        # Pre-trained encoder
        self.pre_train_encoder = self._load_weight()
        print(f"Pre-trained encoder initialized with {self.encoder_layers} layers")
        
        # Retrieval components
        self.retrieval_store = RetrievalStore(
            self.retrieval_dim, 
            doc_dir=self.database_path,
            max_files=5,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len
        )
        
        # Projection layers
        self.query_proj = nn.Linear(self.embed_dim, self.retrieval_dim)
        self.key_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        self.value_proj = nn.Linear(self.retrieval_dim, self.retrieval_dim)
        
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            self.retrieval_dim, num_heads=8, batch_first=True
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(self.embed_dim+self.retrieval_dim, self.embed_dim+self.retrieval_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim+self.retrieval_dim, self.horizon * self.output_dim)
        )
        
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
        
        return temporal_prompts, spatial_prompts
        
    @torch.no_grad()
    def update_retrieval_store(self, embeddings: torch.Tensor, values: torch.Tensor, epoch: int):
        """Update retrieval store with new embeddings and values"""
        try:
            # 生成时间和空间维度的prompts
            temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, values)
            
            # 处理prompts批次
            batch_size = 64
            device = next(self.parameters()).device
            
            # 处理时间维度的prompts
            temporal_embeddings = []
            for i in range(0, len(temporal_prompts), batch_size):
                batch_prompts = temporal_prompts[i:i+batch_size]
                with torch.cuda.amp.autocast():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    temporal_embeddings.append(batch_embeddings.cpu().numpy())
            temporal_embeddings = np.vstack(temporal_embeddings)
            
            # 处理空间维度的prompts
            spatial_embeddings = []
            for i in range(0, len(spatial_prompts), batch_size):
                batch_prompts = spatial_prompts[i:i+batch_size]
                with torch.cuda.amp.autocast():
                    batch_embeddings = self.llm_encoder(batch_prompts)
                    spatial_embeddings.append(batch_embeddings.cpu().numpy())
            spatial_embeddings = np.vstack(spatial_embeddings)
            
            # 更新检索存储
            self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
            self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
            self.retrieval_store._rebuild_indices()
            
        except Exception as e:
            print(f"Error in update_retrieval_store: {e}")
            traceback.print_exc()

    def retrieve(self, query_embed: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant information from temporal and spatial stores"""
        B, L, N, E = query_embed.shape
        device = query_embed.device
        
        # Initialize stores with default values if empty
        if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
            if not self.retrieval_store.warning_shown:
                print("Warning: Initializing retrieval stores with default values")
                self.retrieval_store.warning_shown = True
                self.retrieval_store.load_documents(self.domain)
            
            # 如果加载后仍然为空，直接返回零张量
            if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
                return torch.zeros(B, L, N, self.retrieval_dim).to(device)
        
        try:
            # 将query_embed投影到retrieval_dim维度
            query = self.query_proj(query_embed)  # [B, L, N, retrieval_dim]
            query_np = query.detach().cpu().numpy().astype(np.float32).reshape(-1, self.retrieval_dim)
            
            # Perform temporal and spatial retrieval
            k_temporal = min(self.top_k, len(self.retrieval_store.temporal_vectors))
            k_spatial = min(self.top_k, len(self.retrieval_store.spatial_vectors))
            
            # Get temporal and spatial neighbors
            temporal_distances, temporal_indices = self.retrieval_store.index_temporal.search(
                query_np, k_temporal
            )
            spatial_distances, spatial_indices = self.retrieval_store.index_spatial.search(
                query_np, k_spatial
            )
            
            # Pre-allocate zero vectors
            zero_vector = np.zeros(E, dtype=np.float32)
            retrieved_vectors = []
            
            # Process temporal and spatial retrievals together
            for t_idx_row, s_idx_row in zip(temporal_indices, spatial_indices):
                vectors = []
                # Add temporal vectors
                for idx in t_idx_row:
                    vectors.append(
                        self.retrieval_store.temporal_vectors[idx] 
                        if 0 <= idx < len(self.retrieval_store.temporal_vectors) 
                        else zero_vector
                    )
                # Add spatial vectors
                for idx in s_idx_row:
                    vectors.append(
                        self.retrieval_store.spatial_vectors[idx] 
                        if 0 <= idx < len(self.retrieval_store.spatial_vectors) 
                        else zero_vector
                    )
                retrieved_vectors.extend(vectors)
            
            # Convert to tensor and reshape
            retrieved_vectors = torch.tensor(
                np.array(retrieved_vectors, dtype=np.float32),
                dtype=torch.float32,
                device=device
            )
            k_total = k_temporal + k_spatial
            retrieved_vectors = retrieved_vectors.reshape(B, L, N, k_total, self.retrieval_dim)
            
            # Apply attention mechanism
            retrieved_keys = self.key_proj(retrieved_vectors)
            retrieved_values = self.value_proj(retrieved_vectors)
            
            # Reshape for cross-attention
            query_reshaped = query.reshape(B*L*N, 1, self.retrieval_dim)
            keys_reshaped = retrieved_keys.reshape(B*L*N, k_total, self.retrieval_dim)
            values_reshaped = retrieved_values.reshape(B*L*N, k_total, self.retrieval_dim)
            
            # Perform cross-attention
            retrieved_embed, _ = self.cross_attention(
                query_reshaped, keys_reshaped, values_reshaped
            )
            retrieved_embed = retrieved_embed.reshape(B, L, N, self.retrieval_dim)
            
            return retrieved_embed
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            traceback.print_exc()
            return torch.zeros(B, L, N, self.retrieval_dim).to(device)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, 
               batch_seen: int, epoch: int, train: bool, **kwargs) -> dict:
        """
        Forward pass of the model
        
        Args:
            history_data: Historical input data of shape [B, L, N, C]
            future_data: Future data for teacher forcing
            batch_seen: Number of batches processed
            epoch: Current training epoch
            train: Whether in training mode
            
        Returns:
            Dictionary containing predictions with shape [B, horizon, N, D]
        """
        B, L, N, D = history_data.shape
        
        # 1. Apply patch embedding
        x_patch = self.patch_embed(history_data)
        
        # 2. Process through pre-trained encoder
        input_embed = x_patch
        for layer in self.pre_train_encoder:
            input_embed = layer(input_embed)
        #print("shape of input_embed:", input_embed.shape) # [batch, L//patch_size, node, embed_dim]

        if self.output_type == "only_pre_train": # Ablation 1
            print(" only use pre-train model!!!")
            return input_embed
        
        # 3. Check and initialize retrieval store if empty
        if not (self.retrieval_store.temporal_vectors and self.retrieval_store.spatial_vectors):
            print("No existing documents found, initializing with current batch data")
            try:
                # 使用缓存避免重复计算
                cache_file = os.path.join(self.database_path, f"{self.domain}_init_cache.pt")
                if os.path.exists(cache_file):
                    print("Loading from initialization cache...")
                    cache_data = torch.load(cache_file)
                    self.retrieval_store.temporal_vectors = cache_data['temporal_vectors']
                    self.retrieval_store.spatial_vectors = cache_data['spatial_vectors']
                else:
                    # Update patterns with current batch data
                    self.retrieval_store.update_patterns(history_data)
                    
                    # Generate initial prompts
                    temporal_prompts, spatial_prompts = self._generate_prompt(self.domain, history_data)
                    print(f"Generating initial prompts - temporal: {len(temporal_prompts)}, spatial: {len(spatial_prompts)}")
                    
                    # 使用更大的批处理大小
                    batch_size = 128
                    device = next(self.parameters()).device
                    
                    # 并行处理temporal和spatial prompts
                    all_prompts = temporal_prompts + spatial_prompts
                    all_embeddings = []
                    
                    for i in range(0, len(all_prompts), batch_size):
                        batch_prompts = all_prompts[i:i+batch_size]
                        with torch.no_grad():
                            batch_embeddings = self.llm_encoder(batch_prompts)
                            batch_embeddings = self.llm_proj(batch_embeddings)
                            all_embeddings.append(batch_embeddings.cpu().detach().numpy())
                    
                    all_embeddings = np.vstack(all_embeddings)
                    
                    # 分离temporal和spatial embeddings
                    temporal_embeddings = all_embeddings[:len(temporal_prompts)]
                    spatial_embeddings = all_embeddings[len(temporal_prompts):]
                    
                    # Update retrieval store
                    self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
                    self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
                    
                    # 保存到缓存
                    torch.save({
                        'temporal_vectors': self.retrieval_store.temporal_vectors,
                        'spatial_vectors': self.retrieval_store.spatial_vectors
                    }, cache_file)
                
                print(f"Temporal embeddings shape: {len(self.retrieval_store.temporal_vectors)}, {self.retrieval_store.temporal_vectors[0].shape}")
                print(f"Spatial embeddings shape: {len(self.retrieval_store.spatial_vectors)}, {self.retrieval_store.spatial_vectors[0].shape}")
                
                self.retrieval_store._rebuild_indices()
                self.retrieval_store.save_documents(self.domain, epoch, temporal_prompts, spatial_prompts)
                print("Successfully initialized and saved initial retrieval store")
                
            except Exception as e:
                print(f"Error during initial store creation: {e}")
                traceback.print_exc()
        
        # 4. Regular update logic during training
        if train and epoch % self.update_interval == 0 and batch_seen == 0:
            print(f"Updating retrieval store at epoch {epoch}")
            try:
                with torch.cuda.amp.autocast():  # 使用混合精度训练
                    # 使用更大的批次处理prompts
                    batch_size = 128  # 增加批处理大小
                    
                    # 处理temporal prompts
                    temporal_embeddings = []
                    for i in range(0, len(temporal_prompts), batch_size):
                        batch_prompts = temporal_prompts[i:i+batch_size]
                        batch_embeddings = self.llm_encoder(batch_prompts)
                        batch_embeddings = self.llm_proj(batch_embeddings)
                        temporal_embeddings.append(batch_embeddings.detach().cpu())
                    temporal_embeddings = torch.cat(temporal_embeddings, dim=0).numpy()
                    
                    # 类似地处理spatial prompts
                    # ... 

                # Update retrieval store
                self.retrieval_store.temporal_vectors = [v for v in temporal_embeddings]
                self.retrieval_store.spatial_vectors = [v for v in spatial_embeddings]
                self.retrieval_store._rebuild_indices()
                self.retrieval_store.save_documents(self.domain, epoch, temporal_prompts, spatial_prompts)
                
            except Exception as e:
                print(f"Error during retrieval store update: {e}")
                traceback.print_exc()
        
        elif not train and batch_seen == 0:
            # Load latest patterns at the start of evaluation
            print("Loading latest documents for evaluation")
            self.retrieval_store.load_documents(self.domain)
        
        # 4. Generate current prompts and retrieve relevant embeddings
        try:
            retrieval_embed = self.retrieve(input_embed, history_data)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            retrieval_embed = torch.zeros(B, L // self.patch_size, N, self.retrieval_dim).to(input_embed.device)
        
        #print("shape of retrieval embed:", retrieval_embed.shape) # [batch, T//patch_size, N, retrieval_dim]
        if self.output_type == "only_retrieval":
            pass
        elif self.output_type == "without_retrieval":
            retrieval_embed = torch.zeros_like(input_embed).to(input_embed.device)

        # 5. Combine input and retrieval embeddings
        final_embed = torch.cat([input_embed, retrieval_embed], dim=-1)
        
        # 6. Generate predictions
        B_new, L_new, N_new, E_new = final_embed.shape
        
        # Project to output space
        final_embed_flat = final_embed.reshape(-1, E_new)
        out_flat = self.out_proj(final_embed_flat)
        
        # Reshape output considering patch dimensions
        out = out_flat.reshape(B_new, L_new, N_new, self.horizon * self.output_dim)
        out = out.mean(dim=1)  # Aggregate temporal dimension
        out = out.reshape(B_new, N_new, self.horizon, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # [B, horizon, N, output_dim]
        
        return {'prediction': out}