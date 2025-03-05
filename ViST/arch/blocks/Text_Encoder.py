import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, AutoModel
import os
import time
import hashlib

class TextEncoder(nn.Module):
    """
    Text encoder that handles the complete pipeline from data to text representations
    with clearly separated computational stages:
    1. Generate text prompts from spatio-temporal data
    2. Convert prompts to embeddings using language model
    3. Transform embeddings to text features for downstream tasks
    """
    def __init__(self, config):
        super().__init__()
        # Configuration parameters
        self.config = config
        self.llm_dim = config['llm_dim']
        self.horizon = config.get('horizon', 12)
        self.num_nodes = config['num_nodes']
        self.d_ff = config.get('d_ff', 32)
        self.dropout = config.get('dropout', 0.1)
        self.top_k = config.get('top_k', 5)
        
        # Model setup - lightweight, on-demand loading
        self.model_name = config['llm_model']
        self.tokenizer = None
        self.llm_model = None
        self._model_loaded = False
        
        # Text feature processor
        self.text_processor = nn.Sequential(
            nn.Linear(self.llm_dim, self.d_ff),
            nn.LayerNorm(self.d_ff),
            nn.GELU()
        )
    
        # Efficient caching for each step
        self._prompt_cache = {}
        self._embedding_cache = {}
        self._feature_cache = {}
        self._cache_size_limit = 100
        self._last_cache_clear = time.time()
        self._cache_ttl = 3600  # Cache time-to-live in seconds
    
    def generate_prompts(self, history_data, data_description=""):
        """
        Generate text prompts from spatio-temporal data statistics
        
        Args:
            history_data: Tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            data_description: Optional dataset description
            
        Returns:
            List of prompt strings
        """
        batch_size, seq_len, num_nodes, input_dim = history_data.shape
        
        # Fast hash-based caching
        try:
            # Use tensor properties instead of full content for faster hashing
            data_hash = f"{history_data[:, -1].mean().item():.3f}_{history_data[:, -1].std().item():.3f}"
            cache_key = f"{data_hash}_{data_description}_{batch_size}"
            
            if cache_key in self._prompt_cache:
                return self._prompt_cache[cache_key]
        except:
            pass

        # Efficient statistics calculation
        with torch.no_grad():
            # Only use last timestep for speed
            x = history_data[:, -1]  # [batch_size, num_nodes, input_dim]
            
            # Fast vectorized operations
            min_vals = x.amin(dim=(1, 2))  # [batch_size]
            max_vals = x.amax(dim=(1, 2))  # [batch_size]
            mean_vals = x.mean(dim=(1, 2))  # [batch_size]
            std_vals = x.std(dim=(1, 2))  # [batch_size]
            
            # Quick node importance calculation
            node_avgs = x.mean(dim=2)  # [batch_size, num_nodes]
            top_vals, top_idxs = torch.topk(node_avgs, k=min(self.top_k, num_nodes), dim=1)
        
        # Generate prompts efficiently
        prompts = []
        for b in range(batch_size):
            # Only include top 3 nodes max to keep prompt small
            top_nodes = ", ".join([f"N{top_idxs[b, i].item()}:{top_vals[b, i].item():.2f}" 
                                  for i in range(min(3, self.top_k))])
            
            # Create a simple prompt with essential information
            prompt = (
                f"Dataset: {data_description} Time: {seq_len-1}: "
                f"min={min_vals[b].item():.2f}, max={max_vals[b].item():.2f}, "
                f"mean={mean_vals[b].item():.2f}, std={std_vals[b].item():.2f}, "
                f"horizon={self.horizon}, top=[{top_nodes}]"
            )
            prompts.append(prompt)
        
        # Cache results
        try:
            self._prompt_cache[cache_key] = prompts
            self._manage_cache_size()
        except:
            pass
            
        return prompts
    
    def _load_language_model(self, device):
        """Lazy-load language model only when needed"""
        if self._model_loaded:
            return True
        try:
            # Try to load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                use_fast=True,
                model_max_length=128
            )

            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            # Try to load model with optimizations
            try:
                self.llm_model = BertModel.from_pretrained(
                    self.model_name,
                    torchscript=True,
                    return_dict=False
                )
            except:
                print(f"Local model not found. Downloading {self.model_name}...")
                self.llm_model = BertModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=False
                )
                
            # Move model to device
            self.llm_model = self.llm_model.to(device)
                
            # Freeze LLM parameters
            for param in self.llm_model.parameters():
                param.requires_grad = False
                
            self._model_loaded = True
            return True
                
        except Exception as e:
            print(f"Warning: Could not load transformer models: {e}")
            # Keep track that we tried but failed
            self._model_loaded = False
            return False
    
    def get_text_embeddings(self, prompts, device=None):
        """
        Convert text prompts to embeddings using language model
        
        Args:
            prompts: List of prompt strings
            device: Optional device to place tensors on
            
        Returns:
            Tensor of shape [batch_size, llm_dim]
        """
        if device is None and isinstance(prompts, torch.Tensor):
            device = prompts.device
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Quick check - if prompts are identical, provide identical result
        if all(p == prompts[0] for p in prompts):
            # Process just one and duplicate
            single_embedding = self._get_single_embedding(prompts[0], device)
            return single_embedding.repeat(len(prompts), 1)
            
        # Check cache using a faster hash
        try:
            # Just hash the first prompt with length info for speed
            cache_key = f"{hash(prompts[0])}_{len(prompts)}"
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key].to(device)
        except:
            pass
        
        # Load model if needed
        model_loaded = self._load_language_model(device)
        
        # Fast fallback if model not available
        if not model_loaded:
            batch_size = len(prompts)
            random_seed = int(hash(str(prompts[0])) % 10000)
            torch.manual_seed(random_seed)
            # Create deterministic but varied embeddings
            return torch.randn(batch_size, self.llm_dim, device=device) * 0.1
            
        # Efficient tokenization with batch processing
        with torch.no_grad():
            tokens = self.tokenizer(
                prompts,
                padding='max_length',
                max_length=min(128, self.d_ff),
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = self.llm_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                return_dict=True
            )
            
            # Use mean pooling for efficiency
            attention_mask = tokens.get("attention_mask").unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / (attention_mask.sum(1) + 1e-10)
            
        # Cache embeddings
        try:
            self._embedding_cache[cache_key] = embeddings.detach().cpu()
            self._manage_cache_size()
        except:
            pass
        
        return embeddings
    
    def _get_single_embedding(self, prompt, device):
        """Process a single prompt for efficiency"""
        model_loaded = self._load_language_model(device)
        
        if not model_loaded:
            # Deterministic random embedding
            random_seed = int(hash(prompt) % 10000)
            torch.manual_seed(random_seed)
            return torch.randn(1, self.llm_dim, device=device) * 0.1
            
        with torch.no_grad():
            tokens = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=min(128, self.d_ff),
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = self.llm_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                return_dict=True
            )
            
            attention_mask = tokens.get("attention_mask").unsqueeze(-1)
            embedding = (outputs.last_hidden_state * attention_mask).sum(1) / (attention_mask.sum(1) + 1e-10)
            
        return embedding
    
    def get_text_features(self, text_embeddings):
        """
        Convert text embeddings to features for downstream tasks
        
        Args:
            text_embeddings: Tensor of shape [batch_size, llm_dim]
            
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        device = text_embeddings.device
        
        # Check cache using embedding properties
        try:
            embedding_key = f"{text_embeddings.mean().item():.4f}_{text_embeddings.std().item():.4f}"
            if embedding_key in self._feature_cache:
                return self._feature_cache[embedding_key].to(device)
        except:
            pass
            
        # Process embeddings to features
        text_features = self.text_processor(text_embeddings)  # [batch_size, hidden_dim]
        
        # Cache features
        try:
            self._feature_cache[embedding_key] = text_features.detach().cpu()
            self._manage_cache_size()
        except:
            pass
            
        return text_features
        
    def _manage_cache_size(self):
        """Efficiently manage cache size"""
        current_time = time.time()
        
        # Simple TTL-based clearing
        if current_time - self._last_cache_clear > self._cache_ttl:
            self._prompt_cache.clear()
            self._embedding_cache.clear()
            self._feature_cache.clear()
            self._last_cache_clear = current_time
            return
            
        # Keep cache size in check
        if len(self._prompt_cache) > self._cache_size_limit:
            self._prompt_cache = dict(list(self._prompt_cache.items())[-self._cache_size_limit//2:])
            
        if len(self._embedding_cache) > self._cache_size_limit:
            self._embedding_cache = dict(list(self._embedding_cache.items())[-self._cache_size_limit//2:])
            
        if len(self._feature_cache) > self._cache_size_limit:
            self._feature_cache = dict(list(self._feature_cache.items())[-self._cache_size_limit//2:])
    
    def forward(self, history_data=None, prompts=None, text_embeddings=None, data_description=""):
        """
        Complete pipeline from data to text features with multiple entry points.
        Each stage is clearly separated and can be called independently.
        
        Args:
            history_data: Optional tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            prompts: Optional list of prompt strings (if history_data not provided)
            text_embeddings: Optional tensor of shape [batch_size, llm_dim] (if prompts not provided)
            data_description: Optional dataset description for prompt generation
            
        Returns:
            text_features: Tensor of shape [batch_size, hidden_dim]
            prompts: List of prompt strings (if generated from history_data)
            text_embeddings: Tensor of shape [batch_size, llm_dim] (if generated)
        """
        device = None
        
        # Build complete pipeline based on provided entry point
        if history_data is not None:
            device = history_data.device
            # Step 1: Generate prompts from data
            prompts = self.generate_prompts(history_data, data_description)
            
        if text_embeddings is None and prompts is not None:
            # Step 2: Convert prompts to embeddings
            if device is None and isinstance(prompts, torch.Tensor):
                device = prompts.device
            elif device is None and len(prompts) > 0:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            text_embeddings = self.get_text_embeddings(prompts, device)
            
        if text_embeddings is not None:
            # Step 3: Convert embeddings to features
            text_features = self.get_text_features(text_embeddings)
            
            # Return all calculated components
            # This allows the caller to access any intermediate results they need
            results = (text_features,)
            if prompts is not None:
                results += (prompts,)
            if text_embeddings is not None:
                results += (text_embeddings,)
                
            # If only one result, return it directly
            if len(results) == 1:
                return results[0]
            return results
        
        # If we reached here, we didn't have enough inputs
        raise ValueError("Must provide at least one of: history_data, prompts, or text_embeddings")

class TextualOutputHead(nn.Module):
    """
    Processes text features to generate spatio-temporal predictions
    """
    def __init__(self, config):
        super().__init__()
        # Configuration
        self.horizon = config.get('output_len', config.get('horizon', 12))
        self.num_nodes = config['num_nodes']
        self.output_dim = config['output_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        self.d_ff = config.get('d_ff', 32)
        self.dropout = config.get('dropout', 0.1)
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.hidden_dim)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(self.d_ff, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Output projection - output horizon*output_dim dimensions directly
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.horizon * self.output_dim)
        )
        
        # Cache for inference
        self._cache = {}
        self._cache_size_limit = 50
        
        print(f"TextualOutputHead initialized with horizon={self.horizon}, num_nodes={self.num_nodes}, output_dim={self.output_dim}")
    
    def forward(self, text_features, hidden_state=None, cache_key=None):
        """
        Generate predictions from text features
        
        Args:
            text_features: [batch_size, d_ff]
            hidden_state: Optional [batch_size, embed_dim, num_nodes, output_dim]
            cache_key: Optional cache key for inference
            
        Returns:
            Predictions [batch_size, horizon, num_nodes, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # No caching - always perform full computation for consistent results
        
        # Transform text features
        text_features = self.feature_transform(text_features)  # [B, hidden_dim]
        
        # CRITICAL: Ensure node embeddings have batch dimension first
        # Expand text features for each node
        text_features = text_features.unsqueeze(1)  # [B, 1, hidden_dim]
        text_features = text_features.expand(batch_size, self.num_nodes, self.hidden_dim)  # [B, N, hidden_dim]
        
        # Get node embeddings
        node_embeddings = self.node_embeddings.unsqueeze(0)  # [1, N, hidden_dim]
        node_embeddings = node_embeddings.expand(batch_size, -1, -1)  # [B, N, hidden_dim]
        
        # Concatenate features
        combined_features = torch.cat([text_features, node_embeddings], dim=2)  # [B, N, hidden_dim*2]
        
        # Generate predictions
        predictions = self.output_projection(combined_features)  # [B, N, horizon*output_dim]
        
        # CRITICAL: Reshape to the EXACT expected output format [B, L, N, C]
        # First reshape to [B, N, L, C]
        predictions = predictions.reshape(batch_size, self.num_nodes, self.horizon, self.output_dim)
        # Then transpose to get [B, L, N, C]
        predictions = predictions.permute(0, 2, 1, 3)  # [B, horizon, num_nodes, output_dim]
        
        # Explicit shape verification with descriptive error
        expected_shape = (batch_size, self.horizon, self.num_nodes, self.output_dim)
        if predictions.shape != expected_shape:
            raise ValueError(f"Output shape error! Expected {expected_shape}, got {predictions.shape}")
        
        # Print shape for debugging
        print(f"TextualOutputHead final output shape: {predictions.shape}")
        
        return predictions