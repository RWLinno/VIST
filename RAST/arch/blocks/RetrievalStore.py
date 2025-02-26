import torch
import numpy as np
import faiss
import json
import os
import glob
import traceback
from typing import List, Dict, Union, Optional

class RetrievalStore:
    """Vector store for efficient retrieval of temporal and spatial embeddings"""
    
    def __init__(self, dim: int, doc_dir: str = "./database", max_files: int = 5,
                 num_nodes: int = None, seq_len: int = None):
        """
        Args:
            dim: Dimension of vectors to store
            doc_dir: Directory for document storage
            max_files: Maximum number of files to retain per dimension
            num_nodes: Number of spatial nodes
            seq_len: Input sequence length
        """
        # 使用基础的 IndexFlatL2 而不是 IndexIVFFlat，因为数据量较小
        self.index_temporal = faiss.IndexFlatL2(dim)
        self.index_spatial = faiss.IndexFlatL2(dim)
        
        # 如果确实需要使用 IVF，则需要以下设置：
        """
        # 创建量化器
        quantizer_temporal = faiss.IndexFlatL2(dim)
        quantizer_spatial = faiss.IndexFlatL2(dim)
        
        # 设置聚类中心数量，对于小数据集，使用较小的聚类数
        n_clusters_temporal = min(4, seq_len) if seq_len else 4  # 时间维度通常较小
        n_clusters_spatial = min(32, num_nodes) if num_nodes else 32  # 空间维度可能较大
        
        # 创建 IVF 索引
        self.index_temporal = faiss.IndexIVFFlat(
            quantizer_temporal, dim, n_clusters_temporal, faiss.METRIC_L2
        )
        self.index_spatial = faiss.IndexIVFFlat(
            quantizer_spatial, dim, n_clusters_spatial, faiss.METRIC_L2
        )
        """
        
        # Store model dimensions
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # Separate storage for temporal and spatial data
        self.temporal_vectors = []
        self.temporal_values = []
        self.spatial_vectors = []
        self.spatial_values = []
        
        self.doc_dir = doc_dir
        self.max_files = max_files
        os.makedirs(doc_dir, exist_ok=True)
        
        self.warning_shown = False
        
        # 添加训练数据缓存
        self.cache_size = 1000
        self.temporal_cache = []
        self.spatial_cache = []
    
    def save_documents(self, domain: str, epoch: int = None, temporal_prompts: List[str] = None, spatial_prompts: List[str] = None):
        """Save temporal and spatial documents"""
        if not self.temporal_vectors or not self.spatial_vectors:
            print("No data to save")
            return
        
        try:
            # 批量保存向量和统计信息
            data = {
                'temporal': {
                    'vectors': [v.tolist() if isinstance(v, np.ndarray) else v for v in self.temporal_vectors],
                    'values': self.temporal_values
                },
                'spatial': {
                    'vectors': [v.tolist() if isinstance(v, np.ndarray) else v for v in self.spatial_vectors],
                    'values': self.spatial_values
                }
            }
            
            # 使用单个文件保存所有数据
            store_file = os.path.join(self.doc_dir, f"{domain}_store_epoch_{epoch}.json")
            with open(store_file, "w") as f:
                json.dump(data, f)
            
            # 保存示例prompts（如果提供）
            if temporal_prompts and spatial_prompts:
                self.save_example_prompts(domain, temporal_prompts[:3], spatial_prompts[:3])
            
            print(f"Successfully saved retrieval store data at epoch {epoch}")
            
        except Exception as e:
            print(f"Error saving documents: {e}")
            traceback.print_exc()
    
    def load_documents(self, domain: str):
        """Load all temporal and spatial documents for the given domain"""
        # Clear existing data
        self.temporal_vectors = []
        self.temporal_values = []
        self.spatial_vectors = []
        self.spatial_values = []
        
        try:
            # Load time documents
            time_files = glob.glob(os.path.join(self.doc_dir, f"{domain}_time_store_*.json"))
            for file_path in sorted(time_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
                with open(file_path, "r") as f:
                    time_data = json.load(f)
                    self.temporal_vectors.append(np.array(time_data["vector"], dtype=np.float32))
                    self.temporal_values.append(time_data["value"])
            
            # Load space documents
            space_files = glob.glob(os.path.join(self.doc_dir, f"{domain}_space_store_*.json"))
            for file_path in sorted(space_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
                with open(file_path, "r") as f:
                    space_data = json.load(f)
                    self.spatial_vectors.append(np.array(space_data["vector"], dtype=np.float32))
                    self.spatial_values.append(space_data["value"])
            
            # Rebuild indices
            self._rebuild_indices()
            print(f"Successfully loaded {len(self.temporal_vectors)} time documents and {len(self.spatial_vectors)} space documents")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            traceback.print_exc()
    
    def _rebuild_indices(self):
        """Rebuild both temporal and spatial FAISS indices"""
        try:
            # Rebuild temporal index
            self.index_temporal.reset()
            if self.temporal_vectors:
                temporal_array = np.stack(self.temporal_vectors)
                if len(temporal_array) > 0:
                    print(f"Adding {len(temporal_array)} temporal vectors to index")
                    # 如果使用 IVF，需要先训练
                    """
                    if not self.index_temporal.is_trained:
                        print("Training temporal index...")
                        self.index_temporal.train(temporal_array)
                    """
                    self.index_temporal.add(temporal_array)
            
            # Rebuild spatial index
            self.index_spatial.reset()
            if self.spatial_vectors:
                spatial_array = np.stack(self.spatial_vectors)
                if len(spatial_array) > 0:
                    print(f"Adding {len(spatial_array)} spatial vectors to index")
                    # 如果使用 IVF，需要先训练
                    """
                    if not self.index_spatial.is_trained:
                        print("Training spatial index...")
                        self.index_spatial.train(spatial_array)
                    """
                    self.index_spatial.add(spatial_array)
                
        except Exception as e:
            print(f"Error rebuilding indices: {e}")
            traceback.print_exc()
    
    def update_patterns(self, data: torch.Tensor):
        """Update temporal and spatial patterns from input data"""
        # 使用torch操作替代numpy，减少CPU-GPU数据传输
        B, T, N, D = data.shape
        
        # 计算temporal patterns
        time_slice = data.detach()  # [B, T, N, D]
        mean = torch.mean(time_slice, dim=(0, 2))  # [T, D]
        std = torch.std(time_slice, dim=(0, 2))    # [T, D]
        
        self.temporal_values = [{
            "mean": m.cpu().numpy().tolist(),
            "std": s.cpu().numpy().tolist(),
            "timestamp": t
        } for t, (m, s) in enumerate(zip(mean, std))]
        
        # 计算spatial patterns
        node_data = data.detach().transpose(1, 2)  # [B, N, T, D]
        mean = torch.mean(node_data, dim=(0, 2))  # [N, D]
        std = torch.std(node_data, dim=(0, 2))    # [N, D]
        
        self.spatial_values = [{
            "mean": m.cpu().numpy().tolist(),
            "std": s.cpu().numpy().tolist(),
            "node_id": n
        } for n, (m, s) in enumerate(zip(mean, std))]

    def save_example_prompts(self, domain: str, temporal_prompts: List[str], spatial_prompts: List[str]):
        """Save example prompts for user inspection
        
        Args:
            domain: Domain identifier
            temporal_prompts: List of temporal prompts
            spatial_prompts: List of spatial prompts
        """
        try:
            # Create example prompts file
            example_file = os.path.join(self.doc_dir, f"{domain}_example_prompts.txt")
            
            with open(example_file, "w", encoding="utf-8") as f:
                # Write temporal prompts
                f.write("="*80 + "\n")
                f.write("TEMPORAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
                # Save first 3 temporal prompts as examples
                for i, prompt in enumerate(temporal_prompts[:3]):
                    f.write(f"Temporal Prompt {i+1}:\n")
                    f.write("-"*40 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
                
                # Write spatial prompts
                f.write("="*80 + "\n")
                f.write("SPATIAL PROMPTS EXAMPLES\n")
                f.write("="*80 + "\n\n")
                
                # Save first 3 spatial prompts as examples
                for i, prompt in enumerate(spatial_prompts[:3]):
                    f.write(f"Spatial Prompt {i+1}:\n")
                    f.write("-"*40 + "\n")
                    f.write(prompt)
                    f.write("\n\n")
            
            print(f"Saved example prompts to {example_file}")
            
        except Exception as e:
            print(f"Error saving example prompts: {e}")
            traceback.print_exc()