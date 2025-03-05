
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
                spatial_embed = normalize_minmax(spatial_embed)
                
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
