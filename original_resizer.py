import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SDLatentResizer(nn.Module):
    """
    Architecture for SD1.5 and SDXL models (4 channels).
    """

    def __init__(self, in_channels=4, out_channels=4, time_embed_dim=256):
        super().__init__()

        self.time_embed_dim = time_embed_dim

        # Input convolution - fixed for SD models
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)
        
        # Time embedding
        self.embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Down blocks - fixed architecture for SD models
        self.in_blocks = nn.ModuleList([
            # Block 0
            ResBlock(128, 128, time_embed_dim),
            AttentionBlock(128),
            ResBlock(128, 128, time_embed_dim),
            ResBlock(128, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            AttentionBlock(256),
            ResBlock(256, 256, time_embed_dim),
        ])

        # Up blocks
        self.out_blocks = nn.ModuleList([
            ResBlock(512, 256, time_embed_dim),
            AttentionBlock(256),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(384, 128, time_embed_dim),
            AttentionBlock(128),
            ResBlock(256, 128, time_embed_dim),
        ])

        # Output
        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, out_channels, 3, padding=1)
        
    def forward(self, x, scale=2.0):
        """Forward pass with upscaling."""
        # Create dummy timestep
        timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        
        # Time embedding
        t_emb = self.get_timestep_embedding(timesteps, self.time_embed_dim)
        emb = self.embed(t_emb)
        
        # Input
        h = self.conv_in(x)
        hs = [h]
        
        # Downsampling
        for i, module in enumerate(self.in_blocks):
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)
            hs.append(h)
        
        # Upsampling
        for i, module in enumerate(self.out_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Upscale if needed
        if scale != 1.0:
            target_size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        
        return h
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)[:, :, None, None]
        h = h + emb_out
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)
        
        h_ = torch.bmm(w_, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        
        return x + h_


class FluxLatentResizer(nn.Module):
    """
    Architecture for Flux models (16 channels).
    """

    def __init__(self, in_channels=16, out_channels=16, time_embed_dim=256):
        super().__init__()

        self.time_embed_dim = time_embed_dim

        # Input convolution - fixed for Flux models
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)

        # Time embedding
        self.embed = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Down blocks - same as SD but with 16 input channels
        self.in_blocks = nn.ModuleList([
            # Block 0
            ResBlock(128, 128, time_embed_dim),
            AttentionBlock(128),
            ResBlock(128, 128, time_embed_dim),
            ResBlock(128, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            ResBlock(256, 256, time_embed_dim),
            AttentionBlock(256),
            ResBlock(256, 256, time_embed_dim),
        ])

        # Up blocks
        self.out_blocks = nn.ModuleList([
            ResBlock(512, 256, time_embed_dim),
            AttentionBlock(256),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(512, 256, time_embed_dim),
            ResBlock(384, 128, time_embed_dim),
            AttentionBlock(128),
            ResBlock(256, 128, time_embed_dim),
        ])

        # Output
        self.norm_out = nn.GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, out_channels, 3, padding=1)

    def forward(self, x, scale=2.0):
        """Forward pass with upscaling."""
        # Create dummy timestep
        timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # Time embedding
        t_emb = self.get_timestep_embedding(timesteps, self.time_embed_dim)
        emb = self.embed(t_emb)

        # Input
        h = self.conv_in(x)
        hs = [h]

        # Downsampling
        for i, module in enumerate(self.in_blocks):
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)
            hs.append(h)

        # Upsampling
        for i, module in enumerate(self.out_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, ResBlock):
                h = module(h, emb)
            else:
                h = module(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Upscale if needed
        if scale != 1.0:
            target_size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        return h

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb
