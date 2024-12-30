import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ChannelWiseMaxPool(nn.Module):
    def __init__(self):
        super(ChannelWiseMaxPool, self).__init__()

    def forward(self, x):
        # Apply max pooling along the channel dimension
        max_pooled, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)
        return max_pooled

class ChannelWiseAvgPool(nn.Module):
    def __init__(self):
        super(ChannelWiseAvgPool, self).__init__()

    def forward(self, x):
        # Apply avg pooling along the channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True) # (B,1,H,W)
        return avg_pool

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b c (h w)')
        g_maxpool, _ = torch.max(x, dim=-1, keepdim=True) # (B,C,1)
        return g_maxpool

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b c (h w)')
        g_avgpool = torch.mean(x, dim=-1, keepdim=True) # (B,C,1)
        return g_avgpool

class ChAttnBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch=256, reduction_ratio=0.5) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_ch),
            nn.Linear(hidden_ch, int(hidden_ch*reduction_ratio)),
            nn.Linear(int(hidden_ch*reduction_ratio), in_ch)
        )
        self.g_avgpool = GlobalAvgPool()
        self.g_maxpool = GlobalMaxPool()

    def forward(self, x):
        B = x.shape[0]
        xmean = self.g_avgpool(x).reshape(B,-1) # (B,C,1) -> (B,C)
        xmax = self.g_maxpool(x).reshape(B,-1)  # (B,C,1) -> (B,C)
        
        merged_x = self.mlp(xmean) + self.mlp(xmax) # (B,C) -> (B,C)
        merged_x = F.sigmoid(merged_x)
        merged_x = merged_x.unsqueeze(-1) # (B,C,1,1)
        return  merged_x

class SpAttnBlock(nn.Module):
    def __init__(self, kernel_size=3) -> None:
        super().__init__()
        self.ch_maxpool = ChannelWiseMaxPool()
        self.ch_avgpool = ChannelWiseAvgPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=1)
    
    def forward(self, x):
        xchmaxpool = self.ch_maxpool(x) # (B,1,H,W)
        xchavgpool = self.ch_avgpool(x) # (B,1,H,W)
        merged_x = torch.concat([xchavgpool, xchmaxpool], dim=1) # (B,2,H,W)
        merged_x = self.conv(merged_x) # (B,1,H,W)
        return F.sigmoid(merged_x)

class ResAttnBlock(nn.Module):
    def __init__(self, in_ch:int, hidden_ch:int=256, reduction_ratio=0.5):
        super().__init__()
        self.in_ch = in_ch
        self.ch_attn = ChAttnBlock(in_ch, hidden_ch, reduction_ratio)
        self.sp_attn = SpAttnBlock()

    def forward(self, x):
        y1 = self.ch_attn(x)
        y1 = y1.unsqueeze(-1)
        y1 = x*y1
        y2 = self.sp_attn(y1)
        y2 = y1*y2
        return x + y2

class ClimateDownscale(nn.Module):
    def __init__(self,
                 in_ch:int,
                 aux_ch:int,
                 hidden_chs:list,
                 out_ch:int,
                 scale_factor:int,
                 n_main_layer:int,
                 upsample:str='bilinear',
                 extract_every_n_layer=2,
                 rab_hidden_ch = 64,
                 concat_hidden_ch = 64,
                 device='cpu') -> None:
        super().__init__()
        assert extract_every_n_layer >= 2, "Not Supported."
        assert n_main_layer >= len(hidden_chs), "Invalid number of hidden channels."
        self.in_ch = in_ch
        self.aux_ch = aux_ch
        self.out_ch = out_ch
        self.scale_factor = scale_factor
        self.upsample = upsample
        self.n_main_layer = n_main_layer
        self.extract_every_n_layer = extract_every_n_layer
        self.extract_idx = self._get_extract_idx(n_main_layer)
        self.rab_hidden_ch = rab_hidden_ch

        self.hidden_chs = None
        self.main_layers = None
        self.RAB_blocks = None
        self.aux_concat = None
        self.upsample_layer = None

        
        self.x0_skip = nn.Upsample(scale_factor=self.scale_factor,
                                   mode=self.upsample)
        
        
        self._generalized_hidden_chs(hidden_chs)
        self._create_main_layers()
        self._create_RAB_blocks()
        self._create_shrinkage_layers()
        self._create_upsample_layer(in_ch=in_ch)
        self._create_aux_concat(concat_hidden_ch, in_ch)

        assert self.n_main_layer // self.extract_every_n_layer == len(self.RAB_blocks)
        assert len(self.RAB_blocks) == len(self.shrinkage_layers)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_ch, self.hidden_chs[0], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.concat_out = nn.Sequential(
            nn.Conv2d(in_ch*2+aux_ch, concat_hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(concat_hidden_ch, out_ch, kernel_size=3, padding=1)
        )
        
    def _get_extract_idx(self, n_main_layer):
        ex_idx = []
        for n in range(n_main_layer):
            if n % self.extract_every_n_layer == self.extract_every_n_layer-2:
                ex_idx.append(n)
        return ex_idx

    def _generalized_hidden_chs(self, hidden_chs):
        #TODO: auto fill the hidden channels.
        if self.hidden_chs is None:
            duplicates_of_ch = self.n_main_layer // len(hidden_chs)
            assert self.n_main_layer % len(hidden_chs) == 0, "Invalid length of hidden channels: can not arrange the hidden dims of the main layers."
            if self.n_main_layer // len(hidden_chs) > 1:
                duplicates_of_ch += 1
            self.hidden_chs = [ch for ch in hidden_chs for _ in range(duplicates_of_ch)]
        return

    def _create_main_layers(self):        
        self.main_layers = nn.ModuleList([
            nn.Conv2d(self.hidden_chs[i], self.hidden_chs[i+1], kernel_size=3, padding=1) \
                for i in range(self.n_main_layer-1)
        ])
        return

    def _create_RAB_blocks(self):
        self.RAB_blocks = nn.ModuleList([
            ResAttnBlock(self.hidden_chs[i+1], self.rab_hidden_ch) for i in self.extract_idx
        ])
        return

    def _create_shrinkage_layers(self):
        self.shrinkage_layers = nn.ModuleList([
            nn.Conv2d(self.hidden_chs[i+1], 1, 3, padding=1) for i in self.extract_idx
        ])
        return

    def _create_aux_concat(self, hidden_ch=64, out_ch=1):
        self.aux_concat =  nn.Sequential(
            nn.Conv2d(len(self.shrinkage_layers)+self.hidden_chs[-1], hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=3, padding=1),
        )
        return

    def _create_upsample_layer(self, in_ch, hidden_ch=1):
        #TODO: convtranspose, subpixel if self.upsample is specified.
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=self.scale_factor,
                        mode=self.upsample),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return

    def forward(self, x, topo):
        assert topo.dim() == 4, "Topology tensor should have the shape of (1,1,H,W)"
        x0_skip = self.x0_skip(x) # upsampled skip connection (B, in_ch, H*scale, W*scale)

        RAB_feature_map = []
        x1 = self.conv0(x) # (B, input_ch, H, W) -> (B, hidden_ch[0], H, W)
        for n, conv in enumerate(self.main_layers):
            # print("Main layer index = ", n)
            x1 = conv(x1)
            x1 = F.relu(x1)
            if n % self.extract_every_n_layer == self.extract_every_n_layer-2:
                # print("extract index = ", n//self.extract_every_n_layer)
                x1 = self.RAB_blocks[n//self.extract_every_n_layer](x1)
                x_skip = self.shrinkage_layers[n//self.extract_every_n_layer](x1)
                RAB_feature_map.append(x_skip)
            
    
        # extracted feature maps fusion
        x_skips_cat = torch.cat(RAB_feature_map, dim=1)
        x1 = torch.cat([x1, x_skips_cat], dim=1) # (B, hidden[-1]+num_skip_cat, H, W)
        x1 = self.aux_concat(x1)
        x1 = x1 + x
        x1 = F.relu(x1)

        # upsampling
        x1 = self.upsample_layer(x1) # (B, in_ch, H*scale, W*scale)
        x1 = torch.concat([x1, x0_skip], dim=1)

        topo_batch = torch.repeat_interleave(topo.cpu(), torch.tensor(x1.shape[0]), dim=0)
        topo_batch = topo_batch.to(x1.device)
        x1 = torch.concat([x1, topo_batch], dim=1)
        x1 = self.concat_out(x1)
        return x1