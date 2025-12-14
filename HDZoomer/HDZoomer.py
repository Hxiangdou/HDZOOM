import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "Input feature size not match"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x

    def calculate_mask(self, H, W, device):
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)


class PatchExpand(nn.Module):
    """ Upsample: Input C -> Output C/2, Resolution x2 """
    def __init__(self, dim, scale_factor=2):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)
        self.scale_factor = scale_factor

    def forward(self, x):
        """ x: (B, H, W, C) """
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.scale_factor)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """ 
    Downsample: Input C_in -> Output C_out, Resolution /2 
    【修改】增加了 output_dim 参数，以便在解码器中灵活控制通道数
    """
    def __init__(self, dim, output_dim=None):
        super().__init__()
        self.dim = dim
        # 如果未指定 output_dim，默认为 2*dim (标准 Swin 行为)
        # 但在解码器 Cat 之后，我们通常希望 output_dim 等于下一层 Skip Connection 的维度
        self.output_dim = output_dim if output_dim is not None else 2 * dim
        
        # 输入维度 dim。拼接 2x2 后变成 4*dim。
        # Linear 将 4*dim 映射到 output_dim。
        self.reduction = nn.Linear(4 * dim, self.output_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """ x: (B, 2H, 2W, C) """
        B, H, W, C = x.shape
        
        if (H % 2 != 0) or (W % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x

# --------------------------------------------------------
# 5. Main Inverted Swin Module (修正版)
# --------------------------------------------------------

class BasicLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, window_size,
                 attn_type='expand', mlp_ratio=4.):
        
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.attn_type = attn_type
        
        # 1. Swin Blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        
        # 2. Transition Layer
        if attn_type == 'expand':
            # 放大：C -> C/2, H*W -> 2H*2W
            self.transition = PatchExpand(dim=dim)
        elif attn_type == 'merge':
            # 缩小：通道 dim -> output_dim (这里 dim 已经是 Cat 后的 2*C_skip)
            # 【关键修改】传递 output_dim 给 PatchMerging
            self.transition = PatchMerging(dim=dim, output_dim=output_dim) 
        else: # Bottleneck
            self.transition = nn.Identity()

        # Mask Calculation
        curr_res = input_resolution
        for blk in self.blocks:
            blk.H, blk.W = curr_res
            blk.calculate_mask(curr_res[0], curr_res[1], device='cpu')

    def forward(self, x):
        # x: (B, L, C) or (B, H, W, C)
        # 确保 mask 在正确的 device
        for blk in self.blocks:
            if blk.attn_mask is not None and blk.attn_mask.device != x.device:
                blk.attn_mask = blk.attn_mask.to(x.device)
            x = blk(x)
        
        # Swin Blocks 输出是 Flat 的 (B, L, C)，需要 Reshape 进行 Transition
        B, L, C = x.shape
        H, W = self.input_resolution
        x_reshaped = x.view(B, H, W, C)
        
        # Transition
        x_trans = self.transition(x_reshaped)
        
        # 返回：
        # 1. x_trans: 用于传递给下一层的特征（尺寸已变）
        # 2. x_reshaped: 经过了 Swin 处理但未变尺寸的特征（对于 Expand 层，这里不需要它做 Skip，需要的是 Trans 后的）
        #    但是在 Bottleneck 或其他结构中可能需要。
        #    为了修复 Skip Connection 逻辑，我们在主循环里处理。
        return x_trans, x_reshaped

class InvertedSwinModule(nn.Module):
    def __init__(self, in_dim, input_resolution, depths, num_heads, window_size):
        super().__init__()
        
        self.num_layers = len(depths) - 1 # 减去瓶颈层
        self.input_resolution = input_resolution
        
        # 计算通道变化：
        # Input: C (96)
        # Expand 1: C -> C/2 (48)
        # Expand 2: C/2 -> C/4 (24)
        # Expand 3: C/4 -> C/8 (12) -> Bottleneck -> 12
        # Contract 1: Cat(12+12)=24 -> Merging -> C/4 (24)
        # Contract 2: Cat(24+24)=48 -> Merging -> C/2 (48)
        # Contract 3: Cat(48+48)=96 -> Merging -> C (96)
        
        current_dim = in_dim
        current_res = input_resolution
        
        # --- 编码器 (Expansion Path) ---
        self.expand_path = nn.ModuleList()
        num_expands = 3 # 假设 depths 长度为 4，前3个是 expand
        
        for i in range(num_expands):
            layer = BasicLayer(
                dim=current_dim,
                output_dim=current_dim // 2,
                input_resolution=current_res,
                depth=depths[i],
                num_heads=num_heads,
                window_size=window_size,
                attn_type='expand'
            )
            self.expand_path.append(layer)
            current_dim //= 2
            current_res = (current_res[0] * 2, current_res[1] * 2)

        # --- 瓶颈层 (Bottleneck) ---
        # 此时 current_dim = C/8 (12), current_res = 8H*8W
        self.bottleneck = BasicLayer(
            dim=current_dim,
            output_dim=current_dim,
            input_resolution=current_res,
            depth=depths[-1], # 最后一个 depth
            num_heads=num_heads,
            window_size=window_size,
            attn_type='bottleneck'
        )
        
        # --- 解码器 (Contraction Path) ---
        self.contract_path = nn.ModuleList()
        # 目标输出维度序列: 24, 48, 96
        
        for i in range(num_expands):
            # i=0: Input Cat(12+12)=24. Target Out=24 (匹配 Enc1 的 Skip: 24)
            # i=1: Input Cat(24+24)=48. Target Out=48 (匹配 Enc0 的 Skip: 48)
            # i=2: Input Cat(48+48)=96. Target Out=96 (匹配 Input)
            
            target_out_dim = current_dim * 2 
            input_dim_after_cat = current_dim * 2 # Skip Connection 也是 current_dim
            
            layer = BasicLayer(
                dim=input_dim_after_cat, 
                output_dim=target_out_dim, 
                input_resolution=current_res,
                depth=depths[num_expands - 1 - i], # 倒序使用 depth
                num_heads=num_heads,
                window_size=window_size,
                attn_type='merge'
            )
            self.contract_path.append(layer)
            
            current_dim = target_out_dim
            current_res = (current_res[0] // 2, current_res[1] // 2)

        self.final_proj = nn.LayerNorm(current_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x_flat = x.view(B, H * W, C)
        
        skip_features = []

        # 1. 放大路径 (Expansion Path)
        for layer in self.expand_path:
            x_trans, x_prev = layer(x_flat)
            
            skip_features.append(x_trans) 
            
            x_flat = x_trans.view(B, -1, x_trans.shape[-1]) 

        # 2. 瓶颈层 (Bottleneck)
        x_flat, _ = self.bottleneck(x_flat) 
        x_flat = x_flat.view(B, -1, x_flat.shape[-1])

        # 3. 缩小路径 (Contraction Path)
        for i, layer in enumerate(self.contract_path):
            B, L, C_curr = x_flat.shape
            H_curr = int(L**0.5)
            
            x_img = x_flat.view(B, H_curr, H_curr, C_curr)
            
            skip = skip_features.pop() # (B, H, W, C)
            
            x_cat = torch.cat([x_img, skip], dim=-1) # (B, H, W, C+C)
            
            x_cat_flat = x_cat.view(B, -1, x_cat.shape[-1])
            
            x_flat, _ = layer(x_cat_flat) 
            x_flat = x_flat.view(B, -1, x_flat.shape[-1])

        x_out = self.final_proj(x_flat)
        return x_out.view(B, H, W, C)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB,参数大小：{:.3f}MB,所有buffer的参数字节大小：{:.3f}MB'.format(all_size, param_size/ 1024 / 1024, buffer_size/ 1024 / 1024))
    return 


if __name__ == "__main__":
    H, W = 64, 64
    C = 96
    window_size = 8
    
    # 3个放大 + 1个瓶颈
    depths = [2, 2, 2, 2] 

    x = torch.randn(2, H, W, C)
    
    print(f"1. Input Shape: {x.shape}")
    
    model = InvertedSwinModule(
        in_dim=C,
        input_resolution=(H, W),
        depths=depths,
        num_heads=4,
        window_size=window_size
    )
    
    out = model(x)
    getModelSize(model)
    print(f"2. Output Shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("\n✅ Success! The Inverted Swin U-Net is running correctly.")