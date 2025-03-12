import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from .segment_anything import build_sam_vit_h
from .lisa3d_base import ConvBlock3D, UpsamplingBlock3D


def apply_lora(module, r=8, target_modules=None):
    """应用LoRA到指定模块"""
    if target_modules is None:
        # 默认应用到所有线性层
        target_modules = ["query", "value", "q_proj", "v_proj", "out_proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none"
    )
    return get_peft_model(module, config)


class SAMEncoder3D(nn.Module):
    """3D适配的SAM编码器"""
    def __init__(self, pretrained_path=None, with_lora=True):
        super(SAMEncoder3D, self).__init__()
        
        # 加载预训练SAM编码器
        self.sam_encoder = build_sam_vit_h(pretrained_path)
        
        # 冻结原始参数
        for param in self.sam_encoder.parameters():
            param.requires_grad = False
            
        # 修改第一层卷积以支持3D输入
        original_conv = self.sam_encoder.image_encoder.patch_embed.proj
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        
        # 创建3D卷积层，保持空间维度处理方式一致
        self.patch_embed_3d = nn.Conv3d(
            in_channels=1,  # 3D医学图像通常是单通道
            out_channels=out_channels,
            kernel_size=(1, kernel_size[0], kernel_size[1]),
            stride=(1, stride[0], stride[1]),
            padding=(0, padding[0], padding[1])
        )
        
        # 应用LoRA
        if with_lora:
            self.sam_encoder = apply_lora(self.sam_encoder)
            
        # 3D特征处理层
        self.adapter = nn.Sequential(
            ConvBlock3D(2, out_channels, out_channels),
            nn.Conv3d(out_channels, 256, kernel_size=1)
        )
    
    def forward(self, x):
        # x形状: [B, 1, D, H, W]
        B, C, D, H, W = x.shape
        features = []
        
        # 逐切片处理
        for d in range(D):
            # 提取当前切片并转换为RGB（复制通道）
            slice_2d = x[:, :, d, :, :].repeat(1, 3, 1, 1)  # [B, 3, H, W]
            
            # 通过SAM编码器
            with torch.no_grad():
                slice_features = self.sam_encoder.image_encoder(slice_2d)  # [B, 256, H/16, W/16]
            
            features.append(slice_features)
        
        # 堆叠所有切片特征
        features = torch.stack(features, dim=2)  # [B, 256, D, H/16, W/16]
        
        # 通过3D适配层
        features = self.adapter(features)
        
        return features


class SAMDecoder3D(nn.Module):
    """3D适配的SAM解码器"""
    def __init__(self, in_channels=256, out_channels=1, with_lora=True):
        super(SAMDecoder3D, self).__init__()
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            UpsamplingBlock3D(in_channels, 128),
            ConvBlock3D(2, 128, 128),
            UpsamplingBlock3D(128, 64),
            ConvBlock3D(2, 64, 64),
            UpsamplingBlock3D(64, 32),
            ConvBlock3D(1, 32, 32)
        ])
        
        # 输出层
        self.mask_head = nn.Conv3d(32, out_channels, kernel_size=1)
        
        # 应用LoRA
        if with_lora:
            self.up_blocks = apply_lora(self.up_blocks)
    
    def forward(self, x, text_embeddings=None):
        # x形状: [B, 256, D, H/16, W/16]
        
        # 如果有文本嵌入，融合到特征中
        if text_embeddings is not None:
            # 扩展文本嵌入以匹配特征图形状
            text_embeddings = text_embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            text_embeddings = text_embeddings.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
            
            # 特征融合（加法）
            x = x + text_embeddings
        
        # 上采样路径
        for block in self.up_blocks:
            x = block(x)
        
        # 生成掩码
        masks = self.mask_head(x)  # [B, out_channels, D, H, W]
        
        return masks


class MultiModalFusion(nn.Module):
    """多模态特征融合模块"""
    def __init__(self, text_dim=4096, image_dim=256):
        super(MultiModalFusion, self).__init__()
        
        # 文本特征投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Linear(text_dim // 2, image_dim),
            nn.LayerNorm(image_dim)
        )
    
    def forward(self, text_embeddings, image_embeddings=None):
        # 投影文本特征
        projected_text = self.text_proj(text_embeddings)  # [B, 256]
        
        return projected_text
