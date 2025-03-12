from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model

from .segment_anything import build_sam_vit_h


# 基础损失函数
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


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


# 3D卷积基础模块
class ConvBlock3D(nn.Module):
    """3D卷积块，参考SKCDF的设计"""
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='batchnorm'):
        super(ConvBlock3D, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=8, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            ops.append(nn.GELU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingBlock3D(nn.Module):
    """3D下采样块"""
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='batchnorm'):
        super(DownsamplingBlock3D, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=8, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.GELU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingBlock3D(nn.Module):
    """3D上采样块"""
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='batchnorm'):
        super(UpsamplingBlock3D, self).__init__()

        ops = []
        ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=8, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        ops.append(nn.GELU())

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# SAM编码器和解码器
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


# LISA3D元模型
class Lisa3DMetaModel(nn.Module):
    """LISA3D元模型，封装共享组件"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DMetaModel, self).__init__()

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        
        # 初始化共享组件
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # 多模态语言模型
        self.mllm_model = LlavaForConditionalGeneration.from_pretrained(
            config.mllm_path,
            torch_dtype=torch.float16,
        )
        
        # 应用LoRA到MLLM
        if config.use_lora:
            self.mllm_model = apply_lora(
                self.mllm_model,
                r=config.lora_r,
                target_modules=["q_proj", "v_proj"]
            )
        
        # 冻结MLLM参数
        for param in self.mllm_model.parameters():
            param.requires_grad = False
        
        # 特征融合模块
        self.fusion_module = MultiModalFusion(
            text_dim=config.hidden_size,
            image_dim=config.out_dim
        )
        
        # 设置融合模块为可训练
        self.fusion_module.train()
        for param in self.fusion_module.parameters():
            param.requires_grad = True
        
        # 定义特殊标记
        self.seg_token_idx = config.seg_token_idx


# LISA3D模型
class Lisa3DModel(Lisa3DMetaModel):
    """双编码器/解码器结构的LISA3D模型"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DModel, self).__init__(config, **kwargs)

        # 3D特定配置
        self.config.use_cache = False
        self.config.image_aspect_ratio = "3D"
        
        # 初始化双编码器
        self.encoder_l = SAMEncoder3D(
            pretrained_path=self.vision_pretrained,
            with_lora=config.use_lora
        )
        
        self.encoder_u = SAMEncoder3D(
            pretrained_path=self.vision_pretrained,
            with_lora=config.use_lora
        )
        
        # 初始化双解码器
        self.decoder_l = SAMDecoder3D(
            in_channels=config.out_dim,
            out_channels=config.num_cls,
            with_lora=config.use_lora
        )
        
        self.decoder_u = SAMDecoder3D(
            in_channels=config.out_dim,
            out_channels=config.num_cls,
            with_lora=config.use_lora
        )
        
        # 初始化模型权重
        self._init_weights()
    
    def _init_weights(self):
        # 初始化模型权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_seg_token_features(self, text_tokens, text_outputs):
        """提取[seg]标记的特征"""
        # 找到[seg]标记的位置
        seg_token_mask = (text_tokens == self.seg_token_idx).bool()
        
        # 如果没有找到[seg]标记，返回None
        if not seg_token_mask.any():
            return None
        
        # 提取最后一层隐藏状态
        last_hidden_state = text_outputs.last_hidden_state  # [B, L, D]
        
        # 提取[seg]标记的特征
        seg_features = []
        for b in range(seg_token_mask.size(0)):
            # 找到当前批次中的[seg]标记位置
            seg_indices = seg_token_mask[b].nonzero(as_tuple=True)[0]
            
            if len(seg_indices) > 0:
                # 提取所有[seg]标记的特征并平均
                batch_seg_features = last_hidden_state[b, seg_indices]
                seg_features.append(batch_seg_features.mean(0))
            else:
                # 如果没有[seg]标记，使用零向量
                seg_features.append(torch.zeros_like(last_hidden_state[b, 0]))
        
        # 堆叠所有批次的特征
        seg_features = torch.stack(seg_features, dim=0)  # [B, D]
        
        return seg_features
    
    def process_text(self, text_tokens):
        """处理文本输入并提取特征"""
        # 通过MLLM处理文本
        with torch.no_grad():
            text_outputs = self.mllm_model(
                input_ids=text_tokens,
                return_dict=True
            )
        
        # 提取[seg]标记的特征
        seg_features = self.extract_seg_token_features(text_tokens, text_outputs)
        
        # 如果没有找到[seg]标记，返回None
        if seg_features is None:
            return None
        
        # 投影文本特征
        projected_text = self.fusion_module(seg_features)
        
        return projected_text
    
    def forward_labeled(self, image, text_tokens):
        """标记数据流处理"""
        # 处理文本输入
        text_features = self.process_text(text_tokens)
        
        # 通过编码器处理图像
        image_features = self.encoder_l(image)
        
        # 通过解码器生成掩码
        masks = self.decoder_l(image_features, text_features)
        
        return masks
    
    def forward_unlabeled(self, image, text_tokens):
        """非标记数据流处理"""
        # 处理文本输入
        text_features = self.process_text(text_tokens)
        
        # 生成伪标签（使用编码器L和解码器L）
        with torch.no_grad():
            image_features_l = self.encoder_l(image)
            pseudo_masks = self.decoder_l(image_features_l, text_features)
        
        # 生成预测（使用编码器U和解码器U）
        image_features_u = self.encoder_u(image)
        pred_masks = self.decoder_u(image_features_u, text_features)
        
        return pseudo_masks, pred_masks
    
    def forward(self, image, text_tokens, is_labeled=True):
        """前向传播"""
        if is_labeled:
            return self.forward_labeled(image, text_tokens)
        else:
            return self.forward_unlabeled(image, text_tokens)


# LISA3D分割模型
class LISA3DForSegmentation(nn.Module):
    """LISA3D分割模型"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LISA3DForSegmentation, self).__init__()
        
        # 初始化LISA3D模型
        self.lisa3d = Lisa3DModel(config, **kwargs)
        
        # 配置参数
        self.config = config
        self.loss_weights = {
            'dice': config.dice_weight,
            'ce': config.ce_weight,
            'consistency': config.consistency_weight
        }
    
    def compute_loss(self, masks, targets, num_masks):
        """Compute the loss for supervised learning"""
        # 计算Dice损失
        dice = dice_loss(masks, targets, num_masks)
        
        # 计算交叉熵损失
        ce = sigmoid_ce_loss(masks, targets, num_masks)
        
        # 组合损失
        loss = self.loss_weights['dice'] * dice + self.loss_weights['ce'] * ce
        
        return loss, {'dice': dice, 'ce': ce}
    
    def compute_consistency_loss(self, pseudo_masks, pred_masks, num_masks):
        """Compute the consistency loss for semi-supervised learning"""
        # 将伪标签转换为确定性标签
        with torch.no_grad():
            pseudo_labels = (pseudo_masks > 0).float()
        
        # 计算一致性损失
        consistency = dice_loss(pred_masks, pseudo_labels, num_masks)
        
        return self.loss_weights['consistency'] * consistency, {'consistency': consistency}
    
    def forward(
        self,
        labeled_images=None,
        labeled_masks=None,
        labeled_text_tokens=None,
        unlabeled_images=None,
        unlabeled_text_tokens=None,
        return_loss=True,
    ):
        """LISA3D模型前向传播"""
        losses = {}
        outputs = {}
        
        # 处理标记数据
        if labeled_images is not None and labeled_text_tokens is not None:
            # 前向传播得到预测掩码
            pred_masks = self.lisa3d(
                image=labeled_images,
                text_tokens=labeled_text_tokens,
                is_labeled=True
            )
            outputs['pred_masks'] = pred_masks
            
            # 计算监督学习损失
            if return_loss and labeled_masks is not None:
                num_masks = labeled_masks.shape[0] * labeled_masks.shape[1]  # B * C
                loss, loss_items = self.compute_loss(pred_masks, labeled_masks, num_masks)
                losses['supervised_loss'] = loss
                losses.update(loss_items)
        
        # 处理非标记数据
        if unlabeled_images is not None and unlabeled_text_tokens is not None:
            # 前向传播得到伪标签和预测掩码
            pseudo_masks, pred_masks_u = self.lisa3d(
                image=unlabeled_images,
                text_tokens=unlabeled_text_tokens,
                is_labeled=False
            )
            outputs['pseudo_masks'] = pseudo_masks
            outputs['pred_masks_u'] = pred_masks_u
            
            # 计算半监督学习损失
            if return_loss:
                num_masks = unlabeled_images.shape[0] * self.config.num_cls  # B * C
                consistency_loss, consistency_items = self.compute_consistency_loss(
                    pseudo_masks, pred_masks_u, num_masks
                )
                losses['consistency_loss'] = consistency_loss
                losses.update(consistency_items)
        
        # 计算总损失
        if return_loss and losses:
            total_loss = sum(loss for loss in losses.values() if torch.is_tensor(loss))
            losses['total_loss'] = total_loss
        
        return losses, outputs
    
    def predict(self, image, text_tokens):
        """LISA3D模型推理"""
        # 前向传播得到预测掩码
        with torch.no_grad():
            pred_masks = self.lisa3d(
                image=image,
                text_tokens=text_tokens,
                is_labeled=True
            )
        
        # 应用sigmoid函数并转换为二值掩码
        binary_masks = (pred_masks.sigmoid() > 0.5).float()
        
        return binary_masks, pred_masks
