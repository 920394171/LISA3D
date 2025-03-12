from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from .segment_anything import build_sam_vit_h


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


class Lisa3DMetaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DMetaModel, self).__init__()

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Lisa3DModel(Lisa3DMetaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DModel, self).__init__(config, **kwargs)

        # 3D specific configurations
        self.config.use_cache = False
        self.config.image_aspect_ratio = "3D"
        
        # Encoder for 3D data
        self.encoder = nn.Sequential(
            nn.Conv3d(config.n_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Decoder for labeled data
        self.decoder_l = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, config.num_cls, kernel_size=1),
        )
        
        # Decoder for unlabeled data
        self.decoder_u = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, config.num_cls, kernel_size=1),
        )
        
        # Cross-attention mechanism for feature fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True
        )
        
        # Projection layers for attention
        self.q_proj = nn.Linear(128, 128)
        self.k_proj = nn.Linear(128, 128)
        self.v_proj = nn.Linear(128, 128)


class LISA3DForSegmentation(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LISA3DForSegmentation, self).__init__()
        
        # Loss weights
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.0)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", 1.0)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 1.0)
        
        # Initialize model
        self.model = Lisa3DModel(config, **kwargs)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights for the model
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, **kwargs):
        return self.model_forward(**kwargs)
    
    def model_forward(
        self,
        images_labeled: torch.FloatTensor,
        images_unlabeled: torch.FloatTensor,
        masks_list: List[torch.FloatTensor] = None,
        label_list: List[torch.Tensor] = None,
        inference: bool = False,
        **kwargs,
    ):
        # Process labeled images
        features_labeled = self.model.encoder(images_labeled)
        
        # Process unlabeled images
        features_unlabeled = self.model.encoder(images_unlabeled)
        
        # Apply cross-attention between labeled and unlabeled features
        batch_size, channels, d, h, w = features_labeled.shape
        features_labeled_flat = features_labeled.view(batch_size, channels, -1).permute(0, 2, 1)  # B, D*H*W, C
        features_unlabeled_flat = features_unlabeled.view(batch_size, channels, -1).permute(0, 2, 1)  # B, D*H*W, C
        
        # Compute attention: labeled features attend to unlabeled features
        q = self.model.q_proj(features_labeled_flat)
        k = self.model.k_proj(features_unlabeled_flat)
        v = self.model.v_proj(features_unlabeled_flat)
        
        attn_output, _ = self.model.cross_attention(q, k, v)
        
        # Reshape back to 3D
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, channels, d, h, w)
        
        # Fuse features
        features_labeled_enhanced = features_labeled + attn_output
        
        # Generate outputs
        # 1. Label decoder for labeled data
        output_labeled = self.model.decoder_l(features_labeled_enhanced)
        
        # 2. Label decoder for unlabeled data
        output_unlabeled_from_labeled = self.model.decoder_l(features_unlabeled)
        
        # 3. Unlabeled decoder for unlabeled data
        output_unlabeled = self.model.decoder_u(features_unlabeled)
        
        if inference:
            return {
                "output_labeled": output_labeled,
                "output_unlabeled_from_labeled": output_unlabeled_from_labeled,
                "output_unlabeled": output_unlabeled,
            }
        
        # Calculate losses if not in inference mode
        # Loss for labeled data
        loss_labeled = 0
        for batch_idx in range(len(masks_list)):
            gt_mask = masks_list[batch_idx]
            pred_mask = output_labeled[batch_idx].unsqueeze(0)
            
            loss_labeled += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * self.bce_loss_weight
            loss_labeled += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * self.dice_loss_weight
        
        # Generate pseudo-labels for unlabeled data using label decoder
        pseudo_labels = torch.argmax(output_unlabeled_from_labeled, dim=1, keepdim=True).float()
        
        # Loss for unlabeled data (consistency loss)
        loss_unlabeled = 0
        for batch_idx in range(batch_size):
            pseudo_mask = pseudo_labels[batch_idx].unsqueeze(0)
            pred_mask = output_unlabeled[batch_idx].unsqueeze(0)
            
            loss_unlabeled += sigmoid_ce_loss(pred_mask, pseudo_mask, num_masks=1) * self.bce_loss_weight
            loss_unlabeled += dice_loss(pred_mask, pseudo_mask, num_masks=1) * self.dice_loss_weight
        
        # Total loss
        total_loss = loss_labeled + loss_unlabeled * self.ce_loss_weight
        
        return {
            "loss": total_loss,
            "loss_labeled": loss_labeled,
            "loss_unlabeled": loss_unlabeled,
            "output_labeled": output_labeled,
            "output_unlabeled_from_labeled": output_unlabeled_from_labeled,
            "output_unlabeled": output_unlabeled,
        }
    
    def evaluate(
        self,
        images_labeled,
        images_unlabeled,
        **kwargs,
    ):
        with torch.no_grad():
            outputs = self.model_forward(
                images_labeled=images_labeled,
                images_unlabeled=images_unlabeled,
                inference=True,
                **kwargs
            )
            
            return outputs
