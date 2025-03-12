import torch
import torch.nn as nn
import torch.nn.functional as F

from .lisa3d_model import Lisa3DModel
from .lisa3d_base import dice_loss, sigmoid_ce_loss


class LISA3DForSegmentation(nn.Module):
    """LISA3Du5206u5272u6a21u578b"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LISA3DForSegmentation, self).__init__()
        
        # u521du59cbu5316LISA3Du6a21u578b
        self.lisa3d = Lisa3DModel(config, **kwargs)
        
        # u914du7f6eu53c2u6570
        self.config = config
        self.loss_weights = {
            'dice': config.dice_weight,
            'ce': config.ce_weight,
            'consistency': config.consistency_weight
        }
    
    def compute_loss(self, masks, targets, num_masks):
        """Compute the loss for supervised learning"""
        # u8ba1u7b97Diceu635fu5931
        dice = dice_loss(masks, targets, num_masks)
        
        # u8ba1u7b97u4ea4u53c9u71b5u635fu5931
        ce = sigmoid_ce_loss(masks, targets, num_masks)
        
        # u7ec4u5408u635fu5931
        loss = self.loss_weights['dice'] * dice + self.loss_weights['ce'] * ce
        
        return loss, {'dice': dice, 'ce': ce}
    
    def compute_consistency_loss(self, pseudo_masks, pred_masks, num_masks):
        """Compute the consistency loss for semi-supervised learning"""
        # u5c06u4f2au6807u7b7eu8f6cu6362u4e3au786eu5b9au6027u6807u7b7e
        with torch.no_grad():
            pseudo_labels = (pseudo_masks > 0).float()
        
        # u8ba1u7b97u4e00u81f4u6027u635fu5931
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
        """LISA3Du6a21u578bu524du5411u4f20u64ad"""
        losses = {}
        outputs = {}
        
        # u5904u7406u6807u8bb0u6570u636e
        if labeled_images is not None and labeled_text_tokens is not None:
            # u524du5411u4f20u64adu5f97u5230u9884u6d4bu63a9u7801
            pred_masks = self.lisa3d(
                image=labeled_images,
                text_tokens=labeled_text_tokens,
                is_labeled=True
            )
            outputs['pred_masks'] = pred_masks
            
            # u8ba1u7b97u76d1u7763u5b66u4e60u635fu5931
            if return_loss and labeled_masks is not None:
                num_masks = labeled_masks.shape[0] * labeled_masks.shape[1]  # B * C
                loss, loss_items = self.compute_loss(pred_masks, labeled_masks, num_masks)
                losses['supervised_loss'] = loss
                losses.update(loss_items)
        
        # u5904u7406u975eu6807u8bb0u6570u636e
        if unlabeled_images is not None and unlabeled_text_tokens is not None:
            # u524du5411u4f20u64adu5f97u5230u4f2au6807u7b7eu548cu9884u6d4bu63a9u7801
            pseudo_masks, pred_masks_u = self.lisa3d(
                image=unlabeled_images,
                text_tokens=unlabeled_text_tokens,
                is_labeled=False
            )
            outputs['pseudo_masks'] = pseudo_masks
            outputs['pred_masks_u'] = pred_masks_u
            
            # u8ba1u7b97u534au76d1u7763u5b66u4e60u635fu5931
            if return_loss:
                num_masks = unlabeled_images.shape[0] * self.config.num_cls  # B * C
                consistency_loss, consistency_items = self.compute_consistency_loss(
                    pseudo_masks, pred_masks_u, num_masks
                )
                losses['consistency_loss'] = consistency_loss
                losses.update(consistency_items)
        
        # u8ba1u7b97u603bu635fu5931
        if return_loss and losses:
            total_loss = sum(loss for loss in losses.values() if torch.is_tensor(loss))
            losses['total_loss'] = total_loss
        
        return losses, outputs
    
    def predict(self, image, text_tokens):
        """LISA3Du6a21u578bu63a8u7406"""
        # u524du5411u4f20u64adu5f97u5230u9884u6d4bu63a9u7801
        with torch.no_grad():
            pred_masks = self.lisa3d(
                image=image,
                text_tokens=text_tokens,
                is_labeled=True
            )
        
        # u5e94u7528sigmoidu51fdu6570u5e76u8f6cu6362u4e3au4e8cu503cu63a9u7801
        binary_masks = (pred_masks.sigmoid() > 0.5).float()
        
        return binary_masks, pred_masks
