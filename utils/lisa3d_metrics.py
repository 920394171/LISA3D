import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from scipy.ndimage import distance_transform_edt
from skimage.metrics import hausdorff_distance


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    u8ba1u7b97Diceu76f8u4f3cu5ea6u7cfbu6570
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        smooth: u5e73u6ed1u9879uff0cu907fu514du9664u96f6
    
    Returns:
        u6bcfu4e2au6837u672cu7684Diceu7cfbu6570uff0cu5f62u72b6u4e3a [B*C]
    """
    # u5c55u5e73u7a7au95f4u7ef4u5ea6
    pred_flat = pred.flatten(2)  # [B, C, D*H*W]
    target_flat = target.flatten(2)  # [B, C, D*H*W]
    
    # u8ba1u7b97u4ea4u96c6
    intersection = (pred_flat * target_flat).sum(-1)  # [B, C]
    
    # u8ba1u7b97u5e76u96c6
    union = pred_flat.sum(-1) + target_flat.sum(-1)  # [B, C]
    
    # u8ba1u7b97Diceu7cfbu6570
    dice = (2.0 * intersection + smooth) / (union + smooth)  # [B, C]
    
    # u5c55u5e73u6279u6b21u548cu901au9053u7ef4u5ea6
    dice = dice.flatten()  # [B*C]
    
    return dice


def compute_iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    u8ba1u7b97IoUuff08u4ea4u5e76u6bd4uff09u5206u6570
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        smooth: u5e73u6ed1u9879uff0cu907fu514du9664u96f6
    
    Returns:
        u6bcfu4e2au6837u672cu7684IoUu5206u6570uff0cu5f62u72b6u4e3a [B*C]
    """
    # u5c55u5e73u7a7au95f4u7ef4u5ea6
    pred_flat = pred.flatten(2)  # [B, C, D*H*W]
    target_flat = target.flatten(2)  # [B, C, D*H*W]
    
    # u8ba1u7b97u4ea4u96c6
    intersection = (pred_flat * target_flat).sum(-1)  # [B, C]
    
    # u8ba1u7b97u5e76u96c6
    union = pred_flat.sum(-1) + target_flat.sum(-1) - intersection  # [B, C]
    
    # u8ba1u7b97IoUu5206u6570
    iou = (intersection + smooth) / (union + smooth)  # [B, C]
    
    # u5c55u5e73u6279u6b21u548cu901au9053u7ef4u5ea6
    iou = iou.flatten()  # [B*C]
    
    return iou


def compute_hausdorff_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    percentile: float = 95.0,
) -> List[float]:
    """
    u8ba1u7b97Hausdorffu8dddu79bb
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        percentile: u767eu5206u4f4du6570uff0cu7528u4e8eu8ba1u7b97u767eu5206u4f4dHausdorffu8dddu79bb
    
    Returns:
        u6bcfu4e2au6837u672cu7684Hausdorffu8dddu79bbu5217u8868
    """
    # u8f6cu6362u4e3aNumPyu6570u7ec4
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size, num_classes = pred_np.shape[:2]
    hausdorff_distances = []
    
    # u9010u6837u672cu8ba1u7b97
    for b in range(batch_size):
        for c in range(num_classes):
            # u63d0u53d6u5f53u524du6837u672cu548cu7c7bu522b
            pred_mask = pred_np[b, c].astype(bool)
            target_mask = target_np[b, c].astype(bool)
            
            # u68c0u67e5u662fu5426u6709u5206u5272u7ed3u679c
            if not np.any(pred_mask) and not np.any(target_mask):
                # u4e24u8005u90fdu4e3au7a7auff0cu5b8cu5168u5339u914d
                hausdorff_distances.append(0.0)
                continue
            elif not np.any(pred_mask) or not np.any(target_mask):
                # u5176u4e2du4e00u4e2au4e3au7a7auff0cu6700u5927u8dddu79bb
                hausdorff_distances.append(float('inf'))
                continue
            
            try:
                # u8ba1u7b97Hausdorffu8dddu79bb
                hd = hausdorff_distance(pred_mask, target_mask)
                hausdorff_distances.append(hd)
            except Exception as e:
                # u5982u679cu8ba1u7b97u51fau9519uff0cu8bb0u5f55u65e0u7a77u5927
                print(f"Error computing Hausdorff distance: {e}")
                hausdorff_distances.append(float('inf'))
    
    return hausdorff_distances


def compute_surface_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> List[float]:
    """
    u8ba1u7b97u5e73u5747u8868u9762u8dddu79bb
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
    
    Returns:
        u6bcfu4e2au6837u672cu7684u5e73u5747u8868u9762u8dddu79bbu5217u8868
    """
    # u8f6cu6362u4e3aNumPyu6570u7ec4
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    batch_size, num_classes = pred_np.shape[:2]
    surface_distances = []
    
    # u9010u6837u672cu8ba1u7b97
    for b in range(batch_size):
        for c in range(num_classes):
            # u63d0u53d6u5f53u524du6837u672cu548cu7c7bu522b
            pred_mask = pred_np[b, c].astype(bool)
            target_mask = target_np[b, c].astype(bool)
            
            # u68c0u67e5u662fu5426u6709u5206u5272u7ed3u679c
            if not np.any(pred_mask) and not np.any(target_mask):
                # u4e24u8005u90fdu4e3au7a7auff0cu5b8cu5168u5339u914d
                surface_distances.append(0.0)
                continue
            elif not np.any(pred_mask) or not np.any(target_mask):
                # u5176u4e2du4e00u4e2au4e3au7a7auff0cu6700u5927u8dddu79bb
                surface_distances.append(float('inf'))
                continue
            
            # u8ba1u7b97u8dddu79bbu53d8u6362
            pred_dist = distance_transform_edt(~pred_mask)
            target_dist = distance_transform_edt(~target_mask)
            
            # u8ba1u7b97u8868u9762u8dddu79bb
            pred_surface = pred_dist[target_mask]
            target_surface = target_dist[pred_mask]
            
            # u8ba1u7b97u5e73u5747u8868u9762u8dddu79bb
            avg_surface_distance = (np.mean(pred_surface) + np.mean(target_surface)) / 2
            surface_distances.append(avg_surface_distance)
    
    return surface_distances


def compute_precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    u8ba1u7b97u7cbeu786eu7387u548cu53ecu56deu7387
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        smooth: u5e73u6ed1u9879uff0cu907fu514du9664u96f6
    
    Returns:
        u7cbeu786eu7387u548cu53ecu56deu7387u7684u5143u7ec4uff0cu5f62u72b6u5747u4e3a [B*C]
    """
    # u5c55u5e73u7a7au95f4u7ef4u5ea6
    pred_flat = pred.flatten(2)  # [B, C, D*H*W]
    target_flat = target.flatten(2)  # [B, C, D*H*W]
    
    # u8ba1u7b97u771fu9633u6027uff08TPuff09
    true_positives = (pred_flat * target_flat).sum(-1)  # [B, C]
    
    # u8ba1u7b97u6240u6709u9884u6d4bu4e3au9633u6027u7684u50cfu7d20
    predicted_positives = pred_flat.sum(-1)  # [B, C]
    
    # u8ba1u7b97u6240u6709u5b9eu9645u4e3au9633u6027u7684u50cfu7d20
    actual_positives = target_flat.sum(-1)  # [B, C]
    
    # u8ba1u7b97u7cbeu786eu7387
    precision = (true_positives + smooth) / (predicted_positives + smooth)  # [B, C]
    
    # u8ba1u7b97u53ecu56deu7387
    recall = (true_positives + smooth) / (actual_positives + smooth)  # [B, C]
    
    # u5c55u5e73u6279u6b21u548cu901au9053u7ef4u5ea6
    precision = precision.flatten()  # [B*C]
    recall = recall.flatten()  # [B*C]
    
    return precision, recall


def compute_f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    u8ba1u7b97F1u5206u6570
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        smooth: u5e73u6ed1u9879uff0cu907fu514du9664u96f6
    
    Returns:
        u6bcfu4e2au6837u672cu7684F1u5206u6570uff0cu5f62u72b6u4e3a [B*C]
    """
    # u8ba1u7b97u7cbeu786eu7387u548cu53ecu56deu7387
    precision, recall = compute_precision_recall(pred, target, smooth)
    
    # u8ba1u7b97F1u5206u6570
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return f1


def compute_volume_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    u8ba1u7b97u4f53u79efu76f8u4f3cu5ea6
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        smooth: u5e73u6ed1u9879uff0cu907fu514du9664u96f6
    
    Returns:
        u6bcfu4e2au6837u672cu7684u4f53u79efu76f8u4f3cu5ea6uff0cu5f62u72b6u4e3a [B*C]
    """
    # u5c55u5e73u7a7au95f4u7ef4u5ea6
    pred_flat = pred.flatten(2)  # [B, C, D*H*W]
    target_flat = target.flatten(2)  # [B, C, D*H*W]
    
    # u8ba1u7b97u9884u6d4bu548cu76eeu6807u7684u4f53u79ef
    pred_volume = pred_flat.sum(-1)  # [B, C]
    target_volume = target_flat.sum(-1)  # [B, C]
    
    # u8ba1u7b97u4f53u79efu76f8u4f3cu5ea6
    vs = 1.0 - torch.abs(pred_volume - target_volume) / (pred_volume + target_volume + smooth)
    
    # u5c55u5e73u6279u6b21u548cu901au9053u7ef4u5ea6
    vs = vs.flatten()  # [B*C]
    
    return vs


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Union[torch.Tensor, List[float]]]:
    """
    u8ba1u7b97u6240u6709u5206u5272u8bc4u4f30u6307u6807
    
    Args:
        pred: u9884u6d4bu7684u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
        target: u76eeu6807u5206u5272u7ed3u679cuff0cu5f62u72b6u4e3a [B, C, D, H, W]
    
    Returns:
        u5305u542bu6240u6709u8bc4u4f30u6307u6807u7684u5b57u5178
    """
    metrics = {}
    
    # u8ba1u7b97Diceu7cfbu6570
    metrics["dice"] = compute_dice_score(pred, target)
    
    # u8ba1u7b97IoUu5206u6570
    metrics["iou"] = compute_iou_score(pred, target)
    
    # u8ba1u7b97u7cbeu786eu7387u548cu53ecu56deu7387
    precision, recall = compute_precision_recall(pred, target)
    metrics["precision"] = precision
    metrics["recall"] = recall
    
    # u8ba1u7b97F1u5206u6570
    metrics["f1"] = compute_f1_score(pred, target)
    
    # u8ba1u7b97u4f53u79efu76f8u4f3cu5ea6
    metrics["volume_similarity"] = compute_volume_similarity(pred, target)
    
    # u8ba1u7b97Hausdorffu8dddu79bb
    metrics["hausdorff"] = compute_hausdorff_distance(pred, target)
    
    # u8ba1u7b97u8868u9762u8dddu79bb
    metrics["surface_distance"] = compute_surface_distance(pred, target)
    
    return metrics
