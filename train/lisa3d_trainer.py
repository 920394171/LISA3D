import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import logging
import datetime

# 导入自定义模块
from ..model.LISA3D_complete import LISA3DForSegmentation
from ..config.lisa3d_config import LISA3DConfig, get_default_config, save_config_to_file
from ..data.lisa3d_dataset import build_dataloader
from ..utils.lisa3d_metrics import compute_dice_score, compute_hausdorff_distance


class LISA3DTrainer:
    """LISA3D模型训练器"""
    def __init__(
        self,
        config: LISA3DConfig,
        local_rank: int = -1,
        distributed: bool = False,
    ):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            local_rank: 本地设备排名（用于分布式训练）
            distributed: 是否使用分布式训练
        """
        self.config = config
        self.local_rank = local_rank
        self.distributed = distributed
        
        # 设置设备
        self.device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建优化器和学习率调度器
        self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()
        
        # 创建混合精度训练的缩放器
        self.scaler = GradScaler() if config.use_amp else None
        
        # 创建日志记录器
        self.writer = self._create_logger()
        
        # 初始化训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        # 创建模型实例
        model = LISA3DForSegmentation(self.config)
        
        # 将模型移动到设备上
        model = model.to(self.device)
        
        # 如果使用分布式训练，包装模型
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        
        return model
    
    def _create_optimizer_and_scheduler(self) -> Tuple[optim.Optimizer, Any]:
        """创建优化器和学习率调度器"""
        # 创建优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 创建学习率调度器（余弦退火）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate / 100
        )
        
        return optimizer, scheduler
    
    def _create_logger(self) -> SummaryWriter:
        """创建日志记录器"""
        # 创建日志目录
        log_dir = Path(self.config.log_dir) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建TensorBoard记录器
        writer = SummaryWriter(log_dir=str(log_dir))
        
        # 设置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        
        # 保存配置
        save_config_to_file(self.config, log_dir / "config.json")
        
        return writer
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        # 创建保存目录
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的状态
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model": self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_dice": self.best_dice,
            "config": self.config.__dict__
        }
        
        # 保存检查点
        if is_best:
            torch.save(state, save_dir / "best_model.pth")
            logging.info(f"保存最佳模型到 {save_dir / 'best_model.pth'}")
        
        if epoch % self.config.save_interval == 0:
            torch.save(state, save_dir / f"checkpoint_epoch_{epoch}.pth")
            logging.info(f"保存检查点到 {save_dir / f'checkpoint_epoch_{epoch}.pth'}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            logging.warning(f"检查点{checkpoint_path}不存在，从头开始训练")
            return
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复模型状态
        if self.distributed:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        
        # 恢复优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        # 恢复训练状态
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_dice = checkpoint.get("best_dice", 0.0)
        
        logging.info(f"从检查点{checkpoint_path}恢复训练，当前轮次：{self.epoch}，全局步数：{self.global_step}")
    
    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """训练一个轮次"""
        self.model.train()
        epoch_losses = {}
        
        # 设置进度条
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
        
        # 确定迭代次数（以标记数据为准）
        num_iters = len(labeled_loader)
        
        # 训练循环
        for i in range(num_iters):
            # 获取标记数据
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)
            
            # 获取非标记数据（如果有）
            unlabeled_batch = None
            if unlabeled_iter:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
            
            # 将数据移动到设备上
            labeled_images = labeled_batch["image"].to(self.device)
            labeled_masks = labeled_batch["mask"].to(self.device)
            labeled_text_tokens = labeled_batch["text_tokens"].to(self.device)
            
            unlabeled_images = None
            unlabeled_text_tokens = None
            if unlabeled_batch:
                unlabeled_images = unlabeled_batch["image"].to(self.device)
                unlabeled_text_tokens = unlabeled_batch["text_tokens"].to(self.device)
            
            # 前向传播和损失计算
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast():
                    losses, _ = self.model(
                        labeled_images=labeled_images,
                        labeled_masks=labeled_masks,
                        labeled_text_tokens=labeled_text_tokens,
                        unlabeled_images=unlabeled_images,
                        unlabeled_text_tokens=unlabeled_text_tokens,
                        return_loss=True
                    )
                    total_loss = losses["total_loss"]
                
                # 反向传播
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses, _ = self.model(
                    labeled_images=labeled_images,
                    labeled_masks=labeled_masks,
                    labeled_text_tokens=labeled_text_tokens,
                    unlabeled_images=unlabeled_images,
                    unlabeled_text_tokens=unlabeled_text_tokens,
                    return_loss=True
                )
                total_loss = losses["total_loss"]
                
                # 反向传播
                total_loss.backward()
                self.optimizer.step()
            
            # 更新训练状态
            self.global_step += 1
            
            # 记录损失
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item() if torch.is_tensor(v) else v)
            
            # 打印进度
            if i % 10 == 0:
                loss_str = ", ".join([f"{k}: {v[-1]:.4f}" for k, v in epoch_losses.items()])
                logging.info(f"Epoch {self.epoch}, Step {i}/{num_iters}, {loss_str}")
                
                # 记录到TensorBoard
                for k, v in epoch_losses.items():
                    self.writer.add_scalar(f"train/{k}", v[-1], self.global_step)
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_metrics = {"dice": [], "hausdorff": []}
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移动到设备上
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                text_tokens = batch["text_tokens"].to(self.device)
                
                # 前向传播
                _, outputs = self.model(
                    labeled_images=images,
                    labeled_masks=masks,
                    labeled_text_tokens=text_tokens,
                    return_loss=False
                )
                
                # 获取预测结果
                pred_masks = outputs["pred_masks"]
                
                # 计算评估指标
                binary_preds = (torch.sigmoid(pred_masks) > 0.5).float()
                
                # 计算Dice系数
                dice_scores = compute_dice_score(binary_preds, masks)
                val_metrics["dice"].extend(dice_scores.cpu().numpy().tolist())
                
                # 计算Hausdorff距离
                hausdorff_distances = compute_hausdorff_distance(binary_preds, masks)
                val_metrics["hausdorff"].extend(hausdorff_distances)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # 记录到TensorBoard
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.global_step)
        
        return avg_metrics
    
    def train(self, resume_from: Optional[str] = None) -> None:
        """训练模型"""
        # 加载检查点（如果有）
        if resume_from:
            self._load_checkpoint(resume_from)
        
        # 创建数据加载器
        labeled_train_loader = build_dataloader(self.config, mode="train", is_labeled=True)
        unlabeled_train_loader = None
        if hasattr(self.config, "unlabeled_data_dir") and self.config.unlabeled_data_dir:
            unlabeled_train_loader = build_dataloader(self.config, mode="train", is_labeled=False)
        val_loader = build_dataloader(self.config, mode="val", is_labeled=True)
        
        # 训练循环
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            # 训练一个轮次
            train_losses = self.train_epoch(labeled_train_loader, unlabeled_train_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/lr", current_lr, self.global_step)
            
            # 打印训练损失
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
            logging.info(f"Epoch {epoch} 训练完成，{loss_str}")
            
            # 验证
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                
                # 打印验证指标
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                logging.info(f"Epoch {epoch} 验证完成，{metrics_str}")
                
                # 保存最佳模型
                if val_metrics["dice"] > self.best_dice:
                    self.best_dice = val_metrics["dice"]
                    self._save_checkpoint(epoch, is_best=True)
                    logging.info(f"新的最佳Dice系数: {self.best_dice:.4f}")
            
            # 定期保存检查点
            self._save_checkpoint(epoch)
        
        # 训练结束
        logging.info(f"训练完成，最佳Dice系数: {self.best_dice:.4f}")
        self.writer.close()


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LISA3D模型训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地设备排名")
    args = parser.parse_args()
    
    # 初始化分布式训练
    distributed = args.local_rank >= 0
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
    
    # 加载配置
    if args.config:
        from ..config.lisa3d_config import load_config_from_file
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # 创建训练器
    trainer = LISA3DTrainer(
        config=config,
        local_rank=args.local_rank,
        distributed=distributed
    )
    
    # 开始训练
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
