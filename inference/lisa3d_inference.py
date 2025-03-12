import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

# 导入自定义模块
from ..model.LISA3D_complete import LISA3DForSegmentation
from ..config.lisa3d_config import LISA3DConfig, load_config_from_file, get_default_config
from ..data.lisa3d_dataset import preprocess_volume, tokenize_text
from ..utils.lisa3d_metrics import compute_all_metrics


class LISA3DInference:
    """LISA3D模型推理类"""
    def __init__(
        self,
        config: LISA3DConfig,
        checkpoint_path: str,
        device: str = "cuda",
    ):
        """
        初始化推理类
        
        Args:
            config: 配置对象
            checkpoint_path: 模型检查点路径
            device: 设备类型
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = self._load_model(checkpoint_path)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        加载模型
        
        Args:
            checkpoint_path: 模型检查点路径
            
        Returns:
            加载好的模型
        """
        # 创建模型实例
        model = LISA3DForSegmentation(self.config)
        
        # 加载检查点
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点{checkpoint_path}不存在")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])
        
        # 将模型移动到设备上并设置为评估模式
        model = model.to(self.device)
        model.eval()
        
        logging.info(f"成功从{checkpoint_path}加载模型")
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像张量和原始图像信息
        """
        # 加载图像
        if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):
            # 使用nibabel加载NIfTI文件
            nib_img = nib.load(image_path)
            image_data = nib_img.get_fdata()
            affine = nib_img.affine
            header = nib_img.header
            
            # 保存原始图像信息
            image_info = {
                "affine": affine,
                "header": header,
                "shape": image_data.shape,
                "format": "nifti"
            }
        else:
            # 使用SimpleITK加载其他格式
            sitk_img = sitk.ReadImage(image_path)
            image_data = sitk.GetArrayFromImage(sitk_img)
            spacing = sitk_img.GetSpacing()
            origin = sitk_img.GetOrigin()
            direction = sitk_img.GetDirection()
            
            # 保存原始图像信息
            image_info = {
                "spacing": spacing,
                "origin": origin,
                "direction": direction,
                "shape": image_data.shape,
                "format": "sitk"
            }
        
        # 预处理图像
        processed_volume = preprocess_volume(
            image_data,
            target_shape=self.config.input_shape,
            normalize=True
        )
        
        # 转换为张量并添加批次维度
        image_tensor = torch.from_numpy(processed_volume).float().unsqueeze(0)  # [1, 1, D, H, W]
        
        return image_tensor, image_info
    
    def predict(self, image_path: str, prompt_text: str) -> Dict[str, Any]:
        """
        对图像进行分割预测
        
        Args:
            image_path: 图像路径
            prompt_text: 提示文本（例如："segment the liver"）
            
        Returns:
            预测结果字典
        """
        # 预处理图像
        image_tensor, image_info = self.preprocess_image(image_path)
        
        # 处理文本提示
        text_tokens = tokenize_text(prompt_text, self.config.tokenizer_name_or_path)
        text_tokens = {k: torch.tensor(v).unsqueeze(0) for k, v in text_tokens.items()}  # 添加批次维度
        
        # 将数据移动到设备上
        image_tensor = image_tensor.to(self.device)
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        
        # 进行推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                _, outputs = self.model(
                    labeled_images=image_tensor,
                    labeled_text_tokens=text_tokens,
                    return_loss=False
                )
        
        # 获取预测结果
        pred_masks = outputs["pred_masks"]  # [1, 1, D, H, W]
        binary_masks = (torch.sigmoid(pred_masks) > 0.5).float()
        
        # 将预测结果转回原始图像大小
        resized_masks = self._resize_to_original(binary_masks.cpu().numpy(), image_info["shape"])
        
        # 准备返回结果
        result = {
            "pred_masks": binary_masks.cpu().numpy(),
            "resized_masks": resized_masks,
            "image_info": image_info,
            "prompt_text": prompt_text
        }
        
        return result
    
    def _resize_to_original(self, masks: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        将预测的掩码调整回原始图像大小
        
        Args:
            masks: 预测的掩码，形状为 [B, C, D, H, W]
            original_shape: 原始图像形状
            
        Returns:
            调整大小后的掩码
        """
        from scipy.ndimage import zoom
        
        # 移除批次和通道维度
        mask = masks[0, 0]  # [D, H, W]
        
        # 计算缩放因子
        zoom_factors = (
            original_shape[0] / mask.shape[0],
            original_shape[1] / mask.shape[1],
            original_shape[2] / mask.shape[2]
        )
        
        # 调整大小
        resized_mask = zoom(mask, zoom_factors, order=0)  # 使用最近邻插值
        
        # 确保二值化
        resized_mask = (resized_mask > 0.5).astype(np.float32)
        
        return resized_mask
    
    def save_prediction(self, result: Dict[str, Any], output_path: str) -> None:
        """
        保存预测结果
        
        Args:
            result: 预测结果字典
            output_path: 输出路径
        """
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取掩码和图像信息
        resized_mask = result["resized_masks"]
        image_info = result["image_info"]
        
        # 根据原始图像格式保存掩码
        if image_info["format"] == "nifti":
            # 创建NIfTI图像并保存
            mask_nii = nib.Nifti1Image(resized_mask, image_info["affine"], image_info["header"])
            nib.save(mask_nii, output_path)
        else:
            # 创建SimpleITK图像并保存
            mask_sitk = sitk.GetImageFromArray(resized_mask)
            mask_sitk.SetSpacing(image_info["spacing"])
            mask_sitk.SetOrigin(image_info["origin"])
            mask_sitk.SetDirection(image_info["direction"])
            sitk.WriteImage(mask_sitk, output_path)
        
        logging.info(f"预测结果已保存到 {output_path}")
    
    def visualize(self, result: Dict[str, Any], output_path: Optional[str] = None, slice_idx: Optional[int] = None) -> None:
        """
        可视化预测结果
        
        Args:
            result: 预测结果字典
            output_path: 可视化结果保存路径（可选）
            slice_idx: 要可视化的切片索引（可选，如果未指定则选择中间切片）
        """
        # 加载原始图像
        image_path = result.get("image_path")
        if image_path and os.path.exists(image_path):
            if image_path.endswith(".nii") or image_path.endswith(".nii.gz"):
                image_data = nib.load(image_path).get_fdata()
            else:
                image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        else:
            # 如果没有原始图像路径，使用全零图像
            image_data = np.zeros(result["image_info"]["shape"])
        
        # 获取掩码
        mask = result["resized_masks"]
        
        # 选择切片
        if slice_idx is None:
            slice_idx = mask.shape[0] // 2  # 默认选择中间切片
        
        # 确保切片索引在有效范围内
        slice_idx = max(0, min(slice_idx, mask.shape[0] - 1))
        
        # 提取切片
        image_slice = image_data[slice_idx]
        mask_slice = mask[slice_idx]
        
        # 创建图像
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image_slice, cmap="gray")
        plt.title(f"原始图像 (切片 {slice_idx})")
        plt.axis("off")
        
        # 显示分割掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask_slice, cmap="hot")
        plt.title(f"分割掩码 (切片 {slice_idx})")
        plt.axis("off")
        
        # 显示叠加结果
        plt.subplot(1, 3, 3)
        plt.imshow(image_slice, cmap="gray")
        plt.imshow(mask_slice, cmap="hot", alpha=0.5)
        plt.title(f"叠加结果 (切片 {slice_idx})")
        plt.axis("off")
        
        # 添加提示文本作为标题
        plt.suptitle(f"提示: {result['prompt_text']}")
        plt.tight_layout()
        
        # 保存或显示图像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logging.info(f"可视化结果已保存到 {output_path}")
        else:
            plt.show()
    
    def batch_inference(self, image_dir: str, prompts: List[str], output_dir: str, visualize: bool = True) -> Dict[str, List[float]]:
        """
        批量推理
        
        Args:
            image_dir: 图像目录
            prompts: 提示文本列表
            output_dir: 输出目录
            visualize: 是否生成可视化结果
            
        Returns:
            评估指标字典（如果有真实标签）
        """
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图像文件
        image_files = []
        for ext in [".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"]:
            image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
        
        # 检查是否有匹配的标签文件
        has_labels = False
        label_dir = Path(image_dir).parent / "labels"
        if label_dir.exists():
            has_labels = True
        
        # 初始化评估指标
        all_metrics = {}
        
        # 处理每个图像
        for image_file in tqdm(image_files, desc="处理图像"):
            image_name = image_file.stem
            
            # 对每个提示进行推理
            for prompt in prompts:
                # 生成输出文件名
                prompt_slug = prompt.replace(" ", "_").lower()
                output_mask_path = output_dir / f"{image_name}_{prompt_slug}.nii.gz"
                output_viz_path = output_dir / f"{image_name}_{prompt_slug}_viz.png"
                
                # 执行推理
                result = self.predict(str(image_file), prompt)
                result["image_path"] = str(image_file)  # 添加图像路径用于可视化
                
                # 保存预测结果
                self.save_prediction(result, str(output_mask_path))
                
                # 可视化结果
                if visualize:
                    self.visualize(result, str(output_viz_path))
                
                # 如果有标签，计算评估指标
                if has_labels:
                    label_file = label_dir / f"{image_name}_label.nii.gz"
                    if label_file.exists():
                        # 加载标签
                        if str(label_file).endswith(".nii") or str(label_file).endswith(".nii.gz"):
                            label_data = nib.load(str(label_file)).get_fdata()
                        else:
                            label_data = sitk.GetArrayFromImage(sitk.ReadImage(str(label_file)))
                        
                        # 确保标签是二值的
                        label_data = (label_data > 0.5).astype(np.float32)
                        
                        # 将预测和标签转换为张量
                        pred_tensor = torch.from_numpy(result["resized_masks"]).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                        label_tensor = torch.from_numpy(label_data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                        
                        # 计算评估指标
                        metrics = compute_all_metrics(pred_tensor, label_tensor)
                        
                        # 记录指标
                        for k, v in metrics.items():
                            if k not in all_metrics:
                                all_metrics[k] = []
                            
                            # 如果是张量，转换为列表
                            if isinstance(v, torch.Tensor):
                                all_metrics[k].extend(v.cpu().numpy().tolist())
                            elif isinstance(v, list):
                                all_metrics[k].extend(v)
                            else:
                                all_metrics[k].append(v)
        
        # 如果有评估指标，计算平均值并保存
        if all_metrics:
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            
            # 保存评估指标
            metrics_path = output_dir / "metrics.txt"
            with open(metrics_path, "w") as f:
                for k, v in avg_metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
            
            logging.info(f"评估指标已保存到 {metrics_path}")
            
            # 打印评估结果
            logging.info("评估结果:")
            for k, v in avg_metrics.items():
                logging.info(f"{k}: {v:.4f}")
        
        return all_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LISA3D模型推理")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--image_dir", type=str, help="图像目录（用于批量推理）")
    parser.add_argument("--prompt", type=str, default="segment the organ", help="分割提示文本")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")
    parser.add_argument("--device", type=str, default="cuda", help="设备类型 (cuda 或 cpu)")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_file(args.config)
    
    # 创建推理对象
    inferencer = LISA3DInference(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 单张图像推理或批量推理
    if args.image:
        # 单张图像推理
        result = inferencer.predict(args.image, args.prompt)
        
        # 保存预测结果
        output_path = Path(args.output_dir) / f"prediction.nii.gz"
        inferencer.save_prediction(result, str(output_path))
        
        # 可视化结果
        if args.visualize:
            result["image_path"] = args.image  # 添加图像路径用于可视化
            viz_path = Path(args.output_dir) / "visualization.png"
            inferencer.visualize(result, str(viz_path))
    
    elif args.image_dir:
        # 批量推理
        prompts = [p.strip() for p in args.prompt.split(",")]
        inferencer.batch_inference(
            image_dir=args.image_dir,
            prompts=prompts,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
    
    else:
        logging.error("请指定 --image 或 --image_dir 参数")


if __name__ == "__main__":
    main()
