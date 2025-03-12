from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class LISA3DConfig:
    """LISA3D模型配置"""
    # 模型路径
    mllm_path: str = "liuhaotian/llava-v1.5-7b"  # LLaVA模型路径
    vision_pretrained: Optional[str] = None  # SAM预训练模型路径
    
    # 模型结构参数
    hidden_size: int = 4096  # MLLM隐藏层大小
    out_dim: int = 256  # 特征维度
    num_cls: int = 14  # 分割类别数量（器官数量）
    seg_token_idx: int = 32000  # [seg]标记的索引
    
    # LoRA参数
    use_lora: bool = True  # 是否使用LoRA
    lora_r: int = 8  # LoRA秩
    lora_alpha: int = 16  # LoRA缩放因子
    lora_dropout: float = 0.1  # LoRA dropout率
    
    # 损失函数权重
    dice_weight: float = 1.0  # Dice损失权重
    ce_weight: float = 1.0  # 交叉熵损失权重
    consistency_weight: float = 0.5  # 一致性损失权重
    
    # 训练参数
    batch_size: int = 4  # 批次大小
    labeled_batch_size: int = 2  # 标记数据批次大小
    unlabeled_batch_size: int = 2  # 非标记数据批次大小
    learning_rate: float = 1e-4  # 学习率
    weight_decay: float = 1e-5  # 权重衰减
    num_epochs: int = 100  # 训练轮数
    warmup_steps: int = 500  # 预热步数
    
    # 数据参数
    image_size: List[int] = (128, 256, 256)  # 图像大小 (D, H, W)
    num_workers: int = 4  # 数据加载线程数
    
    # 器官列表
    organ_names: List[str] = [
        "spleen", "right kidney", "left kidney", "gallbladder", "liver", 
        "stomach", "pancreas", "aorta", "inferior vena cava", "right adrenal gland", 
        "left adrenal gland", "duodenum", "bladder", "prostate/uterus"
    ]
    
    # 保存和日志
    save_dir: str = "./checkpoints"  # 模型保存路径
    log_dir: str = "./logs"  # 日志保存路径
    save_interval: int = 10  # 保存间隔（轮）
    eval_interval: int = 5  # 评估间隔（轮）
    
    # 混合精度训练
    use_amp: bool = True  # 是否使用混合精度训练
    
    def __post_init__(self):
        """验证配置参数"""
        assert len(self.organ_names) == self.num_cls, \
            f"器官数量({len(self.organ_names)})与分割类别数量({self.num_cls})不匹配"


def get_default_config() -> LISA3DConfig:
    """获取默认配置"""
    return LISA3DConfig()


def load_config_from_file(config_path: str) -> LISA3DConfig:
    """从文件加载配置"""
    import json
    import os
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件{config_path}不存在")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # 创建配置对象
    config = LISA3DConfig(**config_dict)
    
    return config


def save_config_to_file(config: LISA3DConfig, config_path: str) -> None:
    """保存配置到文件"""
    import json
    import os
    
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # 转换为字典
    config_dict = {k: v for k, v in config.__dict__.items()}
    
    # 保存到文件
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
