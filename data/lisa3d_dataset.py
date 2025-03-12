import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Callable
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk
from transformers import AutoTokenizer


class Medical3DDataset(Dataset):
    """u533bu5b66u56feu50cf3Du6570u636eu96c6u57fau7c7b"""
    def __init__(
        self,
        data_dir: str,
        organ_names: List[str],
        image_size: Tuple[int, int, int] = (128, 256, 256),
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        """
        u521du59cbu5316u533bu5b66u56feu50cf3Du6570u636eu96c6
        
        Args:
            data_dir: u6570u636eu76eeu5f55
            organ_names: u5668u5b98u540du79f0u5217u8868
            image_size: u56feu50cfu5927u5c0f (D, H, W)
            transform: u6570u636eu589eu5f3au51fdu6570
            mode: u6a21u5f0fuff0c'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.organ_names = organ_names
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        # u68c0u67e5u6570u636eu76eeu5f55u662fu5426u5b58u5728
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"u6570u636eu76eeu5f55{data_dir}u4e0du5b58u5728")
        
        # u52a0u8f7du6570u636eu5217u8868
        self.samples = self._load_data_list()
    
    def _load_data_list(self) -> List[Dict]:
        """u52a0u8f7du6570u636eu5217u8868uff0cu9700u8981u5728u5b50u7c7bu4e2du5b9eu73b0"""
        raise NotImplementedError("_load_data_listu65b9u6cd5u9700u8981u5728u5b50u7c7bu4e2du5b9eu73b0")
    
    def _load_volume(self, file_path: str) -> np.ndarray:
        """u52a0u8f7d3Du4f53u79efu6570u636e"""
        # u652fu6301u591au79cdu683cu5f0fu7684u533bu5b66u56feu50cf
        if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
            # u4f7fu7528nibabelu52a0u8f7dNIfTIu683cu5f0f
            nifti_img = nib.load(file_path)
            volume = nifti_img.get_fdata()
        elif file_path.endswith(".mhd") or file_path.endswith(".mha"):
            # u4f7fu7528SimpleITKu52a0u8f7dMHD/MHAu683cu5f0f
            sitk_img = sitk.ReadImage(file_path)
            volume = sitk.GetArrayFromImage(sitk_img)
        else:
            raise ValueError(f"u4e0du652fu6301u7684u6587u4ef6u683cu5f0f: {file_path}")
        
        return volume
    
    def _resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """u8c03u6574u4f53u79efu5927u5c0fu5230u6307u5b9au5c3au5bf8"""
        # u521bu5efaSimpleITKu56feu50cf
        sitk_img = sitk.GetImageFromArray(volume)
        
        # u8bbeu7f6eu8c03u6574u5668
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)  # u7ebfu6027u63d2u503c
        resample.SetOutputDirection(sitk_img.GetDirection())
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        
        # u8ba1u7b97u65b0u7684u50cfu7d20u95f4u8ddd
        original_size = sitk_img.GetSize()
        original_spacing = sitk_img.GetSpacing()
        new_spacing = [
            (original_size[0] * original_spacing[0]) / self.image_size[2],
            (original_size[1] * original_spacing[1]) / self.image_size[1],
            (original_size[2] * original_spacing[2]) / self.image_size[0]
        ]
        
        # u8bbeu7f6eu8f93u51fau53c2u6570
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize((self.image_size[2], self.image_size[1], self.image_size[0]))
        
        # u6267u884cu91cdu91c7u6837
        resampled_img = resample.Execute(sitk_img)
        
        # u8f6cu6362u56deu6570u7ec4
        resampled_volume = sitk.GetArrayFromImage(resampled_img)
        
        return resampled_volume
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """u5f52u4e00u5316u4f53u79efu6570u636e"""
        # u8ba1u7b97u975eu96f6u533au57dfu7684u5747u503cu548cu6807u51c6u5dee
        mask = volume > 0
        if mask.sum() > 0:
            mean = volume[mask].mean()
            std = volume[mask].std()
            if std > 0:
                volume = (volume - mean) / std
            else:
                volume = volume - mean
        
        # u5f3au5236u5c06u503cu8303u56f4u8c03u6574u5230[-1, 1]
        volume = np.clip(volume, -1, 1)
        
        return volume
    
    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """u9884u5904u7406u4f53u79efu6570u636e"""
        # u8c03u6574u5927u5c0f
        volume = self._resize_volume(volume)
        
        # u5f52u4e00u5316
        volume = self._normalize_volume(volume)
        
        return volume
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """u83b7u53d6u6570u636eu6837u672cuff0cu9700u8981u5728u5b50u7c7bu4e2du5b9eu73b0"""
        raise NotImplementedError("__getitem__u65b9u6cd5u9700u8981u5728u5b50u7c7bu4e2du5b9eu73b0")


class LISA3DLabeledDataset(Medical3DDataset):
    """u6807u8bb0u6570u636eu96c6"""
    def __init__(
        self,
        data_dir: str,
        organ_names: List[str],
        tokenizer_path: str,
        image_size: Tuple[int, int, int] = (128, 256, 256),
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # u6dfbu52a0[seg]u6807u8bb0
        if "[seg]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[seg]"]})
            self.seg_token = "[seg]"
        else:
            self.seg_token = "[seg]"
        
        super().__init__(data_dir, organ_names, image_size, transform, mode)
    
    def _load_data_list(self) -> List[Dict]:
        """u52a0u8f7du6807u8bb0u6570u636eu5217u8868"""
        samples = []
        
        # u6839u636eu6a21u5f0fu9009u62e9u76eeu5f55
        target_dir = self.data_dir / self.mode
        if not target_dir.exists():
            raise FileNotFoundError(f"u76eeu5f55{target_dir}u4e0du5b58u5728")
        
        # u904du5386u6240u6709u75c5u4f8bu76eeu5f55
        for case_dir in target_dir.iterdir():
            if not case_dir.is_dir():
                continue
            
            # u67e5u627eu56feu50cfu548cu6807u6ce8u6587u4ef6
            image_file = None
            mask_files = {}
            
            for file_path in case_dir.glob("*"):
                if file_path.name.endswith((".nii", ".nii.gz", ".mhd", ".mha")):
                    if "mask" in file_path.name.lower() or "label" in file_path.name.lower():
                        # u5224u65adu5668u5b98u7c7bu578b
                        for i, organ in enumerate(self.organ_names):
                            if organ.lower() in file_path.name.lower():
                                mask_files[i] = str(file_path)
                                break
                        else:
                            # u5982u679cu6ca1u6709u5339u914du5668u5b98u540du79f0uff0cu53efu80fdu662fu5168u5668u5b98u6807u6ce8
                            if "all" in file_path.name.lower() or len(mask_files) == 0:
                                for i in range(len(self.organ_names)):
                                    mask_files[i] = str(file_path)
                    else:
                        image_file = str(file_path)
            
            # u786eu4fddu627eu5230u56feu50cfu548cu81f3u5c11u4e00u4e2au6807u6ce8
            if image_file and mask_files:
                samples.append({
                    "image_file": image_file,
                    "mask_files": mask_files,
                    "case_id": case_dir.name
                })
        
        return samples
    
    def _get_prompt_for_organ(self, organ_name: str) -> str:
        """u751fu6210u5668u5b98u7684u63d0u793au8bed"""
        return f"Segment the {organ_name} in this 3D medical image. [seg]"
    
    def _tokenize_text(self, text: str) -> Dict:
        """u5c06u6587u672cu8f6cu6362u4e3au6807u8bb0"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )
    
    def __getitem__(self, idx: int) -> Dict:
        """u83b7u53d6u6807u8bb0u6570u636eu6837u672c"""
        sample = self.samples[idx]
        
        # u52a0u8f7du56feu50cf
        image = self._load_volume(sample["image_file"])
        image = self._preprocess_volume(image)
        
        # u968fu673au9009u62e9u4e00u4e2au5668u5b98u8fdbu884cu5206u5272
        organ_idx = random.choice(list(sample["mask_files"].keys()))
        organ_name = self.organ_names[organ_idx]
        
        # u52a0u8f7du5bf9u5e94u7684u6807u6ce8
        mask_file = sample["mask_files"][organ_idx]
        mask = self._load_volume(mask_file)
        
        # u5982u679cu662fu5168u5668u5b98u6807u6ce8uff0cu9700u8981u63d0u53d6u5bf9u5e94u5668u5b98
        if len(sample["mask_files"]) == 1 and len(self.organ_names) > 1:
            # u5047u8bbeu6807u6ce8u4e2du4e0du540cu5668u5b98u4f7fu7528u4e0du540cu7684u6807u7b7eu503c
            mask = (mask == (organ_idx + 1)).astype(np.float32)
        else:
            # u4e8cu503cu5316u6807u6ce8
            mask = (mask > 0).astype(np.float32)
        
        # u8c03u6574u6807u6ce8u5927u5c0f
        mask = self._resize_volume(mask)
        
        # u751fu6210u6587u672cu63d0u793a
        prompt = self._get_prompt_for_organ(organ_name)
        tokenized = self._tokenize_text(prompt)
        
        # u6570u636eu589eu5f3a
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # u8f6cu6362u4e3aPyTorchu5f62u5f0f
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, D, H, W]
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, D, H, W]
        
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "text_tokens": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "organ_idx": organ_idx,
            "organ_name": organ_name,
            "case_id": sample["case_id"]
        }


class LISA3DUnlabeledDataset(Medical3DDataset):
    """u975eu6807u8bb0u6570u636eu96c6"""
    def __init__(
        self,
        data_dir: str,
        organ_names: List[str],
        tokenizer_path: str,
        image_size: Tuple[int, int, int] = (128, 256, 256),
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # u6dfbu52a0[seg]u6807u8bb0
        if "[seg]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[seg]"]})
            self.seg_token = "[seg]"
        else:
            self.seg_token = "[seg]"
        
        super().__init__(data_dir, organ_names, image_size, transform, mode)
    
    def _load_data_list(self) -> List[Dict]:
        """u52a0u8f7du975eu6807u8bb0u6570u636eu5217u8868"""
        samples = []
        
        # u6839u636eu6a21u5f0fu9009u62e9u76eeu5f55
        target_dir = self.data_dir / self.mode
        if not target_dir.exists():
            raise FileNotFoundError(f"u76eeu5f55{target_dir}u4e0du5b58u5728")
        
        # u904du5386u6240u6709u75c5u4f8bu76eeu5f55
        for case_dir in target_dir.iterdir():
            if not case_dir.is_dir():
                continue
            
            # u67e5u627eu56feu50cfu6587u4ef6
            image_file = None
            
            for file_path in case_dir.glob("*"):
                if file_path.name.endswith((".nii", ".nii.gz", ".mhd", ".mha")):
                    if "mask" not in file_path.name.lower() and "label" not in file_path.name.lower():
                        image_file = str(file_path)
                        break
            
            # u786eu4fddu627eu5230u56feu50cf
            if image_file:
                samples.append({
                    "image_file": image_file,
                    "case_id": case_dir.name
                })
        
        return samples
    
    def _get_prompt_for_organ(self, organ_name: str) -> str:
        """u751fu6210u5668u5b98u7684u63d0u793au8bed"""
        return f"Segment the {organ_name} in this 3D medical image. [seg]"
    
    def _tokenize_text(self, text: str) -> Dict:
        """u5c06u6587u672cu8f6cu6362u4e3au6807u8bb0"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )
    
    def __getitem__(self, idx: int) -> Dict:
        """u83b7u53d6u975eu6807u8bb0u6570u636eu6837u672c"""
        sample = self.samples[idx]
        
        # u52a0u8f7du56feu50cf
        image = self._load_volume(sample["image_file"])
        image = self._preprocess_volume(image)
        
        # u968fu673au9009u62e9u4e00u4e2au5668u5b98u8fdbu884cu5206u5272
        organ_idx = random.randint(0, len(self.organ_names) - 1)
        organ_name = self.organ_names[organ_idx]
        
        # u751fu6210u6587u672cu63d0u793a
        prompt = self._get_prompt_for_organ(organ_name)
        tokenized = self._tokenize_text(prompt)
        
        # u6570u636eu589eu5f3a
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # u8f6cu6362u4e3aPyTorchu5f62u5f0f
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, D, H, W]
        
        return {
            "image": image_tensor,
            "text_tokens": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "organ_idx": organ_idx,
            "organ_name": organ_name,
            "case_id": sample["case_id"]
        }


def build_transforms(mode: str = "train"):
    """u6784u5efau6570u636eu589eu5f3au51fdu6570"""
    try:
        import albumentations as A
        
        if mode == "train":
            # u8badu7ec3u96c6u589eu5f3a
            return lambda **kwargs: {
                "image": np.array(kwargs["image"]),
                "mask": np.array(kwargs.get("mask", None))
            }
        else:
            # u9a8cu8bc1u96c6/u6d4bu8bd5u96c6u589eu5f3a
            return lambda **kwargs: {
                "image": np.array(kwargs["image"]),
                "mask": np.array(kwargs.get("mask", None))
            }
    except ImportError:
        # u5982u679cu6ca1u6709albumentationsuff0cu8fd4u56deu7b80u5355u7684u6052u7b49u53d8u6362
        return lambda **kwargs: {
            "image": np.array(kwargs["image"]),
            "mask": np.array(kwargs.get("mask", None))
        }


def build_dataloader(
    config,
    mode: str = "train",
    is_labeled: bool = True,
):
    """u6784u5efau6570u636eu52a0u8f7du5668"""
    # u6784u5efau6570u636eu589eu5f3a
    transform = build_transforms(mode)
    
    # u9009u62e9u6570u636eu96c6u7c7b
    dataset_class = LISA3DLabeledDataset if is_labeled else LISA3DUnlabeledDataset
    
    # u6784u5efau6570u636eu96c6
    dataset = dataset_class(
        data_dir=config.data_dir,
        organ_names=config.organ_names,
        tokenizer_path=config.mllm_path,
        image_size=config.image_size,
        transform=transform,
        mode=mode
    )
    
    # u786eu5b9au6279u6b21u5927u5c0f
    batch_size = config.labeled_batch_size if is_labeled else config.unlabeled_batch_size
    
    # u6784u5efau6570u636eu52a0u8f7du5668
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(mode == "train")
    )
    
    return dataloader
