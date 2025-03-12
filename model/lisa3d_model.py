import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration

from .lisa3d_sam import SAMEncoder3D, SAMDecoder3D, MultiModalFusion, apply_lora


class Lisa3DMetaModel(nn.Module):
    """LISA3Du5143u6a21u578buff0cu5c01u88c5u5171u4eabu7ec4u4ef6"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DMetaModel, self).__init__()

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        
        # u521du59cbu5316u5171u4eabu7ec4u4ef6
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # u591au6a21u6001u8bedu8a00u6a21u578b
        self.mllm_model = LlavaForConditionalGeneration.from_pretrained(
            config.mllm_path,
            torch_dtype=torch.float16,
        )
        
        # u5e94u7528LoRAu5230MLLM
        if config.use_lora:
            self.mllm_model = apply_lora(
                self.mllm_model,
                r=config.lora_r,
                target_modules=["q_proj", "v_proj"]
            )
        
        # u51bbu7ed3MLLMu53c2u6570
        for param in self.mllm_model.parameters():
            param.requires_grad = False
        
        # u7279u5f81u878du5408u6a21u5757
        self.fusion_module = MultiModalFusion(
            text_dim=config.hidden_size,
            image_dim=config.out_dim
        )
        
        # u8bbeu7f6eu878du5408u6a21u5757u4e3au53efu8badu7ec3
        self.fusion_module.train()
        for param in self.fusion_module.parameters():
            param.requires_grad = True
        
        # u5b9au4e49u7279u6b8au6807u8bb0
        self.seg_token_idx = config.seg_token_idx


class Lisa3DModel(Lisa3DMetaModel):
    """u53ccu7f16u7801u5668/u89e3u7801u5668u7ed3u6784u7684LISA3Du6a21u578b"""
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Lisa3DModel, self).__init__(config, **kwargs)

        # 3Du7279u5b9au914du7f6e
        self.config.use_cache = False
        self.config.image_aspect_ratio = "3D"
        
        # u521du59cbu5316u53ccu7f16u7801u5668
        self.encoder_l = SAMEncoder3D(
            pretrained_path=self.vision_pretrained,
            with_lora=config.use_lora
        )
        
        self.encoder_u = SAMEncoder3D(
            pretrained_path=self.vision_pretrained,
            with_lora=config.use_lora
        )
        
        # u521du59cbu5316u53ccu89e3u7801u5668
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
        
        # u521du59cbu5316u6a21u578bu6743u91cd
        self._init_weights()
    
    def _init_weights(self):
        # u521du59cbu5316u6a21u578bu6743u91cd
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
        """u63d0u53d6[seg]u6807u8bb0u7684u7279u5f81"""
        # u627eu5230[seg]u6807u8bb0u7684u4f4du7f6e
        seg_token_mask = (text_tokens == self.seg_token_idx).bool()
        
        # u5982u679cu6ca1u6709u627eu5230[seg]u6807u8bb0uff0cu8fd4u56deNone
        if not seg_token_mask.any():
            return None
        
        # u63d0u53d6u6700u540eu4e00u5c42u9690u85cfu72b6u6001
        last_hidden_state = text_outputs.last_hidden_state  # [B, L, D]
        
        # u63d0u53d6[seg]u6807u8bb0u7684u7279u5f81
        seg_features = []
        for b in range(seg_token_mask.size(0)):
            # u627eu5230u5f53u524du6279u6b21u4e2du7684[seg]u6807u8bb0u4f4du7f6e
            seg_indices = seg_token_mask[b].nonzero(as_tuple=True)[0]
            
            if len(seg_indices) > 0:
                # u63d0u53d6u6240u6709[seg]u6807u8bb0u7684u7279u5f81u5e76u5e73u5747
                batch_seg_features = last_hidden_state[b, seg_indices]
                seg_features.append(batch_seg_features.mean(0))
            else:
                # u5982u679cu6ca1u6709[seg]u6807u8bb0uff0cu4f7fu7528u96f6u5411u91cf
                seg_features.append(torch.zeros_like(last_hidden_state[b, 0]))
        
        # u5806u53e0u6240u6709u6279u6b21u7684u7279u5f81
        seg_features = torch.stack(seg_features, dim=0)  # [B, D]
        
        return seg_features
    
    def process_text(self, text_tokens):
        """u5904u7406u6587u672cu8f93u5165u5e76u63d0u53d6u7279u5f81"""
        # u901au8fc7MLLMu5904u7406u6587u672c
        with torch.no_grad():
            text_outputs = self.mllm_model(
                input_ids=text_tokens,
                return_dict=True
            )
        
        # u63d0u53d6[seg]u6807u8bb0u7684u7279u5f81
        seg_features = self.extract_seg_token_features(text_tokens, text_outputs)
        
        # u5982u679cu6ca1u6709u627eu5230[seg]u6807u8bb0uff0cu8fd4u56deNone
        if seg_features is None:
            return None
        
        # u6295u5f71u6587u672cu7279u5f81
        projected_text = self.fusion_module(seg_features)
        
        return projected_text
    
    def forward_labeled(self, image, text_tokens):
        """u6807u8bb0u6570u636eu6d41u5904u7406"""
        # u5904u7406u6587u672cu8f93u5165
        text_features = self.process_text(text_tokens)
        
        # u901au8fc7u7f16u7801u5668u5904u7406u56feu50cf
        image_features = self.encoder_l(image)
        
        # u901au8fc7u89e3u7801u5668u751fu6210u63a9u7801
        masks = self.decoder_l(image_features, text_features)
        
        return masks
    
    def forward_unlabeled(self, image, text_tokens):
        """u975eu6807u8bb0u6570u636eu6d41u5904u7406"""
        # u5904u7406u6587u672cu8f93u5165
        text_features = self.process_text(text_tokens)
        
        # u751fu6210u4f2au6807u7b7euff08u4f7fu7528u7f16u7801u5668Lu548cu89e3u7801u5668Luff09
        with torch.no_grad():
            image_features_l = self.encoder_l(image)
            pseudo_masks = self.decoder_l(image_features_l, text_features)
        
        # u751fu6210u9884u6d4buff08u4f7fu7528u7f16u7801u5668Uu548cu89e3u7801u5668Uuff09
        image_features_u = self.encoder_u(image)
        pred_masks = self.decoder_u(image_features_u, text_features)
        
        return pseudo_masks, pred_masks
    
    def forward(self, image, text_tokens, is_labeled=True):
        """u524du5411u4f20u64ad"""
        if is_labeled:
            return self.forward_labeled(image, text_tokens)
        else:
            return self.forward_unlabeled(image, text_tokens)
