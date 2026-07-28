# -*- coding: utf-8 -*-
import argparse
import csv
import datetime
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger
from mmengine import Config
from torch.utils import data
from tqdm import tqdm

import methods as model_zoo
from utils import io, ops, pipeline, pt_utils, py_utils, recorder

from segment_anything import sam_model_registry

import monai

import open_clip
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip import get_tokenizer

from prompts import get_prompt_template_by_name
from methods.base.base_ops import LayerNorm2d, PixelNormalizer, rescale_2x, resize_to


""""""
from methods_my.mfmsnet import get_freq_indices, ViTCLIP, UpsampleBlock
from methods_my.cosine import *
from methods_my.BidirectionalCrossAttention import *
from methods_my.crossattforlast import *
from methods_my.target_background_hypergraph import TargetBackgroundHypergraphModule
""""""
# from base_ops import *

logger.remove()
logger_format = "[<green>{time:YYYY-MM-DD HH:mm:ss} - {file}</>] <lvl>{message}</>"
logger.add(sys.stderr, level="DEBUG", format=logger_format)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def parse_device_ids(devices: str):
    if not devices:
        return []
    if devices.lower() == "all":
        return list(range(torch.cuda.device_count()))
    return [int(device_id.strip()) for device_id in devices.split(",") if device_id.strip()]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


CLIP_BACKBONE_CONFIGS = {
    "vit-b-16": {
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "local_dir": "CLIP-ViT-B-16",
    },
    "vit-l-14": {
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "local_dir": "CLIP-ViT-L-14",
    },
    "convnext-l": {
        "model_name": "convnext_large_d_320",
        "pretrained": "laion2b_s29b_b131k_ft_soup",
        "local_dir": "CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    },
    "convnext-xxl": {
        "model_name": "convnext_xxlarge",
        "pretrained": "laion2b_s34b_b82k_augreg_soup",
        "local_dir": "CLIP-ConvNeXt-XXL",
    },
}


class CLIPBackbone(nn.Module):
    """Frozen OpenCLIP backbone with a trainable, common four-level feature pyramid."""

    def __init__(
        self,
        backbone_name="convnext-l",
        pretrained=None,
        local_model_path=None,
        cache_root=None,
        template_set="camoprompts",
    ):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name not in CLIP_BACKBONE_CONFIGS:
            raise ValueError(
                f"Unsupported CLIP backbone {backbone_name!r}; "
                f"choose from {sorted(CLIP_BACKBONE_CONFIGS)}"
            )
        backbone_cfg = CLIP_BACKBONE_CONFIGS[backbone_name]
        model_name = backbone_cfg["model_name"]
        pretrained_was_overridden = pretrained is not None
        pretrained = pretrained or backbone_cfg["pretrained"]
        cache_root = cache_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
        local_dir = os.path.join(cache_root, backbone_cfg["local_dir"])

        if local_model_path is not None:
            if not os.path.isfile(local_model_path):
                raise FileNotFoundError(f"CLIP checkpoint not found: {local_model_path}")
            pretrained_source = local_model_path
            logger.info(f"Loading CLIP checkpoint from {local_model_path}")
        else:
            local_candidates = (
                os.path.join(local_dir, "open_clip_pytorch_model.bin"),
                os.path.join(local_dir, "open_clip_model.safetensors"),
                os.path.join(local_dir, "pytorch_model.bin"),
            )
            pretrained_source = None
            if not pretrained_was_overridden:
                pretrained_source = next((path for path in local_candidates if os.path.isfile(path)), None)
            if pretrained_source is None:
                os.makedirs(local_dir, exist_ok=True)
                pretrained_source = pretrained
                logger.info(
                    f"No local weights found for {backbone_name}; OpenCLIP will download "
                    f"{model_name}/{pretrained} into {local_dir}"
                )
            else:
                logger.info(f"Loading cached CLIP checkpoint from {pretrained_source}")

        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_source,
            cache_dir=local_dir,
        )
        self.backbone_name = backbone_name
        self.model_name = model_name
        self.model_type = "vit" if model_name.startswith("ViT-") else "convnext"
        self.mean = getattr(self.clip_model.visual, "image_mean", None) or OPENAI_DATASET_MEAN
        self.std = getattr(self.clip_model.visual, "image_std", None) or OPENAI_DATASET_STD
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.template_set = get_prompt_template_by_name(template_set)
        self.dim_latent = self.clip_model.text_projection.shape[-1]
        self.feature_channels = [192, 384, 768, 1536]
        raw_channels = self._infer_raw_channels()
        self.feature_adapters = nn.ModuleDict(
            {
                name: nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
                for name, in_ch, out_ch in zip(
                    ("res2", "res3", "res4", "res5"),
                    raw_channels,
                    self.feature_channels,
                )
            }
        )
        self.out_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32, "emb": -1}
        self.out_chans = dict(zip(("res2", "res3", "res4", "res5"), self.feature_channels))
        self.out_chans["emb"] = self.dim_latent
        logger.info(
            f"Created CLIP backbone {backbone_name} ({model_name}/{pretrained}); "
            f"text/image embedding dim={self.dim_latent}"
        )

    def _infer_raw_channels(self):
        visual = self.clip_model.visual
        if self.model_type == "vit":
            width = int(visual.conv1.out_channels)
            return [width] * 4
        trunk = visual.trunk
        channels = []
        for stage in trunk.stages:
            stage_channels = None
            for module in reversed(list(stage.modules())):
                if isinstance(module, nn.Conv2d):
                    stage_channels = module.out_channels
                    break
            if stage_channels is None:
                raise RuntimeError(f"Cannot infer output channels for {self.model_name} stage")
            channels.append(stage_channels)
        if len(channels) != 4:
            raise RuntimeError(f"Expected four ConvNeXt stages, got {len(channels)}")
        return channels

    def output_shape(self):
        return {
            name: dict(channels=self.out_chans[name], stride=self.out_strides[name])
            for name in ["res2", "res3", "res4", "res5", "emb"]
        }

    @property
    def device(self):
        for param in self.clip_model.parameters():
            return param.device

    @property
    def text_device(self):
        return self.clip_model.token_embedding.weight.device

    @torch.no_grad()
    def get_text_embs(self, text_list, normalize=True):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
        text_tokens = self.text_tokenizer(text_list).to(self.text_device)

        # list -> TD
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]
        #
        x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768

        x = x.permute(1, 0, 2)  # NLD -> LND
        attn_mask = self.clip_model.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=text_tokens.device)
        x = self.clip_model.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        # take feats from the eot embedding (eot_token is the highest number in each sequence)
        text_embs = (
            x[torch.arange(x.shape[0], device=text_tokens.device), text_tokens.argmax(dim=-1)]
            @ self.clip_model.text_projection
        )

        if normalize:
            text_embs = F.normalize(text_embs, dim=-1)  # Nc,768
        return text_embs

    @torch.no_grad()
    def get_text_embs_by_template(self, text_list):
        """对输入的所有类别名称都使用一套模板构建平均嵌入
        return: NumberofClasses,D
        """
        self.eval()

        text_embs = []
        for text in text_list:
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.text_tokenizer([template.format(text) for template in self.template_set]).to(
                self.text_device
            )#text——tokens（6，77）

            # list -> TD
            cast_dtype = self.clip_model.transformer.get_cast_dtype()
            x = self.clip_model.token_embedding(text_tokens).to(cast_dtype)  # [num_temp, n_ctx, d_model]（6，77，768）
            #
            x = x + self.clip_model.positional_embedding.to(cast_dtype)  # 80,77,768  77,768（6，77，768）

            x = x.permute(1, 0, 2)  # NLD -> LND
            attn_mask = self.clip_model.attn_mask
            if attn_mask is not None:
                attn_mask = attn_mask.to(device=text_tokens.device)
            x = self.clip_model.transformer(x, attn_mask=attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]

            # take feats from the eot embedding (eot_token is the highest number in each sequence) => Nc,768
            text_emb = (
                x[torch.arange(x.shape[0], device=text_tokens.device), text_tokens.argmax(dim=-1)]
                @ self.clip_model.text_projection
            ) #（6，768）
            text_embs.append(text_emb)
        text_embs = torch.stack(text_embs, dim=0)  # Nc,Nt,768 （14，6，768）

        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.mean(1)#（14，768）
        text_embs /= text_embs.norm(dim=-1, keepdim=True)#（14，768）
        return text_embs  # Nc,768 （14，768）

    def visual_feats_to_embs(self, x, normalize: bool = True):
        """
        将图像特征转换为图像嵌入向量
        """
        self.eval()

        if self.model_type == "convnext":
            x = self.clip_model.visual.trunk.head(x)
            x = self.clip_model.visual.head(x)
        else:
            x = x.mean(dim=(-2, -1))
            if self.clip_model.visual.proj is not None:
                x = x @ self.clip_model.visual.proj
        return F.normalize(x, dim=-1) if normalize else x

    def get_visual_feats(self, x):
        self.clip_model.eval()
        with torch.no_grad():
            if self.model_type == "convnext":
                raw_out = self._get_convnext_visual_feats(x)
            else:
                raw_out = self._get_vit_visual_feats(x)
        out = {
            name: self.feature_adapters[name](raw_out[name]).contiguous()
            for name in ("res2", "res3", "res4", "res5")
        }
        out["clip_vis_dense"] = raw_out["clip_vis_dense"].contiguous()
        return out

    def _get_convnext_visual_feats(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        for i, stage in enumerate(self.clip_model.visual.trunk.stages):
            x = stage(x)
            out[f"res{i + 2}"] = x
        out["clip_vis_dense"] = self.clip_model.visual.trunk.norm_pre(x)
        return out

    @staticmethod
    def _resize_vit_positional_embedding(positional_embedding, grid_hw, dtype, device):
        pos = positional_embedding.to(device=device, dtype=dtype)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
        old_h = int(round(patch_pos.shape[1] ** 0.5))
        old_w = patch_pos.shape[1] // old_h
        if old_h * old_w != patch_pos.shape[1]:
            raise RuntimeError(f"Non-rectangular ViT positional embedding with {patch_pos.shape[1]} patches")
        if (old_h, old_w) != tuple(grid_hw):
            patch_pos = patch_pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=grid_hw, mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_hw[0] * grid_hw[1], -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def _get_vit_visual_feats(self, image):
        visual = self.clip_model.visual
        batch_size = image.shape[0]
        x = visual.conv1(image)
        grid_hw = x.shape[-2:]
        x = x.reshape(batch_size, x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(device=x.device, dtype=x.dtype)
        cls = cls.reshape(1, 1, -1).expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._resize_vit_positional_embedding(
            visual.positional_embedding, grid_hw, x.dtype, x.device
        )
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, as expected by OpenCLIP 2.24

        blocks = visual.transformer.resblocks
        capture_indices = [max(0, round((i + 1) * len(blocks) / 4) - 1) for i in range(4)]
        captured = []
        for index, block in enumerate(blocks):
            x = block(x)
            if index in capture_indices:
                tokens = x.permute(1, 0, 2)[:, 1:]
                feat = tokens.reshape(batch_size, grid_hw[0], grid_hw[1], -1).permute(0, 3, 1, 2)
                captured.append(feat.contiguous())
        if len(captured) != 4:
            raise RuntimeError(f"Expected four ViT intermediate features, got {len(captured)}")

        input_hw = image.shape[-2:]
        out = {}
        for name, feat, stride in zip(("res2", "res3", "res4", "res5"), captured, (4, 8, 16, 32)):
            target_hw = (max(1, input_hw[0] // stride), max(1, input_hw[1] // stride))
            out[name] = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        dense = captured[-1].permute(0, 2, 3, 1)
        dense = visual.ln_post(dense).permute(0, 3, 1, 2)
        out["clip_vis_dense"] = dense.contiguous()
        return out


class ImageTestDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: list):
        super().__init__()
        self.input_hw = input_hw

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "test":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TE_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TE_MASK_DIR, unique_id + mask_suffix)
            self.total_data_paths.append((class_name, image_path, mask_path))
        logger.info(f"[TestSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

    def __getitem__(self, index):
        class_name, image_path, mask_path = self.total_data_paths[index]

        image = io.read_color_array(image_path)

        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)
        return dict(data={"image": image}, info=dict(text=class_name, mask_path=mask_path, group_name="image"))

    def __len__(self):
        return len(self.total_data_paths)


class ImageTrainDataset(data.Dataset):
    def __init__(self, dataset_info: Config, input_hw: dict):
        super().__init__()
        self.input_hw = input_hw

        with open(dataset_info.OVCamo_CLASS_JSON_PATH, mode="r", encoding="utf-8") as f:
            class_infos = json.load(f)
        with open(dataset_info.OVCamo_SAMPLE_JSON_PATH, mode="r", encoding="utf-8") as f:
            sample_infos = json.load(f)

        self.classes = []
        for class_info in class_infos:
            if class_info["split"] == "train":
                self.classes.append(class_info["name"])

        self.total_data_paths = []
        for sample_info in sample_infos:
            class_name = sample_info["base_class"]
            if class_name not in self.classes:
                continue

            unique_id = sample_info["unique_id"]
            image_suffix = os.path.splitext(sample_info["image"])[1]
            mask_suffix = os.path.splitext(sample_info["mask"])[1]
            image_path = os.path.join(dataset_info.OVCamo_TR_IMAGE_DIR, unique_id + image_suffix)
            mask_path = os.path.join(dataset_info.OVCamo_TR_MASK_DIR, unique_id + mask_suffix)
            # depth_path = os.path.join(dataset_info.OVCamo_TR_DEPTH_DIR, unique_id + mask_suffix)
            # self.total_data_paths.append((class_name, image_path, mask_path, depth_path))
            self.total_data_paths.append((class_name, image_path, mask_path))
        logger.info(f"[TrainSet] {len(self.total_data_paths)} Samples, {len(self.classes)} Classes")

        self.trains = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
            ],
            additional_targets={"depth": "mask"},
        )

    def __getitem__(self, index):
        # class_name, image_path, mask_path, depth_path = self.total_data_paths[index]
        class_name, image_path, mask_path = self.total_data_paths[index]

        image = io.read_color_array(image_path)
        mask = io.read_gray_array(mask_path, thr=0)
        mask = (mask * 255).astype(np.uint8)
        # depth = io.read_gray_array(depth_path, to_normalize=True)
        # depth = (depth * 255).astype(np.uint8)
        if image.shape[:2] != mask.shape:
            h, w = mask.shape
            image = ops.resize(image, height=h, width=w)
            # depth = ops.resize(depth, height=h, width=w)

        image = ops.resize(image, height=self.input_hw[0], width=self.input_hw[1])
        mask = ops.resize(mask, height=self.input_hw[0], width=self.input_hw[1])
        # depth = ops.resize(depth, height=self.input_hw[0], width=self.input_hw[1])
        # assert all([x.dtype == np.uint8 for x in [image, mask, depth]])
        assert all([x.dtype == np.uint8 for x in [image, mask]])

        # transformed = self.trains(image=image, mask=mask, depth=depth)
        transformed = self.trains(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        # depth = transformed["depth"]
        image = torch.from_numpy(image).div(255).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).gt(0).float().unsqueeze(0)
        # depth = torch.from_numpy(depth).div(255).float().unsqueeze(0)
        # return dict(data={"image": image, "mask": mask, "depth": depth}, info={"text": class_name})
        return dict(data={"image": image, "mask": mask,}, info={"text": class_name})

    def __len__(self):
        return len(self.total_data_paths)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def _load_torch_checkpoint(ckpt_path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def _extract_prefixed_state_dict(raw_state, prefix):
    state = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _load_compatible_state_dict(module, state_dict, module_name):
    module_state = module.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in module_state and module_state[key].shape == value.shape
    }
    skipped_keys = sorted(set(state_dict) - set(compatible_state))
    missing_keys, unexpected_keys = module.load_state_dict(compatible_state, strict=False)
    logger.info(
        f"Loaded {len(compatible_state)}/{len(module_state)} compatible {module_name} tensors; "
        f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, skipped={len(skipped_keys)}"
    )
    if skipped_keys:
        logger.info(f"Skipped {module_name} checkpoint tensors: {skipped_keys}")


def _prepare_sam2_modeling_imports(sam2_root):
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    if "sam2" not in sys.modules:
        import types

        sam2_pkg = types.ModuleType("sam2")
        sam2_pkg.__path__ = [os.path.join(sam2_root, "sam2")]
        sam2_pkg.__package__ = "sam2"
        sys.modules["sam2"] = sam2_pkg


class SAM2MaskDecoderAdapter(nn.Module):
    def __init__(self, checkpoint_path, sam2_root):
        super().__init__()
        _prepare_sam2_modeling_imports(sam2_root)

        from sam2.modeling.sam.mask_decoder import MaskDecoder as SAM2MaskDecoder
        from sam2.modeling.sam.transformer import TwoWayTransformer as SAM2TwoWayTransformer

        self.uses_high_res_features = True
        self.decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=SAM2TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
        )

        raw_state = _load_torch_checkpoint(checkpoint_path)
        decoder_state = _extract_prefixed_state_dict(raw_state, "sam_mask_decoder.")
        if not decoder_state:
            raise RuntimeError(f"No sam_mask_decoder.* tensors found in {checkpoint_path}")
        _load_compatible_state_dict(self.decoder, decoder_state, "SAM2 mask decoder")

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        high_res_features=None,
    ):
        image_batch = image_embeddings.shape[0]
        prompt_batch = sparse_prompt_embeddings.shape[0]
        if prompt_batch == 1 and image_batch != 1:
            sparse_prompt_embeddings = sparse_prompt_embeddings.expand(image_batch, -1, -1)
            dense_prompt_embeddings = dense_prompt_embeddings.expand(image_batch, -1, -1, -1)
        elif prompt_batch != image_batch:
            raise ValueError(
                f"SAM2 decoder expects prompt batch to match image batch; "
                f"got prompt_batch={prompt_batch}, image_batch={image_batch}"
            )

        if high_res_features is None:
            raise ValueError("SAM2 high-res decoder requires high_res_features=[feat_s0, feat_s1].")

        masks, iou_pred, _, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return masks, iou_pred


def load_sam2_prompt_encoder_weights(prompt_encoder, checkpoint_path):
    raw_state = _load_torch_checkpoint(checkpoint_path)
    prompt_state = _extract_prefixed_state_dict(raw_state, "sam_prompt_encoder.")
    if not prompt_state:
        raise RuntimeError(f"No sam_prompt_encoder.* tensors found in {checkpoint_path}")
    _load_compatible_state_dict(prompt_encoder, prompt_state, "SAM2 prompt encoder")


class OVSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        clip_backbone="convnext-l",
        clip_pretrained=None,
        clip_checkpoint=None,
        clip_cache_root=None,
        target_background_hypergraph_cfg=None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        self.train_class_embs = None
        self.bg_class_embs = None
        self.cached_class_names = None
        self.template_set = "camoprompts"
        self.clip = CLIPBackbone(
            backbone_name=clip_backbone,
            pretrained=clip_pretrained,
            local_model_path=clip_checkpoint,
            cache_root=clip_cache_root,
            template_set=self.template_set,
        )
        self.normalizer = PixelNormalizer(mean=self.clip.mean, std=self.clip.std)

        self.linear_layer = nn.Linear(self.clip.dim_latent, 2*256)
        self.guide_norm = nn.LayerNorm(self.clip.dim_latent)

        c2, c3, c4, c5 = self.clip.feature_channels
        self.in_channels = c5
        self.decoder_channel_list = [c4, c3, c2]
        self.skip_channel_down_list = [64, 64, 64, 64]
        self.scale_branches = 2
        self.frequency_branches = 16
        self.frequency_selection = 'top'
        self.block_repetition = 1
        self.min_channel = 64
        self.min_resolution = 8

        self.decoder_stage1 = UpsampleBlock(self.in_channels, self.decoder_channel_list[0], self.skip_channel_down_list[0], self.scale_branches, self.frequency_branches, self.frequency_selection, self.block_repetition, self.min_channel, self.min_resolution)
        self.decoder_stage2 = UpsampleBlock(self.decoder_channel_list[0], self.decoder_channel_list[1], self.skip_channel_down_list[1], self.scale_branches, self.frequency_branches, self.frequency_selection, self.block_repetition, self.min_channel, self.min_resolution)
        self.decoder_stage3 = UpsampleBlock(self.decoder_channel_list[1], self.decoder_channel_list[2], self.skip_channel_down_list[2], self.scale_branches, self.frequency_branches, self.frequency_selection, self.block_repetition, self.min_channel, self.min_resolution)

        self.skip_channel_list = [c4, c3, c2]

        self.skip_connection1 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[0], self.skip_channel_down_list[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[0]), nn.ReLU(inplace=True))
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[1], self.skip_channel_down_list[1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[1]), nn.ReLU(inplace=True))
        self.skip_connection3 = nn.Sequential(
            nn.Conv2d(self.skip_channel_list[2], self.skip_channel_down_list[2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.skip_channel_down_list[2]), nn.ReLU(inplace=True))

        self.trans_imgbedding = nn.Sequential(
            nn.Conv2d(c2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True)
        )
        self.sam2_feat_s1 = nn.Sequential(
            nn.Conv2d(c3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True),
        )
        self.sam2_feat_s0 = nn.Sequential(
            nn.Conv2d(c2, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True),
        )

        # self.classify_using_cosine_similarity = classify_using_cosine_similarity()
        # self.trans_x3_to_class = nn.Sequential(
        #     nn.Conv2d(192,1,1),
        # )

        # self.BidirectionalCrossAttention = BidirectionalCrossAttention()
        attention_hidden = min(768, self.clip.dim_latent)
        self.crossattforlastlayer = CrossAttnEnhancer(
            vis_ch=self.in_channels,
            txt_ch=self.clip.dim_latent,
            hidden=attention_hidden,
            tri_hidden=min(512, attention_hidden),
        )
        target_background_hypergraph_cfg = dict(target_background_hypergraph_cfg or {})
        target_background_hypergraph_cfg.setdefault("hidden_dim", attention_hidden)
        self.target_background_hypergraph = TargetBackgroundHypergraphModule(
            vis_dim=self.in_channels,
            text_dim=self.clip.dim_latent,
            **target_background_hypergraph_cfg,
        )

    def get_background_text_embs(self, class_names):
        bg_prompts = [
            f"A photo of the background surrounding the camouflaged {class_name}."
            for class_name in class_names
        ]
        return self.clip.get_text_embs(bg_prompts)

    def map_classifier(self, logits, image_deep, normed_class_embs):
        prob = logits.sigmoid()
        image_embs = resize_to(image_deep, tgt_hw=prob.shape[-2:])

        # image_embs (B,C)
        image_embs = (prob * image_embs).sum((-1, -2)) / prob.sum((-1, -2))
        image_embs = image_embs[..., None, None]#(B,1536,1,1)

        # B,C => B,D
        normed_image_embs = self.clip.visual_feats_to_embs(image_embs, normalize=True)
        class_logits = normed_image_embs @ normed_class_embs.T  # B,N
        class_logits = self.clip.clip_model.logit_scale.exp() * class_logits
        return class_logits

    def forward(self, image, class_names: list, return_intermediates=False):
        # [N=num_classes, 768]
        class_names_key = tuple(class_names)
        if self.train_class_embs is None or self.cached_class_names != class_names_key:#here
            self.train_class_embs = self.clip.get_text_embs_by_template(class_names)#frozen
            self.bg_class_embs = self.get_background_text_embs(class_names)
            self.cached_class_names = class_names_key
        normed_class_embs = self.train_class_embs  # Nc,D train（14，768） test(61,768)
        normed_class_embs_used_to_class = normed_class_embs
        normed_class_embs = self.guide_norm(normed_class_embs)#train（14，768） test(61,768)
        normed_bg_embs = self.guide_norm(self.bg_class_embs)


        image = self.normalizer(image)
        image_embedding = self.clip.get_visual_feats(image)  # frozen clip-convnext input_size=384*384
        image_deep = image_embedding['clip_vis_dense']  # used to class
        """
        此时得到的image_embedding形状是：
        dict:6
        0:image_embedding[stem]:(B,192,96,96)-初步提取-conv+ln
        1:image_embedding[res2]:(B,192,96,96)
        2:image_embedding[res3]:(B,384,48,48)
        3:image_embedding[res4]:(B,768,24,24)
        4:image_embedding[res5]:(B,1536,12,12)
        5:image_embedding[clip_vis_dense]:(B,1536,12,12)-identity-与res一样
        因此，进行多及视觉特征融合时，用res2、res3、res4、res5
        输入到sam掩码解码器的形状必须是（B，256,64,64）在这之前进行多级视觉特征融合，然后将形状转换为（B，256,64,64）
        """

        res2, res3, res4, res5 = image_embedding['res2'], image_embedding['res3'], image_embedding['res4'], image_embedding['res5'],

        # vis = [res2, res3, res4, res5]
        # txt = normed_class_embs
        # [res2, res3, res4, res5], normed_class_embs = self.BidirectionalCrossAttention(vis, txt)
        original_res5 = res5
        if return_intermediates:
            res5, normed_class_embs, normed_bg_embs, cross_debug = self.crossattforlastlayer(
                res5, normed_class_embs, normed_bg_embs, return_intermediates=True
            )
            res5, normed_class_embs, hypergraph_debug = self.target_background_hypergraph(
                res5, normed_class_embs, normed_bg_embs, return_intermediates=True
            )
            hypergraph_output = res5
        else:
            res5, normed_class_embs, normed_bg_embs = self.crossattforlastlayer(
                res5,
                normed_class_embs,
                normed_bg_embs,
            )
            res5, normed_class_embs = self.target_background_hypergraph(res5, normed_class_embs, normed_bg_embs)

        normed_class_embs = self.linear_layer(normed_class_embs)#train（14，512） test(61,512)
        if normed_class_embs.dim() == 3:
            normed_class_embs = (normed_class_embs.softmax(dim=1) * normed_class_embs).sum(dim=1)
            normed_class_embs = normed_class_embs.view(normed_class_embs.shape[0], 2, 256)
        else:
            normed_class_embs = (normed_class_embs.softmax(dim=0)*normed_class_embs).sum(dim=0).unsqueeze(0)#(1,512)
            normed_class_embs = normed_class_embs.view(1, 2, 256)#(1,2,256)

        x1 = self.decoder_stage1(res5, self.skip_connection1(res4))#(B,768,24,24)
        x2 = self.decoder_stage2(x1, self.skip_connection2(res3))#(B,384,48,48)
        x3 = self.decoder_stage3(x2, self.skip_connection3(res2))#(B,192,96,96)
        # x3_to_class = self.trans_x3_to_class(x3)

        #前面视觉特征融合完毕后，在这里转换成需要的形状：(B，256,64,64)
        image_embedding = self.trans_imgbedding(x3)
        high_res_features = None
        if getattr(self.mask_decoder, "uses_high_res_features", False):
            high_res_features = [
                self.sam2_feat_s0(x3),  # (B, 32, 256, 256)
                self.sam2_feat_s1(x2),  # (B, 64, 128, 128)
            ]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            texts=normed_class_embs,
            masks=None,
        )
        mask_decoder_kwargs = dict(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        if high_res_features is not None:
            mask_decoder_kwargs["high_res_features"] = high_res_features
        low_res_masks, _ = self.mask_decoder(**mask_decoder_kwargs)#(B,1,256,256)
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        cls_logits = self.map_classifier(low_res_masks, image_deep, normed_class_embs_used_to_class)
        if return_intermediates:
            return ori_res_masks, cls_logits, {
                "original_feature": original_res5,
                "cross_attention": cross_debug["cross_attention"],
                "tri_attention": cross_debug["tri_attention"],
                "high_pass": hypergraph_debug["high_pass"],
                "low_pass": hypergraph_debug["low_pass"],
                "hypergraph_output": hypergraph_output,
                "decoder_input": image_embedding,
                "prediction_logits": ori_res_masks,
            }
        return ori_res_masks, cls_logits

    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("clip.clip_model."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            else:
                param.requires_grad = True
                param_groups["retrained"].append(param)
        logger.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


@torch.no_grad()
def evaluate_model(model, cfg, metric_names=("sm", "wfm", "mae", "fm", "em", "iou"), save_results=None, desc="[EVAL]"):
    te_dataset = ImageTestDataset(dataset_info=cfg.root_info, input_hw=cfg.test.input_hw)
    te_loader = data.DataLoader(te_dataset, cfg.test.batch_size, num_workers=cfg.test.num_workers, pin_memory=True)

    if save_results is None:
        save_results = cfg.test.save_results

    if save_results:
        save_path = cfg.path.save
        logger.info(f"Results will be saved into {save_path}")
    else:
        save_path = ""

    was_training = model.training
    model.eval()
    dataset_classes = te_loader.dataset.classes
    metricer = recorder.OVCOSMetricer(class_names=dataset_classes, metric_names=metric_names)
    for batch in tqdm(te_loader, total=len(te_loader), ncols=79, desc=desc):
        batch_images = pt_utils.to_device(batch["data"], device=cfg.device)  # B,1,H,W
        gt_classes = batch["info"]["text"]
        # outputs = model(data=batch_images, gt_classes=gt_classes, class_names=dataset_classes)
        image = batch_images['image']
        outputs = model(image=image, class_names=dataset_classes)

        # probs = outputs["prob"].squeeze(1).cpu().detach().numpy()  # B,H,W
        probs = outputs[0].sigmoid().squeeze(1).cpu().detach().numpy()
        probs = (probs > 0.5).astype(np.uint8)
        pred_class_ids = torch.argmax(outputs[1], dim=-1).cpu().tolist()
        mask_paths = batch["info"]["mask_path"]
        for idx_in_batch, pred in enumerate(probs):
            mask_path = Path(mask_paths[idx_in_batch])
            mask = io.read_gray_array(mask_path.as_posix(), thr=0)
            mask = (mask * 255).astype(np.uint8)
            mask_h, mask_w = mask.shape

            pred = ops.minmax(pred)
            pred = ops.resize(pred, height=mask_h, width=mask_w)

            pre_cls = dataset_classes[pred_class_ids[idx_in_batch]]
            gt_cls = gt_classes[idx_in_batch]

            if save_path:
                ops.save_array_as_image(pred, save_name=f"[{pre_cls}]{mask_path.name}", save_dir=save_path)
            metricer.step(
                pre=(pred * 255).astype(np.uint8),
                gt=mask,
                pre_cls=pre_cls,
                gt_cls=gt_cls,
                gt_path=mask_path.as_posix(),
            )
    avg_ovcos_results = metricer.show()
    if was_training:
        model.train()
    return avg_ovcos_results


@torch.no_grad()
def test(model, cfg, metric_names=("sm", "wfm", "mae", "fm", "em", "iou")):
    avg_ovcos_results = evaluate_model(
        model=model,
        cfg=cfg,
        metric_names=metric_names,
        save_results=cfg.test.save_results,
        desc="[EVAL]",
    )
    logger.info(str(avg_ovcos_results))


seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
# cross entropy loss
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")


def append_eval_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["epoch"] + [key for key in row.keys() if key != "epoch"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(model, cfg):
    tr_dataset = ImageTrainDataset(dataset_info=cfg.root_info, input_hw=cfg.train.input_hw)
    tr_loader = data.DataLoader(
        dataset=tr_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=pt_utils.customized_worker_init_fn if cfg.use_custom_worker_init else None,
    )

    counter = recorder.TrainingCounter(
        epoch_length=len(tr_loader),
        epoch_based=cfg.train.epoch_based,
        num_epochs=cfg.train.num_epochs,
        num_total_iters=cfg.train.num_iters,
    )
    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )
    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=cfg.train.scheduler,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)
    scheduler.plot_lr_coef_curve(save_path=cfg.path.pth_log)
    logger.info(
        f"Trainable Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if v.requires_grad))}"
    )
    logger.info(
        f"Fixed Parameters: {sum((v.numel() for v in model.parameters(recurse=True) if not v.requires_grad))}"
    )

    scaler = pipeline.Scaler(optimizer=optimizer)
    logger.info(f"Scheduler:\n{scheduler}\nOptimizer:\n{optimizer}")

    loss_recorder = recorder.HistoryBuffer()
    iter_time_recorder = recorder.HistoryBuffer()
    raw_model = unwrap_model(model)
    logger.info(f"Image Mean: {raw_model.normalizer.mean.flatten()}, Image Std: {raw_model.normalizer.std.flatten()}")
    best_eval_sm = -float("inf")

    train_start_time = time.perf_counter()
    for curr_epoch in range(counter.num_epochs):
        logger.info(f"Exp_Name: {cfg.exp_name}")
        device = cfg.device
        model.to(device)
        model.train()
        # an epoch starts
        for batch_idx, batch in enumerate(tr_loader):
            iter_start_time = time.perf_counter()
            scheduler.step(curr_idx=counter.curr_iter)  # update learning rate

            data_batch = pt_utils.to_device(data=batch["data"], device=device)
            gt_classes = batch["info"]["text"]
            outputs = model(
                image=data_batch['image'],
                # gt_classes=gt_classes,
                class_names=tr_dataset.classes,
            )
            mask = data_batch['mask']
            loss = seg_loss(outputs[0], mask) + ce_loss(outputs[0], mask.float())
            loss = loss / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)
            if counter.every_n_iters(cfg.train.grad_acc_step):  # Accumulates scaled gradients.
                scaler.update_grad()

            item_loss = loss.item()
            data_shape = tuple(data_batch["mask"].shape)
            loss_recorder.update(value=item_loss, num=data_shape[0])

            if cfg.log_interval > 0 and (
                counter.every_n_iters(cfg.log_interval)
                or counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.is_last_total_iter()
            ):
                gpu_mem = f"{torch.cuda.max_memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                eta_seconds = iter_time_recorder.avg * (counter.num_total_iters - counter.curr_iter - 1)
                eta_string = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
                progress = (
                    f"{counter.curr_iter}:{counter.num_total_iters} "
                    f"{batch_idx}/{counter.num_inner_iters} "
                    f"{counter.curr_epoch}/{counter.num_epochs}"
                )
                loss_info = f"(M:{loss_recorder.global_avg:.5f}/C:{item_loss:.5f})"
                lr_info = f"LR: {optimizer.lr_string()}"
                logger.info(f"{eta_string}({gpu_mem}) | {progress} | {lr_info} | {loss_info} | {data_shape}")

            # if counter.curr_iter < 3:  # plot some batches of the training phase
            #     recorder.plot_results(
            #         # dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], **outputs["vis"]),
            #         dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"], ),
            #         save_path=os.path.join(cfg.path.pth_log, "img", f"iter_{counter.curr_iter}.jpg"),
            #     )

            iter_time_recorder.update(value=time.perf_counter() - iter_start_time)
            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        # if curr_epoch < 3:
        #     recorder.plot_results(
        #         dict(img=data_batch["image"], msk=data_batch["mask"], dep=data_batch["depth"],),
        #         save_path=os.path.join(cfg.path.pth_log, "img", f"epoch_{curr_epoch}.jpg"),
        #     )

        completed_epoch = curr_epoch + 1
        if cfg.train_eval and completed_epoch >= cfg.train_eval_start_epoch:
            logger.info(f"Start evaluating epoch {completed_epoch}...")
            eval_results = evaluate_model(
                model=model,
                cfg=cfg,
                save_results=False,
                desc=f"[EVAL E{completed_epoch}]",
            )
            eval_row = {"epoch": completed_epoch, **eval_results}
            append_eval_csv(cfg.train_eval_csv, eval_row)
            logger.info(f"Epoch {completed_epoch} Eval: {eval_results}")

            curr_sm = float(eval_results["sm"])
            if curr_sm > best_eval_sm:
                best_eval_sm = curr_sm
                os.makedirs(os.path.dirname(cfg.train_eval_best_weight), exist_ok=True)
                io.save_weight(model=model, save_path=cfg.train_eval_best_weight)
                logger.info(
                    f"Best sm updated to {best_eval_sm:.5f} at epoch {completed_epoch}; "
                    f"saved to {cfg.train_eval_best_weight}"
                )

        counter.update_epoch_counter()
        # an epoch ends

    io.save_weight(model=model, save_path=cfg.path.final_state_net, suffix="-final")

    total_train_time = time.perf_counter() - train_start_time
    total_other_time = datetime.timedelta(seconds=int(total_train_time - iter_time_recorder.global_sum))
    logger.info(f"Total Time: {datetime.timedelta(seconds=int(total_train_time))} ({total_other_time} on others)")


def parse_cfg():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default='configs/ovcoser.py', type=str)
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml", type=str)
    parser.add_argument("--model-name", default='OVCoser', type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--info", type=str)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument(
        "--clip-backbone",
        type=str,
        default="convnext-l",
        choices=sorted(CLIP_BACKBONE_CONFIGS),
        help="OpenCLIP visual/text backbone.",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=None,
        help="Override the default OpenCLIP pretrained tag for the selected backbone.",
    )
    parser.add_argument("--clip-checkpoint", type=str, default=None)
    parser.add_argument(
        "--clip-cache-root",
        type=str,
        default="pretrained",
        help="Root directory used to find/cache automatically downloaded CLIP weights.",
    )
    parser.add_argument("--use-sam2-decoder", type=str2bool, default=True)
    parser.add_argument("--sam2-checkpoint", type=str, default="pretrained/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-root", type=str, default="sam2-main")
    parser.add_argument("--load-sam2-prompt-encoder", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default=None, help="Primary device, e.g. cuda:0 or cpu.")
    parser.add_argument("--devices", type=str, default=None, help="CUDA device ids for DataParallel, e.g. 0,1 or all.")
    parser.add_argument("--train-eval", type=str2bool, default=False, help="Evaluate after each epoch during training.")
    parser.add_argument(
        "--train-eval-start-epoch",
        type=int,
        default=1,
        help="1-based epoch index to start training-time evaluation.",
    )
    parser.add_argument("--train-eval-csv", type=str, default=None, help="CSV path for training-time eval history.")
    parser.add_argument(
        "--train-eval-best-weight",
        type=str,
        default=None,
        help="Path to overwrite with the best-sm weight during training-time evaluation.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.root_info, mode="r") as f:
        cfg.root_info = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    if cfg.checkpoint and not os.path.isabs(cfg.checkpoint):
        cfg.checkpoint = os.path.join(cfg.proj_root, cfg.checkpoint)
    if cfg.clip_checkpoint and not os.path.isabs(cfg.clip_checkpoint):
        cfg.clip_checkpoint = os.path.join(cfg.proj_root, cfg.clip_checkpoint)
    if cfg.clip_cache_root and not os.path.isabs(cfg.clip_cache_root):
        cfg.clip_cache_root = os.path.join(cfg.proj_root, cfg.clip_cache_root)
    if cfg.sam2_checkpoint and not os.path.isabs(cfg.sam2_checkpoint):
        cfg.sam2_checkpoint = os.path.join(cfg.proj_root, cfg.sam2_checkpoint)
    if cfg.sam2_root and not os.path.isabs(cfg.sam2_root):
        cfg.sam2_root = os.path.join(cfg.proj_root, cfg.sam2_root)
    cfg.exp_name = py_utils.construct_exp_name(
        model_name=f"{cfg.model_name}-{cfg.clip_backbone}",
        cfg=cfg,
    )
    cfg.output_dir = os.path.join(cfg.proj_root, "outputs")
    cfg.path = py_utils.construct_path(output_dir=cfg.output_dir, exp_name=cfg.exp_name)
    cfg.train_eval_start_epoch = max(1, cfg.train_eval_start_epoch)
    if cfg.train_eval_csv is None:
        cfg.train_eval_csv = os.path.join(cfg.path.pth_log, "train_eval.csv")
    elif not os.path.isabs(cfg.train_eval_csv):
        cfg.train_eval_csv = os.path.join(cfg.proj_root, cfg.train_eval_csv)
    if cfg.train_eval_best_weight is None:
        cfg.train_eval_best_weight = os.path.join(cfg.path.pth, "best_sm.pth")
    elif not os.path.isabs(cfg.train_eval_best_weight):
        cfg.train_eval_best_weight = os.path.join(cfg.proj_root, cfg.train_eval_best_weight)
    cfg.device_ids = parse_device_ids(cfg.devices)
    if cfg.device_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("--devices requires CUDA, but torch.cuda.is_available() is False.")
        max_device_id = max(cfg.device_ids)
        if max_device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Invalid --devices {cfg.devices}; only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        cfg.device = f"cuda:{cfg.device_ids[0]}"
    elif cfg.device is None:
        cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    logger.add(cfg.path.log, level="INFO", format=logger_format)
    logger.info(cfg.pretty_text)
    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
    sam_model = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)

    mask_decoder = sam_model.mask_decoder
    if cfg.use_sam2_decoder:
        if not os.path.exists(cfg.sam2_checkpoint):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {cfg.sam2_checkpoint}. "
                "Put sam2.1_hiera_base_plus.pt under pretrained/ or pass --sam2-checkpoint."
            )
        if not os.path.isdir(cfg.sam2_root):
            raise FileNotFoundError(f"SAM2 root not found: {cfg.sam2_root}")
        mask_decoder = SAM2MaskDecoderAdapter(
            checkpoint_path=cfg.sam2_checkpoint,
            sam2_root=cfg.sam2_root,
        )
        logger.info(f"Using SAM2.1 Hiera-B+ high-res mask decoder from {cfg.sam2_checkpoint}")
        if cfg.load_sam2_prompt_encoder:
            load_sam2_prompt_encoder_weights(sam_model.prompt_encoder, cfg.sam2_checkpoint)

    model_class = model_zoo.__dict__.get(cfg.model_name)
    assert model_class is not None, "Please check your --model-name"
    model_code = inspect.getsource(OVSAM)

    model = OVSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_backbone=cfg.clip_backbone,
        clip_pretrained=cfg.clip_pretrained,
        clip_checkpoint=cfg.clip_checkpoint,
        clip_cache_root=cfg.clip_cache_root,
        target_background_hypergraph_cfg=getattr(cfg, "target_background_hypergraph", {}),
    )
    logger.info(model_code)
    model.to(cfg.device)
    torch.set_float32_matmul_precision("high")

    if cfg.load_from:
        io.load_weight(model=model, load_path=cfg.load_from, strict=True)

    if len(cfg.device_ids) > 1:
        logger.info(f"Using DataParallel on CUDA devices: {cfg.device_ids}")
        if cfg.train.batch_size < len(cfg.device_ids):
            logger.warning(
                f"Batch size ({cfg.train.batch_size}) is smaller than the number of GPUs ({len(cfg.device_ids)})."
            )
        model = nn.DataParallel(model, device_ids=cfg.device_ids, output_device=cfg.device_ids[0])

    if not cfg.evaluate:
        train(model=model, cfg=cfg)
    else:
        cfg.test.save_results = True

    if cfg.evaluate or cfg.has_test:
        test(model=model, cfg=cfg)

    logger.info("End training...")


if __name__ == "__main__":
    main()
