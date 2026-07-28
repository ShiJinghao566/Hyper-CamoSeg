#!/usr/bin/env python3
"""Fine-tune OVCoser from a trained weight on a small, explicit sample list."""

import argparse
import json
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from mmengine import Config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from main import (
    OVSAM,
    SAM2MaskDecoderAdapter,
    ce_loss,
    load_sam2_prompt_encoder_weights,
    seg_loss,
)
from segment_anything import sam_model_registry
from utils import io, ops, pipeline, pt_utils


DEFAULT_SAMPLES = (
    "train:seahorse:004668_seahorse",
    "train:lizard:002628_lizard",
    "test:deer:005108_deer",
    "test:turtle:002459_turtle",
    "test:ant:009020_ant",
    "test:batfish:003515_batfish",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-weight", required=True)
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml")
    parser.add_argument("--config", default="configs/ovcoser.py")
    parser.add_argument("--output-dir", default="outputs/selected_samples_finetune")
    parser.add_argument(
        "--samples", nargs="+", default=DEFAULT_SAMPLES,
        help="Entries formatted as split:class_name:unique_id.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=112358)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--sam-checkpoint", default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-checkpoint", default="pretrained/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-root", default="sam2-main")
    parser.add_argument("--clip-checkpoint", default=None)
    return parser.parse_args()


def resolve(project_root, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def read_hypergraph_config(path):
    return dict(Config.fromfile(path).get("target_background_hypergraph", {}))


def parse_sample_specs(specs):
    selected = []
    seen = set()
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3 or parts[0] not in ("train", "test"):
            raise ValueError(f"Invalid sample '{spec}'; expected split:class_name:unique_id")
        item = tuple(parts)
        if item in seen:
            raise ValueError(f"Duplicate sample: {spec}")
        seen.add(item)
        selected.append(item)
    if not selected:
        raise ValueError("At least one --samples entry is required.")
    return selected


class SelectedSamplesDataset(Dataset):
    def __init__(self, root_info_path, project_root, specs, input_hw):
        with open(root_info_path, "r", encoding="utf-8") as f:
            root_info = yaml.safe_load(f)

        def data_path(key):
            return resolve(project_root, root_info[key])

        with open(data_path("OVCamo_SAMPLE_JSON_PATH"), "r", encoding="utf-8") as f:
            sample_infos = json.load(f)
        by_id = {item["unique_id"]: item for item in sample_infos}

        self.items = []
        self.classes = []
        for split, class_name, unique_id in parse_sample_specs(specs):
            # visualize_intermediate_binary_mask.py names output folders as
            # ``{selection_index:02d}_{unique_id}``. Accept that folder name too.
            requested_id = unique_id
            if unique_id not in by_id:
                prefix, separator, candidate = unique_id.partition("_")
                if separator and len(prefix) == 2 and prefix.isdigit() and candidate in by_id:
                    unique_id = candidate
                    print(f"Resolved visualization folder id {requested_id} -> dataset unique_id {unique_id}")
            if unique_id not in by_id:
                raise KeyError(
                    f"unique_id not found in sample_info.json: {requested_id}. "
                    "Note: the leading two-digit visualization selection index is not part of unique_id."
                )
            sample = by_id[unique_id]
            if sample["base_class"] != class_name:
                raise ValueError(
                    f"Class mismatch for {unique_id}: requested {class_name}, JSON has {sample['base_class']}"
                )
            prefix = "OVCamo_TR" if split == "train" else "OVCamo_TE"
            image = os.path.join(data_path(f"{prefix}_IMAGE_DIR"), unique_id + Path(sample["image"]).suffix)
            mask = os.path.join(data_path(f"{prefix}_MASK_DIR"), unique_id + Path(sample["mask"]).suffix)
            if not os.path.isfile(image) or not os.path.isfile(mask):
                raise FileNotFoundError(f"Missing image or mask for {split}:{class_name}:{unique_id}: {image}, {mask}")
            self.items.append((class_name, unique_id, image, mask))
            if class_name not in self.classes:
                self.classes.append(class_name)

        self.input_hw = tuple(input_hw)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        class_name, unique_id, image_path, mask_path = self.items[index]
        image = io.read_color_array(image_path)
        mask = (io.read_gray_array(mask_path, thr=0) * 255).astype(np.uint8)
        if image.shape[:2] != mask.shape:
            image = ops.resize(image, height=mask.shape[0], width=mask.shape[1])
        height, width = self.input_hw
        image = ops.resize(image, height=height, width=width)
        mask = ops.resize(mask, height=height, width=width)
        transformed = self.transform(image=image, mask=mask)
        image = torch.from_numpy(transformed["image"]).div(255).float().permute(2, 0, 1)
        mask = torch.from_numpy(transformed["mask"]).gt(0).float().unsqueeze(0)
        return {"image": image, "mask": mask, "class_name": class_name, "unique_id": unique_id}


def build_model(args, project_root):
    sam_checkpoint = resolve(project_root, args.sam_checkpoint)
    sam2_checkpoint = resolve(project_root, args.sam2_checkpoint)
    sam2_root = resolve(project_root, args.sam2_root)
    clip_checkpoint = resolve(project_root, args.clip_checkpoint)
    for label, path in (("SAM", sam_checkpoint), ("SAM2", sam2_checkpoint), ("trained weight", args.trained_weight)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    if not os.path.isdir(sam2_root):
        raise FileNotFoundError(f"SAM2 root not found: {sam2_root}")

    sam = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
    decoder = SAM2MaskDecoderAdapter(sam2_checkpoint, sam2_root)
    load_sam2_prompt_encoder_weights(sam.prompt_encoder, sam2_checkpoint)
    model = OVSAM(
        image_encoder=sam.image_encoder,
        mask_decoder=decoder,
        prompt_encoder=sam.prompt_encoder,
        clip_checkpoint=clip_checkpoint,
        target_background_hypergraph_cfg=read_hypergraph_config(resolve(project_root, args.config)),
    )
    io.load_weight(load_path=args.trained_weight, model=model, strict=True)
    return model.to(args.device)


def save_state(model, output_dir, name):
    path = os.path.join(output_dir, name)
    io.save_weight(model=model, save_path=path)
    return path


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    project_root = os.path.dirname(os.path.abspath(__file__))
    args.trained_weight = resolve(project_root, args.trained_weight)
    root_info = resolve(project_root, args.root_info)
    output_dir = resolve(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    pt_utils.initialize_seed_cudnn(seed=args.seed, deterministic=True)
    dataset = SelectedSamplesDataset(root_info, project_root, args.samples, args.input_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        worker_init_fn=pt_utils.customized_worker_init_fn,
    )
    print("Selected samples:")
    for class_name, unique_id, image, _ in dataset.items:
        print(f"  {class_name:10s} {unique_id}  <- {image}")

    model = build_model(args, project_root)
    optimizer = pipeline.construct_optimizer(
        model=model, initial_lr=args.lr, mode="adamw", group_mode="finetune",
        cfg={"weight_decay": args.weight_decay, "diff_factor": 0.1},
    )
    scaler = pipeline.Scaler(optimizer=optimizer, set_to_none=True)
    print(f"Fine-tuning {len(dataset)} samples, classes={dataset.classes}, batches/epoch={len(loader)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for batch in progress:
            images = batch["image"].to(args.device, non_blocking=True)
            masks = batch["mask"].to(args.device, non_blocking=True)
            mask_logits, _ = model(image=images, class_names=dataset.classes)
            loss = seg_loss(mask_logits, masks) + ce_loss(mask_logits, masks)
            scaler.calculate_grad(loss)
            scaler.update_grad()
            running_loss += loss.item() * images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.5f}")
        mean_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:03d}: mean_loss={mean_loss:.6f}")
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_state(model, output_dir, f"state_epoch_{epoch:03d}.pth")

    final_path = save_state(model, output_dir, "state_selected_finetuned_final.pth")
    print(f"Finished. Final weight: {final_path}")


if __name__ == "__main__":
    main()
