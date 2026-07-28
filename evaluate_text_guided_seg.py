#!/usr/bin/env python3
"""Evaluate BaCLIP/OVSAM as text-guided segmentation.

This script uses the ground-truth class name of each image as the only text
candidate, so the metrics measure mask quality only and do not include class
prediction penalties.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from mmengine import Config
from PIL import Image
from py_sod_metrics.sod_metrics import (
    MAE,
    Emeasure,
    Fmeasure,
    Smeasure,
    WeightedFmeasure,
)
from py_sod_metrics.utils import TYPE, get_adaptive_threshold, prepare_data
from torch.utils import data
from tqdm import tqdm

from main import ImageTestDataset, OVSAM
from segment_anything import sam_model_registry
from utils import io, ops, pt_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-guided BaCLIP segmentation evaluation without class penalty."
    )
    parser.add_argument("--config", default="configs/ovcoser.py", type=str)
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml", type=str)
    parser.add_argument("--load-from", required=True, type=str)
    parser.add_argument("--output-dir", default="outputs/OVCoser_text_guided_seg", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--model-type", default="vit_b", type=str)
    parser.add_argument("--checkpoint", default="pretrained/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument("--clip-checkpoint", default=None, type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--metrics-json", default=None, type=str)
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args()


def load_cfg(args):
    cfg = Config.fromfile(args.config)
    with open(args.root_info, mode="r", encoding="utf-8") as f:
        cfg.root_info = Config(yaml.safe_load(f))

    repo_root = Path(__file__).resolve().parent
    cfg.proj_root = str(repo_root)
    cfg.device = args.device
    cfg.model_type = args.model_type
    cfg.checkpoint = args.checkpoint
    cfg.clip_checkpoint = args.clip_checkpoint
    cfg.load_from = args.load_from
    if cfg.checkpoint and not os.path.isabs(cfg.checkpoint):
        cfg.checkpoint = str(repo_root / cfg.checkpoint)
    if cfg.clip_checkpoint and not os.path.isabs(cfg.clip_checkpoint):
        cfg.clip_checkpoint = str(repo_root / cfg.clip_checkpoint)
    if cfg.load_from and not os.path.isabs(cfg.load_from):
        cfg.load_from = str(repo_root / cfg.load_from)
    if args.num_workers is not None:
        cfg.test.num_workers = args.num_workers
    return cfg


def build_model(cfg):
    sam_model = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)
    model = OVSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_checkpoint=cfg.clip_checkpoint,
    )
    io.load_weight(model=model, load_path=cfg.load_from, strict=True)
    model.to(cfg.device)
    model.eval()
    return model


def resize_to_shape(array, shape):
    if array.shape == shape:
        return array
    return ops.resize(array, height=shape[0], width=shape[1])


def adaptive_iou(pred, gt):
    pred, gt = prepare_data(pred, gt)
    adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
    binary_pred = pred >= adaptive_threshold
    union = np.count_nonzero(np.bitwise_or(binary_pred, gt))
    if union == 0:
        return 0
    inter = np.count_nonzero(np.bitwise_and(binary_pred, gt))
    return inter / union


def changeable_iou(pred, gt):
    pred, gt = prepare_data(pred, gt)
    fg = np.count_nonzero(gt)
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    tp_hist, _ = np.histogram(pred[gt], bins=bins)
    fp_hist, _ = np.histogram(pred[~gt], bins=bins)
    tps = np.cumsum(np.flip(tp_hist))
    fps = np.cumsum(np.flip(fp_hist))
    fns = fg - tps
    curve = np.array(tps + fns + fps, dtype=TYPE)
    np.divide(tps, curve, out=curve, where=curve != 0)
    return curve


def summarize_metrics(metric_objs, adaptive_ious, changeable_ious):
    raw = {
        "sm": metric_objs["sm"].get_results()["sm"],
        "wfm": metric_objs["wfm"].get_results()["wfm"],
        "mae": metric_objs["mae"].get_results()["mae"],
        "fm": metric_objs["fm"].get_results()["fm"],
        "em": metric_objs["em"].get_results()["em"],
        "iou": {
            "adp": np.asarray(adaptive_ious, dtype=TYPE).reshape(-1, 1),
            "curve": np.stack(changeable_ious, axis=0),
        },
    }
    results = {
        "sm": raw["sm"].mean(),
        "wfm": raw["wfm"].mean(),
        "mae": raw["mae"].mean(),
        "adpfm": raw["fm"]["adp"].mean(),
        "maxfm": raw["fm"]["curve"].mean(axis=0).max(),
        "avgfm": raw["fm"]["curve"].mean(axis=0).mean(),
        "adpem": raw["em"]["adp"].mean(),
        "maxem": raw["em"]["curve"].mean(axis=0).max(),
        "avgem": raw["em"]["curve"].mean(axis=0).mean(),
        "adpiou": raw["iou"]["adp"].mean(),
        "maxiou": raw["iou"]["curve"].mean(axis=0).max(),
        "avgiou": raw["iou"]["curve"].mean(axis=0).mean(),
    }
    return {key: round(float(value), 3) for key, value in results.items()}


@torch.no_grad()
def evaluate(model, cfg, args):
    dataset = ImageTestDataset(dataset_info=cfg.root_info, input_hw=cfg.test.input_hw)
    if args.limit is not None:
        dataset.total_data_paths = dataset.total_data_paths[: args.limit]
    loader = data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        pin_memory=True,
    )

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pre"
    if args.save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_objs = {
        "sm": Smeasure(),
        "wfm": WeightedFmeasure(),
        "mae": MAE(),
        "fm": Fmeasure(),
        "em": Emeasure(),
    }
    adaptive_ious = []
    changeable_ious = []

    for batch in tqdm(loader, ncols=88, desc="BaCLIP text-guided eval"):
        batch_data = pt_utils.to_device(batch["data"], device=cfg.device)
        gt_class = batch["info"]["text"][0]
        mask_path = Path(batch["info"]["mask_path"][0])

        logits, _ = model(image=batch_data["image"], class_names=[gt_class])
        pred = logits.sigmoid().squeeze().detach().cpu().numpy()
        pred = (pred > args.threshold).astype(np.uint8)

        gt = io.read_gray_array(mask_path.as_posix(), thr=0)
        gt = (gt * 255).astype(np.uint8)
        pred = resize_to_shape(pred, gt.shape)
        pred = (pred > 0).astype(np.uint8) * 255

        for metric_obj in metric_objs.values():
            metric_obj.step(pred, gt)
        adaptive_ious.append(adaptive_iou(pred, gt))
        changeable_ious.append(changeable_iou(pred, gt))

        if args.save_preds:
            Image.fromarray(pred).save(pred_dir / f"[{gt_class}]{mask_path.name}")

    results = summarize_metrics(metric_objs, adaptive_ious, changeable_ious)
    metrics_path = Path(args.metrics_json) if args.metrics_json else output_dir / "metrics.json"
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Metrics saved to {metrics_path}")


def main():
    args = parse_args()
    cfg = load_cfg(args)
    pt_utils.initialize_seed_cudnn(seed=cfg.base_seed, deterministic=cfg.deterministic)
    model = build_model(cfg)
    torch.set_float32_matmul_precision("high")
    evaluate(model, cfg, args)


if __name__ == "__main__":
    main()
