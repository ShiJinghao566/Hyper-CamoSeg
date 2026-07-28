#!/usr/bin/env python3
"""Generate segmentation Grad-CAM maps for OVCamo decoder features."""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from segment_anything import sam_model_registry
from visualize_intermediates import (
    absolute,
    build_model,
    clean_prediction_mask,
    hard_mask_overlay,
    load_dataset_index,
    safe_name,
    titled_panel,
)


CAM_STAGES = [
    ("01_original_feature_gradcam", "original_feature", "Grad-CAM: original res5"),
    ("02_decoder_input_main_gradcam", "decoder_input", "Grad-CAM: decoder main input"),
    ("03_decoder_input_s1_gradcam", "decoder_input_s1", "Grad-CAM: decoder feat_s1"),
    ("04_decoder_input_s0_gradcam", "decoder_input_s0", "Grad-CAM: decoder feat_s0"),
    ("05_decoder_prompt_fused_gradcam", "decoder_prompt_fused", "Grad-CAM: main + dense prompt"),
    ("06_transformer_block1_gradcam", "transformer_block1", "Grad-CAM: transformer block 1"),
    ("07_transformer_block2_gradcam", "transformer_block2", "Grad-CAM: transformer block 2"),
    ("08_transformer_output_gradcam", "transformer_output", "Grad-CAM: transformer output"),
    ("09_upscale_fusion_s1_gradcam", "upscale_fusion_s1", "Grad-CAM: upscale + feat_s1"),
    ("10_upscale_fusion_s0_gradcam", "upscale_fusion_s0", "Grad-CAM: upscale + feat_s0"),
]

# Match every non-prediction Grad-CAM stage to the visualized
# Upscale+feat_s1 resolution before computing channel weights and heatmaps.
FEATURE_VIS_HW = (256, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-weight", required=True)
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml")
    parser.add_argument("--output-dir", default="outputs/gradcam_visualizations")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
    )
    parser.add_argument("--config", default="configs/ovcoser.py")
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--input-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--sam-checkpoint", default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-checkpoint", default="pretrained/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-root", default="sam2-main")
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--cam-target", choices=("pred", "gt"), default="pred")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--mask-smooth-kernel", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--save-npy", action="store_true")
    return parser.parse_args()


class DecoderGradCAMCapture:
    """Keep differentiable activations for the ten requested visual stages."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.kinds = {}
        self.handles = []

    def _record_tensor(self, name, tensor, kind="map"):
        if tensor.requires_grad:
            tensor.retain_grad()
        self.activations[name] = tensor
        self.kinds[name] = kind

    def _original_pre_hook(self, _module, inputs):
        # CLIP's get_visual_feats() is decorated with no_grad. Reintroduce res5 as
        # a differentiable leaf while keeping its value and all model weights fixed.
        visual = inputs[0].detach().requires_grad_(True)
        self._record_tensor("original_feature", visual)
        return (visual, *inputs[1:])

    def _output_hook(self, name, index=None, kind="map"):
        def hook(_module, _inputs, output):
            tensor = output[index] if index is not None else output
            self._record_tensor(name, tensor, kind=kind)
        return hook

    def _pre_input_hook(self, name):
        def hook(_module, inputs):
            self._record_tensor(name, inputs[0])
        return hook

    def __enter__(self):
        cross = self.model.crossattforlastlayer
        adapter = self.model.mask_decoder
        decoder = getattr(adapter, "decoder", adapter)
        transformer = decoder.transformer
        if len(transformer.layers) < 2:
            raise RuntimeError("The Grad-CAM script expects a two-layer SAM2 TwoWayTransformer.")

        self.handles.extend(
            [
                cross.register_forward_pre_hook(self._original_pre_hook),
                self.model.trans_imgbedding.register_forward_hook(self._output_hook("decoder_input")),
                self.model.sam2_feat_s1.register_forward_hook(self._output_hook("decoder_input_s1")),
                self.model.sam2_feat_s0.register_forward_hook(self._output_hook("decoder_input_s0")),
                transformer.register_forward_pre_hook(self._pre_input_hook("decoder_prompt_fused")),
                transformer.layers[0].register_forward_hook(
                    self._output_hook("transformer_block1", index=1, kind="tokens")
                ),
                transformer.layers[1].register_forward_hook(
                    self._output_hook("transformer_block2", index=1, kind="tokens")
                ),
                transformer.register_forward_hook(
                    self._output_hook("transformer_output", index=1, kind="tokens")
                ),
                decoder.output_upscaling[2].register_forward_hook(
                    self._output_hook("upscale_fusion_s1")
                ),
                decoder.output_upscaling[4].register_forward_hook(
                    self._output_hook("upscale_fusion_s0")
                ),
            ]
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def as_feature_map(self, name):
        activation = self.activations[name]
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError(f"No gradient reached stage: {name}")
        if self.kinds[name] == "tokens":
            reference = self.activations["decoder_prompt_fused"]
            height, width = reference.shape[-2:]
            activation = activation.transpose(1, 2).reshape(
                activation.shape[0], activation.shape[2], height, width
            )
            gradient = gradient.transpose(1, 2).reshape(
                gradient.shape[0], gradient.shape[2], height, width
            )
        if tuple(activation.shape[-2:]) != FEATURE_VIS_HW:
            # Visualization only. This does not alter the decoder forward pass.
            activation = F.interpolate(
                activation, size=FEATURE_VIS_HW, mode="bilinear", align_corners=False
            )
            gradient = F.interpolate(
                gradient, size=FEATURE_VIS_HW, mode="bilinear", align_corners=False
            )
        return activation, gradient


def calculate_gradcam(activation, gradient):
    """Standard Grad-CAM: ReLU(sum_c GAP(dy/dA_c) * A_c)."""
    activation = activation.detach().float()
    gradient = gradient.detach().float()
    weights = gradient.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=FEATURE_VIS_HW, mode="bilinear", align_corners=False)[0, 0]
    cam = cam.cpu().numpy()
    low, high = float(np.nanmin(cam)), float(np.nanmax(cam))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(cam.shape, dtype=np.uint8)
    return np.clip((cam - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def cam_overlay(cam_u8, image_bgr, alpha):
    cam_u8 = cv2.resize(cam_u8, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0.0)


def read_gt(mask_path, height, width, device):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(gt > 0).to(device=device, dtype=torch.float32)[None, None]


def make_target_score(mask_logits, mask_path, args):
    probability = torch.sigmoid(mask_logits).detach()
    if args.cam_target == "gt":
        target_region = read_gt(mask_path, mask_logits.shape[-2], mask_logits.shape[-1], mask_logits.device)
    else:
        target_region = (probability >= args.mask_threshold).to(mask_logits.dtype)
        # An empty hard prediction would produce an identically zero score.
        if target_region.sum() == 0:
            target_region = probability
    return (mask_logits * target_region).sum() / target_region.sum().clamp_min(1.0), target_region


def save_sample(model, image_path, mask_path, class_names, sample_dir, args, split):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_bgr = cv2.resize(image_bgr, (args.input_size[1], args.input_size[0]), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255).to(args.device)

    model.zero_grad(set_to_none=True)
    with DecoderGradCAMCapture(model) as capture:
        mask_logits, class_logits = model(image=image_tensor, class_names=class_names)
        score, target_region = make_target_score(mask_logits, mask_path, args)
        score.backward()

        os.makedirs(sample_dir, exist_ok=True)
        cv2.imwrite(os.path.join(sample_dir, "00_input.png"), image_bgr)
        panels = [titled_panel(image_bgr, "Input image")]
        shapes = {}
        for file_stem, key, title in CAM_STAGES:
            activation, gradient = capture.as_feature_map(key)
            cam = calculate_gradcam(activation, gradient)
            overlay = cam_overlay(cam, image_bgr, args.alpha)
            cv2.imwrite(os.path.join(sample_dir, file_stem + "_raw.png"), cam)
            cv2.imwrite(os.path.join(sample_dir, file_stem + "_overlay.png"), overlay)
            panels.append(titled_panel(overlay, title))
            shapes[key] = list(activation.shape)
            if args.save_npy:
                np.save(os.path.join(sample_dir, file_stem + ".npy"), cam)

    probability = torch.sigmoid(mask_logits.detach())[0, 0].cpu().numpy()
    probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(sample_dir, "11_prediction_probability.png"), probability_u8)
    target_u8 = (target_region.detach()[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(sample_dir, f"11_cam_target_{args.cam_target}.png"), target_u8)
    binary_mask = clean_prediction_mask(
        probability,
        threshold=args.prediction_threshold,
        kernel_size=args.mask_smooth_kernel,
    )
    cv2.imwrite(os.path.join(sample_dir, "11_prediction_mask.png"), binary_mask)
    prediction_overlay = hard_mask_overlay(image_bgr, binary_mask, alpha=0.55)
    cv2.imwrite(os.path.join(sample_dir, "11_prediction_overlay.png"), prediction_overlay)
    panels.append(titled_panel(prediction_overlay, "Cleaned prediction mask"))

    pred_id = int(class_logits.detach().argmax(dim=-1)[0].item())
    metadata = {
        "split": split,
        "image_path": image_path,
        "mask_path": mask_path,
        "cam_target": args.cam_target,
        "mask_threshold": args.mask_threshold,
        "prediction_threshold": args.prediction_threshold,
        "mask_smooth_kernel": args.mask_smooth_kernel,
        "target_score": float(score.detach().cpu()),
        "predicted_class": class_names[pred_id],
        "feature_shapes": shapes,
    }
    with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    rows = []
    for start in range(0, len(panels), 3):
        row = panels[start:start + 3]
        while len(row) < 3:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(os.path.join(sample_dir, "overview.png"), np.concatenate(rows, axis=0))


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    args.root_info = absolute(project_root, args.root_info)
    args.output_dir = absolute(project_root, args.output_dir)
    if args.max_per_class <= 0 or args.sample_stride <= 0:
        raise ValueError("--max-per-class and --sample-stride must be positive")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold must be in [0, 1]")
    if not 0.0 <= args.prediction_threshold <= 1.0:
        raise ValueError("--prediction-threshold must be in [0, 1]")
    if args.mask_smooth_kernel <= 0 or args.mask_smooth_kernel % 2 == 0:
        raise ValueError("--mask-smooth-kernel must be a positive odd integer")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    model = build_model(args, project_root)
    # Gradients are required only for captured activations, never for weights.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    datasets = {
        split: load_dataset_index(args.root_info, project_root, split)
        for split in args.splits
    }
    selected = {
        split: {
            name: grouped[name][::args.sample_stride][:args.max_per_class]
            for name in class_names
        }
        for split, (class_names, grouped) in datasets.items()
    }
    total = sum(
        len(items)
        for split_selection in selected.values()
        for items in split_selection.values()
    )
    failures = []
    progress = tqdm(total=total, desc="Grad-CAM", unit="sample")
    for split in args.splits:
        class_names, _ = datasets[split]
        for class_name in class_names:
            for index, (unique_id, image_path, mask_path) in enumerate(selected[split][class_name], 1):
                sample_dir = os.path.join(
                    args.output_dir,
                    split,
                    safe_name(class_name),
                    f"{index:02d}_{safe_name(unique_id)}",
                )
                try:
                    save_sample(model, image_path, mask_path, class_names, sample_dir, args, split)
                except Exception as exc:
                    failures.append((split, class_name, unique_id, str(exc)))
                    print(f"\n[WARN] {split}/{class_name}/{unique_id}: {exc}")
                progress.update(1)
    progress.close()

    os.makedirs(args.output_dir, exist_ok=True)
    failure_path = os.path.join(args.output_dir, "failures.json")
    if failures:
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"Failed samples: {len(failures)} (details: {failure_path})")
    elif os.path.isfile(failure_path):
        os.remove(failure_path)
    print(f"Saved Grad-CAM visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
