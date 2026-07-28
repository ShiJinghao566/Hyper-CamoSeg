#!/usr/bin/env python3
"""Visualize the intermediate features of the OVCamo training pipeline.

For every training class, the script samples up to N items at a configurable stride and writes one
directory per sample containing the input, GT, ten feature visualizations, the
predicted mask, its overlay, and a contact sheet.
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from main import OVSAM, SAM2MaskDecoderAdapter, load_sam2_prompt_encoder_weights
from segment_anything import sam_model_registry


STAGES = [
    ("01_original_feature", "original_feature", "Original res5 feature"),
    ("02_decoder_input_main", "decoder_input", "Decoder input: main 256x64x64"),
    ("03_decoder_input_s1", "decoder_input_s1", "Decoder input: feat_s1 64x128x128"),
    ("04_decoder_input_s0", "decoder_input_s0", "Decoder input: feat_s0 32x256x256"),
    ("05_decoder_prompt_fused", "decoder_prompt_fused", "Main + dense prompt"),
    ("06_transformer_block1", "transformer_block1", "Two-way transformer block 1"),
    ("07_transformer_block2", "transformer_block2", "Two-way transformer block 2"),
    ("08_transformer_output", "transformer_output", "Transformer visual output"),
    ("09_upscale_fusion_s1", "upscale_fusion_s1", "Upscale + feat_s1"),
    ("10_upscale_fusion_s0", "upscale_fusion_s0", "Upscale + feat_s0"),
]

# All non-prediction feature visualizations use the current visualized
# Upscale+feat_s1 resolution as their common spatial reference.
FEATURE_VIS_HW = (256, 256)
NODE_GRID_HW = (12, 12)


class IntermediateFeatureCapture:
    """Capture all requested stages without changing OVSAM.forward's API."""

    def __init__(self, model):
        self.model = model
        self.features = {}
        self.handles = []

    @staticmethod
    def _first_tensor(value):
        return value[0] if isinstance(value, (tuple, list)) else value

    def _save_pre_input(self, name):
        def hook(_module, inputs):
            self.features[name] = inputs[0]
        return hook

    def _save_output(self, name):
        def hook(_module, _inputs, output):
            self.features[name] = self._first_tensor(output)
        return hook

    def _save_upscale_s1_for_visualization(self, _module, _inputs, output):
        # Visualization only: match the following feat_s0 fusion resolution.
        self.features["upscale_fusion_s1"] = F.interpolate(
            self._first_tensor(output),
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

    def _save_transformer_block(self, name):
        def hook(_module, _inputs, output):
            image_tokens = output[1]
            fused = self.features["decoder_prompt_fused"]
            height, width = fused.shape[-2:]
            self.features[name] = image_tokens.transpose(1, 2).reshape(
                image_tokens.shape[0], image_tokens.shape[2], height, width
            )
        return hook

    def _save_transformer_output(self, _module, _inputs, output):
        image_tokens = output[1]
        fused = self.features["decoder_prompt_fused"]
        height, width = fused.shape[-2:]
        self.features["transformer_output"] = image_tokens.transpose(1, 2).reshape(
            image_tokens.shape[0], image_tokens.shape[2], height, width
        )

    def _save_mask_token_vector(self, _module, _inputs, output):
        self.features.setdefault("_mask_token_vector_list", []).append(output)

    def __enter__(self):
        cross = self.model.crossattforlastlayer
        decoder_adapter = self.model.mask_decoder
        decoder = getattr(decoder_adapter, "decoder", decoder_adapter)
        transformer = decoder.transformer
        self.handles.extend(
            [
                # Input to TBTA is the original CLIP res5 feature.
                cross.register_forward_pre_hook(self._save_pre_input("original_feature")),
                self.model.trans_imgbedding.register_forward_hook(self._save_output("decoder_input")),
                self.model.sam2_feat_s1.register_forward_hook(self._save_output("decoder_input_s1")),
                self.model.sam2_feat_s0.register_forward_hook(self._save_output("decoder_input_s0")),
                # Transformer input is exactly image_embeddings + dense_prompt_embeddings.
                transformer.register_forward_pre_hook(self._save_pre_input("decoder_prompt_fused")),
                transformer.layers[0].register_forward_hook(self._save_transformer_block("transformer_block1")),
                transformer.layers[1].register_forward_hook(self._save_transformer_block("transformer_block2")),
                transformer.register_forward_hook(self._save_transformer_output),
                # The decoder manually unpacks output_upscaling; activation outputs include skip fusion.
                decoder.output_upscaling[2].register_forward_hook(self._save_upscale_s1_for_visualization),
                decoder.output_upscaling[4].register_forward_hook(self._save_output("upscale_fusion_s0")),
            ]
        )
        for hypernetwork in decoder.output_hypernetworks_mlps:
            self.handles.append(hypernetwork.register_forward_hook(self._save_mask_token_vector))
        return self.features

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        token_vectors = self.features.pop("_mask_token_vector_list", None)
        if token_vectors:
            self.features["mask_token_vectors"] = torch.stack(token_vectors, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-weight", required=True, help="Trained OVSAM state_dict path.")
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml")
    parser.add_argument("--output-dir", default="outputs/intermediate_visualizations")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Dataset splits to visualize; outputs are stored in separate subdirectories.",
    )
    parser.add_argument("--max-per-class", type=int, default=10)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=20,
        help="Take one sample every N items within each class; use 20 or 50 to increase diversity.",
    )
    parser.add_argument("--input-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--sam-checkpoint", default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam2-checkpoint", default="pretrained/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam2-root", default="sam2-main")
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--config", default="configs/ovcoser.py")
    parser.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=0.5,
        help="Normalized Upscale+feat_s1 cell-response threshold for foreground nodes.",
    )
    parser.add_argument(
        "--background-darkness",
        type=float,
        default=0.25,
        help="Brightness multiplier for image cells classified as background nodes.",
    )
    parser.add_argument("--save-npy", action="store_true", help="Also save raw feature tensors (large).")
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected trained-weight keys.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose overview.png already exists; useful for resuming an interrupted run.",
    )
    parser.add_argument(
        "--cuda-sync",
        action="store_true",
        help="Synchronize CUDA after model forward so the traceback identifies the failing sample accurately.",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=50,
        help="Release unused CUDA allocator blocks every N processed samples; 0 disables it.",
    )
    return parser.parse_args()


def absolute(project_root, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def safe_name(value):
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    return value or "unnamed"


def load_hypergraph_config(config_path):
    # Avoid importing mmengine here; the training config is a plain Python file.
    namespace = {}
    with open(config_path, "r", encoding="utf-8") as f:
        exec(compile(f.read(), config_path, "exec"), {}, namespace)
    return dict(namespace.get("target_background_hypergraph", {}))


def unwrap_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise TypeError("The trained weight must contain a PyTorch state_dict.")
    return {key.removeprefix("module."): value for key, value in raw.items()}


def build_model(args, project_root):
    sam_checkpoint = absolute(project_root, args.sam_checkpoint)
    sam2_checkpoint = absolute(project_root, args.sam2_checkpoint)
    sam2_root = absolute(project_root, args.sam2_root)
    clip_checkpoint = absolute(project_root, args.clip_checkpoint)
    trained_weight = absolute(project_root, args.trained_weight)

    for label, path in (
        ("SAM checkpoint", sam_checkpoint),
        ("SAM2 checkpoint", sam2_checkpoint),
        ("trained weight", trained_weight),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if not os.path.isdir(sam2_root):
        raise FileNotFoundError(f"SAM2 root not found: {sam2_root}")
    if clip_checkpoint is not None and not os.path.isfile(clip_checkpoint):
        raise FileNotFoundError(f"CLIP checkpoint not found: {clip_checkpoint}")

    sam_model = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
    mask_decoder = SAM2MaskDecoderAdapter(sam2_checkpoint, sam2_root)
    load_sam2_prompt_encoder_weights(sam_model.prompt_encoder, sam2_checkpoint)
    hypergraph_cfg = load_hypergraph_config(absolute(project_root, args.config))
    model = OVSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_checkpoint=clip_checkpoint,
        target_background_hypergraph_cfg=hypergraph_cfg,
    )

    raw = torch.load(trained_weight, map_location="cpu")
    incompat = model.load_state_dict(unwrap_state_dict(raw), strict=not args.non_strict)
    if args.non_strict:
        print(f"Missing keys: {len(incompat.missing_keys)}; unexpected keys: {len(incompat.unexpected_keys)}")
    return model.to(args.device).eval()


def load_dataset_index(root_info_path, project_root, split):
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported dataset split: {split}")
    with open(root_info_path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    def resolve_data_path(path):
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(project_root, path))

    class_json = resolve_data_path(info["OVCamo_CLASS_JSON_PATH"])
    sample_json = resolve_data_path(info["OVCamo_SAMPLE_JSON_PATH"])
    prefix = "OVCamo_TR" if split == "train" else "OVCamo_TE"
    image_dir = resolve_data_path(info[f"{prefix}_IMAGE_DIR"])
    mask_dir = resolve_data_path(info[f"{prefix}_MASK_DIR"])
    with open(class_json, "r", encoding="utf-8") as f:
        classes = [item["name"] for item in json.load(f) if item["split"] == split]
    with open(sample_json, "r", encoding="utf-8") as f:
        samples = json.load(f)

    grouped = {name: [] for name in classes}
    for sample in samples:
        class_name = sample["base_class"]
        if class_name not in grouped:
            continue
        unique_id = sample["unique_id"]
        image_path = os.path.join(image_dir, unique_id + os.path.splitext(sample["image"])[1])
        mask_path = os.path.join(mask_dir, unique_id + os.path.splitext(sample["mask"])[1])
        grouped[class_name].append((unique_id, image_path, mask_path))
    return classes, grouped


def load_training_index(root_info_path, project_root):
    """Backward-compatible helper used by older external scripts."""
    return load_dataset_index(root_info_path, project_root, split="train")


def feature_response(feature):
    """Convert BxCxHxW features into a robust single-channel RMS response."""
    feature = feature.detach().float()
    if tuple(feature.shape[-2:]) != FEATURE_VIS_HW:
        feature = F.interpolate(
            feature,
            size=FEATURE_VIS_HW,
            mode="bilinear",
            align_corners=False,
        )
    feature = feature[0]
    response = torch.sqrt(torch.mean(feature.square(), dim=0) + 1e-12).cpu().numpy()
    finite = np.isfinite(response)
    if not finite.any():
        return np.zeros(response.shape, dtype=np.uint8)
    valid = response[finite]
    low, high = np.percentile(valid, (1.0, 99.0))
    if high <= low:
        low, high = float(valid.min()), float(valid.max())
    response = np.nan_to_num((response - low) / max(high - low, 1e-12), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(response * 255.0, 0, 255).astype(np.uint8)


def heatmap_overlay(feature, image_rgb, alpha):
    response = feature_response(feature)
    response = cv2.resize(response, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_bgr = cv2.applyColorMap(response, cv2.COLORMAP_TURBO)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)


def mask_token_heatmap(token_vectors):
    """Render the four 32-D dynamic mask vectors as a labeled matrix heatmap."""
    matrix = token_vectors.detach().float()[0].cpu().numpy()
    finite = np.isfinite(matrix)
    valid = matrix[finite]
    if valid.size == 0:
        normalized = np.zeros(matrix.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(valid, (1.0, 99.0))
        normalized = np.nan_to_num((matrix - low) / max(high - low, 1e-12), nan=0.0)
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    cell_w, cell_h, label_w = 16, 52, 110
    heat = cv2.resize(
        normalized,
        (normalized.shape[1] * cell_w, normalized.shape[0] * cell_h),
        interpolation=cv2.INTER_NEAREST,
    )
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    canvas = np.zeros((heat.shape[0] + 34, heat.shape[1] + label_w, 3), dtype=np.uint8)
    canvas[34:, label_w:] = heat
    cv2.putText(canvas, "channel 0 ... 31", (label_w + 5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for row in range(matrix.shape[0]):
        cv2.putText(
            canvas,
            f"mask token {row}",
            (5, 34 + row * cell_h + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def visualize_fg_bg_nodes(feature, image_bgr, threshold, background_darkness):
    """Map a 16x16 foreground/background node decision back to the input image."""
    response = feature_response(feature).astype(np.float32) / 255.0
    node_response = cv2.resize(
        response,
        (NODE_GRID_HW[1], NODE_GRID_HW[0]),
        interpolation=cv2.INTER_AREA,
    )
    foreground_nodes = node_response >= threshold
    pixel_mask = cv2.resize(
        foreground_nodes.astype(np.uint8),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    visualization = np.clip(
        image_bgr.astype(np.float32) * background_darkness, 0, 255
    ).astype(np.uint8)
    visualization[pixel_mask] = image_bgr[pixel_mask]

    # Draw exact node boundaries after mapping the 16x16 grid to the input image.
    height, width = image_bgr.shape[:2]
    grid_color = (190, 190, 190)
    for row in range(1, NODE_GRID_HW[0]):
        y = round(row * height / NODE_GRID_HW[0])
        cv2.line(visualization, (0, y), (width - 1, y), grid_color, 1, cv2.LINE_AA)
    for col in range(1, NODE_GRID_HW[1]):
        x = round(col * width / NODE_GRID_HW[1])
        cv2.line(visualization, (x, 0), (x, height - 1), grid_color, 1, cv2.LINE_AA)
    return visualization, foreground_nodes, node_response


def titled_panel(image_bgr, title, width=384, height=384):
    panel = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    banner_h = 38
    canvas = np.zeros((height + banner_h, width, 3), dtype=np.uint8)
    canvas[banner_h:] = panel
    cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_sample(model, image_path, mask_path, class_names, sample_dir, args, split):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_bgr = cv2.resize(image_bgr, (args.input_size[1], args.input_size[0]), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255).to(args.device)

    with torch.inference_mode(), IntermediateFeatureCapture(model) as features:
        mask_logits, class_logits = model(image=image_tensor, class_names=class_names)
        if args.cuda_sync and args.device.startswith("cuda"):
            torch.cuda.synchronize()

    missing = [key for _, key, _ in STAGES if key not in features]
    if missing:
        raise RuntimeError(f"Failed to capture intermediate features: {missing}")

    os.makedirs(sample_dir, exist_ok=True)
    cv2.imwrite(os.path.join(sample_dir, "00_input.png"), image_bgr)
    panels = [titled_panel(image_bgr, "Input image")]

    if os.path.isfile(mask_path):
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (args.input_size[1], args.input_size[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(sample_dir, "00_ground_truth.png"), gt)

    for file_stem, key, title in STAGES:
        overlay = heatmap_overlay(features[key], image_rgb, args.alpha)
        cv2.imwrite(os.path.join(sample_dir, file_stem + ".png"), overlay)
        panels.append(titled_panel(overlay, title))
        if args.save_npy:
            np.save(os.path.join(sample_dir, file_stem + ".npy"), features[key].detach().float().cpu().numpy())

    node_image, foreground_nodes, node_response = visualize_fg_bg_nodes(
        features["upscale_fusion_s1"],
        image_bgr,
        args.node_threshold,
        args.background_darkness,
    )
    cv2.imwrite(os.path.join(sample_dir, "11_fg_bg_nodes.png"), node_image)
    node_mask_u8 = foreground_nodes.astype(np.uint8) * 255
    node_mask_u8 = cv2.resize(node_mask_u8, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(sample_dir, "11_fg_node_mask.png"), node_mask_u8)
    panels.append(titled_panel(node_image, "16x16 foreground/background nodes"))

    token_image = mask_token_heatmap(features["mask_token_vectors"])
    cv2.imwrite(os.path.join(sample_dir, "12_mask_token_vectors.png"), token_image)
    panels.append(titled_panel(token_image, "Mask-token dynamic vectors"))
    if args.save_npy:
        np.save(
            os.path.join(sample_dir, "11_node_response_16x16.npy"),
            node_response,
        )
        np.save(
            os.path.join(sample_dir, "12_mask_token_vectors.npy"),
            features["mask_token_vectors"].detach().float().cpu().numpy(),
        )

    probability = torch.sigmoid(mask_logits)[0, 0].detach().float().cpu().numpy()
    probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    # Only the saved prediction mask is binarized: background=0, foreground=255.
    mask_u8 = (probability >= 0.5).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(sample_dir, "13_prediction_mask.png"), mask_u8)
    # Keep the prediction overlay unchanged: visualize the continuous probability.
    mask_color = cv2.applyColorMap(probability_u8, cv2.COLORMAP_TURBO)
    mask_overlay = cv2.addWeighted(image_bgr, 0.45, mask_color, 0.55, 0.0)
    cv2.imwrite(os.path.join(sample_dir, "13_prediction_overlay.png"), mask_overlay)
    panels.append(titled_panel(mask_overlay, "Predicted mask"))

    pred_id = int(class_logits.argmax(dim=-1)[0].item())
    metadata = {
        "split": split,
        "image_path": image_path,
        "mask_path": mask_path,
        "predicted_class": class_names[pred_id],
        "predicted_class_id": pred_id,
        "feature_shapes": {key: list(value.shape) for key, value in features.items()},
        "node_grid_hw": list(NODE_GRID_HW),
        "node_threshold": args.node_threshold,
        "foreground_node_count": int(foreground_nodes.sum()),
        "background_node_count": int(foreground_nodes.size - foreground_nodes.sum()),
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
    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be positive")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if not 0.0 <= args.node_threshold <= 1.0:
        raise ValueError("--node-threshold must be in [0, 1]")
    if not 0.0 <= args.background_darkness <= 1.0:
        raise ValueError("--background-darkness must be in [0, 1]")

    model = build_model(args, project_root)
    datasets = {
        split: load_dataset_index(args.root_info, project_root, split)
        for split in args.splits
    }
    old_failure_path = os.path.join(args.output_dir, "failures.json")
    if os.path.isfile(old_failure_path):
        os.remove(old_failure_path)
    selected_by_split = {
        split: {
            name: grouped[name][::args.sample_stride][:args.max_per_class]
            for name in class_names
        }
        for split, (class_names, grouped) in datasets.items()
    }
    total = sum(
        len(items)
        for split_selection in selected_by_split.values()
        for items in split_selection.values()
    )
    progress = tqdm(total=total, desc="Visualizing", unit="sample")
    failures = []
    processed = 0
    for split in args.splits:
        class_names, _ = datasets[split]
        for class_name in class_names:
            class_dir = os.path.join(args.output_dir, split, safe_name(class_name))
            for index, (unique_id, image_path, mask_path) in enumerate(
                selected_by_split[split][class_name], 1
            ):
                sample_dir = os.path.join(class_dir, f"{index:02d}_{safe_name(unique_id)}")
                if args.skip_existing and os.path.isfile(os.path.join(sample_dir, "overview.png")):
                    progress.update(1)
                    continue
                try:
                    save_sample(model, image_path, mask_path, class_names, sample_dir, args, split)
                except Exception as exc:
                    failures.append((split, class_name, unique_id, str(exc)))
                    print(f"\n[WARN] {split}/{class_name}/{unique_id}: {exc}")
                    if "CUDA error" in str(exc) or "illegal memory access" in str(exc):
                        progress.close()
                        raise RuntimeError(
                            "A fatal CUDA error occurred. The CUDA context is no longer usable, so the "
                            "run was stopped instead of reporting false failures for every later sample. "
                            "Restart the Python process and use --skip-existing --cuda-sync to resume and "
                            "locate the first real failure."
                        ) from exc
                finally:
                    processed += 1
                    if (
                        args.empty_cache_every > 0
                        and processed % args.empty_cache_every == 0
                        and args.device.startswith("cuda")
                    ):
                        gc.collect()
                        torch.cuda.empty_cache()
                progress.update(1)
    progress.close()
    print(f"Saved visualizations to: {args.output_dir}")
    if failures:
        failure_path = os.path.join(args.output_dir, "failures.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(failure_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"Failed samples: {len(failures)} (details: {failure_path})")


if __name__ == "__main__":
    main()
