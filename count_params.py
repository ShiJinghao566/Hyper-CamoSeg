import argparse
import os
from collections import defaultdict

from mmengine import Config
import torch
import yaml

from main import OVSAM
from segment_anything import sam_model_registry


def format_count(num_params):
    return f"{num_params:,} ({num_params / 1e6:.3f}M)"


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def count_by_top_module(model):
    stats = defaultdict(lambda: [0, 0])
    for name, param in model.named_parameters():
        top_name = name.split(".", 1)[0]
        stats[top_name][0] += param.numel()
        if param.requires_grad:
            stats[top_name][1] += param.numel()
    return stats


def resolve_optional_path(path):
    if path is None:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def build_model(args):
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(
        dict(
            model_name=args.model_name,
            root_info=args.root_info,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            clip_checkpoint=args.clip_checkpoint,
            devices=args.devices,
            train_eval=args.train_eval,
            train_eval_start_epoch=args.train_eval_start_epoch,
        )
    )
    with open(cfg.root_info, mode="r", encoding="utf-8") as f:
        cfg.root_info = yaml.safe_load(f)

    sam_model = sam_model_registry[cfg.model_type](checkpoint=resolve_optional_path(cfg.checkpoint))
    model = OVSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        clip_checkpoint=resolve_optional_path(cfg.clip_checkpoint),
        target_background_hypergraph_cfg=getattr(cfg, "target_background_hypergraph", {}),
    )
    if args.load_from:
        state = torch.load(resolve_optional_path(args.load_from), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=args.strict)
    model.get_grouped_params()
    return model


def main():
    parser = argparse.ArgumentParser("Count model parameters.")
    parser.add_argument("--config", default="configs/ovcoser.py")
    parser.add_argument("--model-name", default="OVCoser")
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--train-eval", default=True)
    parser.add_argument("--train-eval-start-epoch", type=int, default=25)
    parser.add_argument("--model-type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    parser.add_argument("--checkpoint", default="pretrained/sam_vit_b_01ec64.pth")
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--load-from", default=None, help="Optional trained model checkpoint.")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading --load-from.")
    parser.add_argument("--by-module", action="store_true", help="Print parameter counts by top-level module.")
    args = parser.parse_args()

    model = build_model(args)
    total, trainable, frozen = count_params(model)

    print("Parameter Count")
    print(f"  Total:     {format_count(total)}")
    print(f"  Trainable: {format_count(trainable)}")
    print(f"  Frozen:    {format_count(frozen)}")
    print(f"  Trainable ratio: {trainable / total * 100:.2f}%")

    if args.by_module:
        print("\nBy Top-Level Module")
        for name, (module_total, module_trainable) in sorted(count_by_top_module(model).items()):
            module_frozen = module_total - module_trainable
            print(
                f"  {name:<28}"
                f" total={format_count(module_total):>20}"
                f" trainable={format_count(module_trainable):>20}"
                f" frozen={format_count(module_frozen):>20}"
            )


if __name__ == "__main__":
    main()
