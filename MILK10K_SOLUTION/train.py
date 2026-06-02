from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from models.model_factory import build_model
from src.dataset import build_meta_processor
from src.train import train
from src.utils import load_config, merge_cli_args, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a MILK10k classification model.")
    p.add_argument("--config",         required=True)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--image_size",     type=int,   default=None)
    p.add_argument("--model_name",     type=str,   default=None)
    p.add_argument("--train_csv",      type=str,   default=None)
    p.add_argument("--val_csv",        type=str,   default=None)
    p.add_argument("--image_dir",      type=str,   default=None)
    p.add_argument("--checkpoint_dir", type=str,   default=None)
    p.add_argument("--output_dir",     type=str,   default=None)
    p.add_argument("--num_workers",    type=int,   default=None)
    p.add_argument("--loss_name",      type=str,   default=None)
    p.add_argument("--mode",           type=str,   default=None,
                   choices=["single_image", "dual_image"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = merge_cli_args(cfg, args)

    print("=" * 60)
    print(f"  Model  : {cfg.get('model_name', '?')}")
    print(f"  Mode   : {cfg.get('mode', 'single_image')}")
    print(f"  ImgSize: {cfg.get('image_size', 224)}")
    print(f"  Epochs : {cfg.get('epochs', 30)}")
    print(f"  LR     : {cfg.get('lr', 1e-4)}")
    print(f"  Loss   : {cfg.get('loss_name', 'bce')}")
    print(f"  TrainCSV: {cfg.get('train_csv','?')}")
    print(f"  ImageDir: {cfg.get('image_dir','?')}")
    print("=" * 60)

    set_seed(cfg.get("seed", 42))

    meta_dim = 0
    if cfg.get("use_metadata", False):
        train_csv = cfg.get("train_csv")
        if train_csv and Path(train_csv).exists():
            proc     = build_meta_processor(pd.read_csv(train_csv))
            meta_dim = proc.meta_dim
            print(f"Metadata dim: {meta_dim}")

    model = build_model(cfg, meta_dim=meta_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    best_f1 = train(cfg, model)
    print(f"\nDone. Best macro F1 on validation: {best_f1:.4f}")


if __name__ == "__main__":
    main()