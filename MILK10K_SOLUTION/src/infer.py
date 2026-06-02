from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_factory import build_model
from src.dataset import MILK10kDataset, MetadataProcessor, build_meta_processor
from src.submission import build_submission
from src.transforms import get_val_transforms, get_tta_transforms
from src.utils import get_device, load_checkpoint, load_config, set_seed


@torch.no_grad()
def run_inference(model, loader, device, use_amp=True):
    model.eval()
    all_lesions, all_probs = [], []
    pbar = tqdm(loader, desc="  Infer", leave=False, dynamic_ncols=True)
    for batch in pbar:
        all_lesions.extend(batch["lesion"])
        with torch.cuda.amp.autocast(enabled=use_amp):
            if "clinical_image" in batch and "derm_image" in batch:
                clin = batch["clinical_image"].to(device, non_blocking=True)
                derm = batch["derm_image"].to(device, non_blocking=True)
                meta = batch.get("metadata")
                if meta is not None:
                    meta = meta.to(device, non_blocking=True)
                logits = model(clin, derm, meta)
            else:
                img  = batch["image"].to(device, non_blocking=True)
                meta = batch.get("metadata")
                if meta is not None:
                    meta = meta.to(device, non_blocking=True)
                logits = model(img, meta)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return all_lesions, np.vstack(all_probs)


def _rebuild_meta_processor(cfg):
    if not cfg.get("use_metadata", False):
        return None
    train_csv = cfg.get("train_csv")
    if not train_csv or not Path(train_csv).exists():
        print("[WARN] use_metadata=True but train_csv not found")
        return None
    return build_meta_processor(pd.read_csv(train_csv))


def infer(config_path, checkpoint, test_csv, out_path, image_dir=None, use_tta=False):
    cfg    = load_config(config_path)
    device = get_device()
    set_seed(cfg.get("seed", 42))
    use_amp = cfg.get("use_amp", True) and torch.cuda.is_available()

    if image_dir is None:
        image_dir = cfg.get("test_image_dir") or cfg.get("image_dir")

    meta_processor = _rebuild_meta_processor(cfg)
    meta_dim = meta_processor.meta_dim if meta_processor else 0

    model = build_model(cfg, meta_dim=meta_dim)
    ckpt  = load_checkpoint(checkpoint, model, device=str(device))
    model = model.to(device)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}, best_f1={ckpt.get('best_macro_f1','?')}")

    val_transform = get_val_transforms(cfg.get("image_size", 224))
    mode          = cfg.get("mode", "single_image")
    img_type      = cfg.get("image_type", "dermoscopy")
    batch_size    = cfg.get("batch_size", 32) * 2
    num_workers   = cfg.get("num_workers", 4)

    if use_tta:
        tta_transforms = get_tta_transforms(cfg.get("image_size", 224))
        all_probs_list, lesion_ids = [], None
        for i, t in enumerate(tta_transforms):
            ds = MILK10kDataset(
                csv_path=test_csv, image_dir=image_dir, transform=t,
                mode=mode, image_type=img_type, is_test=True,
                meta_processor=meta_processor, cfg=cfg,
            )
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
            ids, probs = run_inference(model, loader, device, use_amp)
            if lesion_ids is None:
                lesion_ids = ids
            all_probs_list.append(probs)
        probs = np.mean(all_probs_list, axis=0)
    else:
        test_ds = MILK10kDataset(
            csv_path=test_csv, image_dir=image_dir, transform=val_transform,
            mode=mode, image_type=img_type, is_test=True,
            meta_processor=meta_processor, cfg=cfg,
        )
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        lesion_ids, probs = run_inference(model, loader, device, use_amp)

    print(f"Inference done: {len(lesion_ids)} lesions")
    build_submission(lesion_ids, probs, out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_csv",   required=True)
    p.add_argument("--out",        required=True)
    p.add_argument("--image_dir",  default=None)
    p.add_argument("--tta",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(config_path=args.config, checkpoint=args.checkpoint,
          test_csv=args.test_csv, out_path=args.out,
          image_dir=args.image_dir, use_tta=args.tta)