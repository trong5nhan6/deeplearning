import warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABEL_COLS = ["AKIEC","BCC","BEN_OTH","BKL","DF","INF","MAL_OTH","MEL","NV","SCCKA","VASC"]

_MONET_BASES = [
    "MONET_ulceration_crust","MONET_hair","MONET_vasculature_vessels",
    "MONET_erythema","MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

class MetadataProcessor:
    KNOWN_SITES = [
        "head_neck_face","upper_extremity","lower_extremity","torso",
        "palms_soles","oral_genital","anterior_torso","posterior_torso",
        "lateral_torso","unknown",
    ]
    SEX_MAP = {"male":0.0,"m":0.0,"female":1.0,"f":1.0}

    def __init__(self):
        self.age_mean = 50.0
        self.age_std  = 15.0
        self.site_index = {}
        self.meta_dim   = 0
        for i, s in enumerate(self.KNOWN_SITES):
            self.site_index[s] = i

    def fit(self, df):
        if "age_approx" in df.columns:
            vals = pd.to_numeric(df["age_approx"], errors="coerce").dropna()
            if len(vals) > 0:
                self.age_mean = float(vals.mean())
                self.age_std  = max(float(vals.std()), 1.0)
        if "site" in df.columns:
            for s in df["site"].dropna().unique():
                norm = self._norm_site(s)
                if norm not in self.site_index:
                    self.site_index[norm] = len(self.site_index)
        n_sites = len(self.site_index) + 1
        self.meta_dim = 3 + n_sites + 7*2

    @staticmethod
    def _norm_site(s):
        return str(s).lower().strip().replace(" ","_").replace("/","_")

    def transform(self, row):
        feats = []
        age = pd.to_numeric(row.get("age_approx", float("nan")), errors="coerce")
        feats.append(float((age - self.age_mean) / self.age_std) if not pd.isna(age) else 0.0)
        sex_raw = str(row.get("sex","")).lower().strip()
        feats.append(self.SEX_MAP.get(sex_raw, 0.5))
        tone = pd.to_numeric(row.get("skin_tone_class", float("nan")), errors="coerce")
        feats.append(float(tone)/5.0 if not pd.isna(tone) else 0.5)
        n_sites  = len(self.site_index) + 1
        site_vec = np.zeros(n_sites, dtype=np.float32)
        site_norm = self._norm_site(str(row.get("site","")))
        idx = self.site_index.get(site_norm, len(self.site_index))
        if idx < n_sites:
            site_vec[idx] = 1.0
        feats.extend(site_vec.tolist())
        for base in _MONET_BASES:
            col = "clin_" + base
            v = pd.to_numeric(row.get(col, float("nan")), errors="coerce")
            feats.append(0.0 if pd.isna(v) else float(v))
        for base in _MONET_BASES:
            col = "derm_" + base
            v = pd.to_numeric(row.get(col, float("nan")), errors="coerce")
            feats.append(0.0 if pd.isna(v) else float(v))
        return np.array(feats, dtype=np.float32)


class MILK10kDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None,
                 mode="single_image", image_type="dermoscopy",
                 is_test=False, meta_processor=None, cfg=None):
        self.df           = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir    = Path(image_dir)
        self.transform    = transform
        self.mode         = mode
        self.image_type   = image_type
        self.is_test      = is_test
        self.processor    = meta_processor
        self.use_metadata = meta_processor is not None
        cfg = cfg or {}
        self.lesion_col   = cfg.get("lesion_col")   or "lesion_id"
        self.clinical_col = cfg.get("clinical_col") or "clinical_path"
        self.derm_col     = cfg.get("derm_col")     or "derm_path"
        if self.lesion_col not in self.df.columns:
            self.lesion_col = self.df.columns[0]
        if self.clinical_col not in self.df.columns:
            self.clinical_col = None
        if self.derm_col not in self.df.columns:
            self.derm_col = None
        self.label_cols = [c for c in LABEL_COLS if c in self.df.columns]

    def _load_image(self, rel_path):
        full_path = self.image_dir / rel_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Image not found: {full_path}\n"
                f"  image_dir={self.image_dir}  rel_path={rel_path}"
            )
        return np.array(Image.open(full_path).convert("RGB"))

    def _apply_transform(self, img, key="default"):
        if self.transform is None:
            return torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        t = self.transform.get(key) if isinstance(self.transform, dict) else self.transform
        if t is None and isinstance(self.transform, dict):
            t = list(self.transform.values())[0]
        return t(image=img)["image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        lesion = str(row[self.lesion_col])
        sample = {"lesion": lesion}

        if self.mode == "dual_image":
            if self.clinical_col is None or self.derm_col is None:
                raise ValueError("dual_image requires clinical_path and derm_path columns")
            clin_img = self._load_image(str(row[self.clinical_col]))
            derm_img = self._load_image(str(row[self.derm_col]))
            sample["clinical_image"] = self._apply_transform(clin_img, "clinical")
            sample["derm_image"]     = self._apply_transform(derm_img, "derm")
        else:
            if self.image_type == "dermoscopy" and self.derm_col:
                img_path = str(row[self.derm_col])
            elif self.clinical_col:
                img_path = str(row[self.clinical_col])
            else:
                raise ValueError("No image column found for single_image mode")
            img = self._load_image(img_path)
            sample["image"] = self._apply_transform(img, "default")

        if self.use_metadata and self.processor is not None:
            sample["metadata"] = torch.tensor(self.processor.transform(row), dtype=torch.float32)

        if not self.is_test and self.label_cols:
            labels = [float(pd.to_numeric(row.get(c, 0), errors="coerce") or 0.0) for c in LABEL_COLS]
            sample["labels"] = torch.tensor(labels, dtype=torch.float32)

        return sample


def build_meta_processor(df):
    proc = MetadataProcessor()
    proc.fit(df)
    return proc

def detect_meta_cols(df):
    return {}