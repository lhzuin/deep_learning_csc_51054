# data/datamodule_fusion.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# reuse your own implementations:
from .meta_utils import (
    parse_tweets_meta,      # same as before
    build_meta_features,    # same as before
    author_based_split,     # same as before
)


class FusionDataset(Dataset):
    """
    Wraps a DataFrame + precomputed metadata array.
    Expects df to have columns: full_text, user_desc, label.
    """
    def __init__(self, df, meta_array):
        self.df = df.reset_index(drop=True)
        self.meta = torch.tensor(meta_array, dtype=torch.float32)
        self.labels = torch.tensor(self.df["label"].astype(int).values, dtype=torch.long)
        self.texts = self.df["full_text"].astype(str).tolist()
        # you may have "user.description" in df, rename to user_desc beforehand
        self.descs = self.df["user_desc"].astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "full_text": self.texts[idx],
            "user_desc": self.descs[idx],
            "meta": self.meta[idx],
            "label": self.labels[idx],
        }


def fusion_collate_fn(batch):
    # merge list of dicts into one dict where:
    #  - text fields are list[str]
    #  - meta is FloatTensor (B, D)
    #  - label is LongTensor (B,)
    full_texts = [b["full_text"] for b in batch]
    descs      = [b["user_desc"] for b in batch]
    metas      = torch.stack([b["meta"] for b in batch], dim=0)
    labels     = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "full_text": full_texts,
        "user_desc": descs,
        "meta": metas,
        "label": labels,
    }


class FusionDataModule:
    def __init__(
        self,
        train_path: str,
        batch_size: int = 16,
        num_workers: int = 2,
        val_size: float = 0.1,
        random_state: int = 42,
        meta_ckpt_path: str = "checkpoints/meta_tower_best.pt",
    ):
        self.train_path   = train_path
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.val_size     = val_size
        self.random_state = random_state
        self.meta_ckpt_path = meta_ckpt_path

        self.train_ds = None
        self.val_ds   = None

        # for compatibility with Trainer.init_model (n_source_buckets)
        self.n_source_buckets = None  # not used in fusion, but harmless

    def setup(self, checkpoint_save_path, type="random"):
        """
        type: "random" or "author_based" for the split, same spirit as your other datamodules.
        """
        # 1) parse raw tweets with your helper (builds full_text, author_pseudo_id, basic fields)
        raw_df = parse_tweets_meta(self.train_path, expect_label=True)

        # 2) build user_desc column from your user.description/value
        if "user.description" in raw_df.columns:
            raw_df["user_desc"] = raw_df["user.description"].fillna("")
        else:
            raw_df["user_desc"] = ""

        # 3) load meta tower checkpoint and reuse scaler/feature_cols/stats/src2idx
        meta_ckpt = torch.load(self.meta_ckpt_path, map_location="cpu", weights_only=False)
        scaler       = meta_ckpt["scaler"]         # sklearn StandardScaler
        feature_cols = meta_ckpt["feature_cols"]   # your best_with_seven list
        stats        = meta_ckpt["stats"]
        src2idx      = meta_ckpt["src2idx"]

        # 4) split BEFORE computing meta features, in author-based way if desired
        if type == "author_based":
            train_raw, val_raw = author_based_split(raw_df, val_size=self.val_size, random_state=self.random_state)
        else:
            # simple random split on rows
            train_raw = raw_df.sample(frac=1 - self.val_size, random_state=self.random_state)
            val_raw   = raw_df.drop(train_raw.index).reset_index(drop=True)
            train_raw = train_raw.reset_index(drop=True)

        print("FusionDataModule: label distribution")
        print("  Train:", train_raw["label"].value_counts(normalize=True))
        print("  Val:",   val_raw["label"].value_counts(normalize=True))

        # 5) build meta features using same stats/src2idx as meta tower pretraining
        train_meta_df, _, _ = build_meta_features(train_raw, fit_stats=stats, src2idx=src2idx, K=len(src2idx))
        val_meta_df,   _, _ = build_meta_features(val_raw,   fit_stats=stats, src2idx=src2idx, K=len(src2idx))

        # 6) project to feature_cols and scale
        X_train_meta = scaler.transform(train_meta_df[feature_cols].values.astype(np.float32))
        X_val_meta   = scaler.transform(val_meta_df[feature_cols].values.astype(np.float32))

        # 7) build datasets
        self.train_ds = FusionDataset(train_meta_df, X_train_meta)
        self.val_ds   = FusionDataset(val_meta_df,   X_val_meta)

    def train_class_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=fusion_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=fusion_collate_fn,
        )