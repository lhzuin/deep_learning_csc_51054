# datamodule_tweets.py
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data import load_dataset, make_loader, load_desc_dataset
from sklearn.model_selection import train_test_split
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

import json
from pathlib import Path

class TweetsDataModule:
    def __init__(self, train_path: str, batch_size: int, num_workers: int = 2, val_size: float = 0.1, random_state: int = 42):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_df = None
        self.val_df = None
        self.src2idx = None
        self.random_state = random_state
        self.val_size = val_size

    def setup(self, ckpt_path, type = "random"):
        if type == "random":
            self.setup_random(ckpt_path)
        else:
            self.setup_author_based(ckpt_path)

    
    def setup_random(self, ckpt_path):
        df, src2idx, stats = load_dataset(self.train_path, expect_label=True)
        self.src2idx = src2idx
        self.stats = stats
        ckpt_path = Path(ckpt_path)

        map_path  = ckpt_path.with_suffix(".src2idx.json")
        stats_path = ckpt_path.with_suffix(".stats.json")

        with map_path.open("w", encoding="utf-8") as f:
            json.dump(src2idx, f, ensure_ascii=False, indent=2)

        
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        from sklearn.model_selection import train_test_split
        self.train_df, self.val_df = train_test_split(df, test_size=self.val_size, stratify=df["label"], random_state=self.random_state)

    def setup_author_based(self, ckpt_path):
        df, src2idx, stats = load_dataset(self.train_path, expect_label=True)
        self.src2idx = src2idx
        self.stats = stats
        ckpt_path = Path(ckpt_path)

        map_path  = ckpt_path.with_suffix(".src2idx.json")
        stats_path = ckpt_path.with_suffix(".stats.json")

        with map_path.open("w", encoding="utf-8") as f:
            json.dump(src2idx, f, ensure_ascii=False, indent=2)

        
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        from sklearn.model_selection import train_test_split



        user_ids = df["author_pseudo_id"].astype(str)


        unique_users = user_ids.unique()
        train_users, val_users = train_test_split(
            unique_users, test_size=self.val_size, random_state=self.random_state
        )

        train_mask = user_ids.isin(train_users)
        val_mask   = user_ids.isin(val_users)

        self.train_df = df[train_mask].reset_index(drop=True)
        self.val_df   = df[val_mask].reset_index(drop=True)

        print("User-level split:")
        print("  #train tweets:", len(self.train_df))
        print("  #val tweets:  ", len(self.val_df))
        print("  #unique users train:", len(np.unique(user_ids[train_mask])))
        print("  #unique users val:  ", len(np.unique(user_ids[val_mask])))

    @property
    def n_source_buckets(self):
        return 1 + len(self.src2idx.values())

    def train_class_dataloader(self):
        return make_loader(self.train_df, bs=self.batch_size, shuffle=True, is_train=True)

    def val_dataloader(self):
        return make_loader(self.val_df, bs=self.batch_size, shuffle=False, is_train=True)
    

from data import load_desc_dataset  # import from where you just defined it


# --------------------- Dataset -----------------------------

class DescriptionDataset(Dataset):
    """
    Dataset for description-only text:
      expects df with columns:
        - 'user.description'
        - 'label'
    We expose the text under 'full_text' so the model
    can stay unchanged and still read batch["full_text"].
    """
    def __init__(self, df, is_train=True):
        self.texts = df["user.description"].astype(str).tolist()
        self.y = torch.tensor(df["label"].values, dtype=torch.long) if is_train else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"user_desc": self.texts[i]}
        if self.y is not None:
            item["label"] = self.y[i]
        return item


# --------------------- DataModule --------------------------

class DescriptionsDataModule:
    """
    Hydra-compatible datamodule for description-only experiments.

    Same interface as TweetsDataModule:
      - setup(ckpt_path, type="random"|"author_based")
      - train_class_dataloader()
      - val_dataloader()
      - n_source_buckets (dummy=1)
    """
    def __init__(
        self,
        train_path: str,
        batch_size: int,
        num_workers: int = 2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.random_state = random_state

        self.train_df = None
        self.val_df   = None

    # We keep the same signature as TweetsDataModule.setup
    def setup(self, ckpt_path, type="random"):
        if type == "random":
            self._setup_random()
        else:
            self._setup_author_based()

        # we don't need src2idx/stats here, but if you want
        # to drop-in replace TweetsDataModule you can still
        # write some minimal metadata next to ckpt_path
        ckpt_path = Path(ckpt_path)
        meta_path = ckpt_path.with_suffix(".desc_meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_train": len(self.train_df),
                    "n_val": len(self.val_df),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _setup_random(self):
        # For random split we can dedupe by description
        df = load_desc_dataset(self.train_path, expect_label=True, dedupe_by_desc=True)

        self.train_df, self.val_df = train_test_split(
            df,
            test_size=self.val_size,
            stratify=df["label"],
            random_state=self.random_state,
        )

    def _setup_author_based(self):
        # For author-based split: keep all rows (no dedupe!)
        df = load_desc_dataset(self.train_path, expect_label=True, dedupe_by_desc=False)

        user_ids = df["author_pseudo_id"].astype(str)
        unique_users = user_ids.unique()

        train_users, val_users = train_test_split(
            unique_users,
            test_size=self.val_size,
            random_state=self.random_state,
        )

        train_mask = user_ids.isin(train_users)
        val_mask   = user_ids.isin(val_users)

        self.train_df = df[train_mask].reset_index(drop=True)
        self.val_df   = df[val_mask].reset_index(drop=True)

        print("User-level split (descriptions only):")
        print("  #train examples:", len(self.train_df))
        print("  #val examples:  ", len(self.val_df))
        print("  #unique users train:", len(np.unique(user_ids[train_mask])))
        print("  #unique users val:  ", len(np.unique(user_ids[val_mask])))

    # --------- Interface expected by Trainer -----------------

    @property
    def n_source_buckets(self):
        # text-only model does not use this, but Trainer passes it.
        return 1

    def train_class_dataloader(self):
        return DataLoader(
            DescriptionDataset(self.train_df, is_train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            DescriptionDataset(self.val_df, is_train=True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )