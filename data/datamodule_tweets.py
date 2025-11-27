# datamodule_tweets.py
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data import load_dataset, make_loader

import json
from pathlib import Path

class TweetsDataModule:
    def __init__(self, train_path: str, batch_size: int, num_workers: int = 2):
        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_df = None
        self.val_df = None
        self.src2idx = None

    def setup(self, ckpt_path):
        df, src2idx, stats = load_dataset(self.train_path, expect_label=True)
        ckpt_path = Path(ckpt_path)

        map_path  = ckpt_path.with_suffix(".src2idx.json")
        stats_path = ckpt_path.with_suffix(".stats.json")

        with map_path.open("w", encoding="utf-8") as f:
            json.dump(src2idx, f, ensure_ascii=False, indent=2)

        
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        from sklearn.model_selection import train_test_split
        self.train_df, self.val_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42) #TODO: Import those from config
        self.src2idx = src2idx
        self.stats = stats

    @property
    def n_source_buckets(self):
        return 1 + len(self.src2idx.values())

    def train_class_dataloader(self):
        return make_loader(self.train_df, bs=self.batch_size, shuffle=True, is_train=True)

    def val_dataloader(self):
        return make_loader(self.val_df, bs=self.batch_size, shuffle=False, is_train=True)