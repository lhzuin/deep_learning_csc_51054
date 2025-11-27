# predict_tweets.py
import json
from pathlib import Path

import hydra
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data import load_dataset, make_loader


OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)


@hydra.main(config_path="../configs", config_name="train_v14_distilcamembert", version_base="1.1")
def predict(cfg):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    ckpt_path = Path(cfg.checkpoint_load_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- load src2idx mapping saved during training ----
    map_path = ckpt_path.with_suffix(".src2idx.json")
    if not map_path.exists():
        raise FileNotFoundError(f"src2idx mapping not found: {map_path}")

    with map_path.open("r", encoding="utf-8") as f:
        src2idx = json.load(f)
    stats_path = ckpt_path.with_suffix(".stats.json")
    if not stats_path.exists():
        raise FileNotFoundError(f"stats file not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    # ---- load and preprocess test data with SAME mapping ----
    test_path = Path(cfg.predict_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # expect_label=False -> we don't need a 'label' column
    df_test, _, _ = load_dataset(
        train_path=str(test_path),
        expect_label=False,
        src2idx=src2idx,
        stats=stats,
    )

    # DataLoader with is_train=False → TweetDataset does NOT expect labels
    batch_size = cfg.datamodule.batch_size
    test_loader = make_loader(df_test, bs=batch_size, shuffle=False, is_train=False)

    # ---- instantiate model with correct n_source_buckets ----
    n_source_buckets = 1 + len(src2idx.values())
    model = hydra.utils.instantiate(
        cfg.model.instance,
        n_source_buckets=n_source_buckets
    ).to(device)

    # load trained weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- run prediction ----
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # move tensors to device
            batch["source_idx"] = batch["source_idx"].to(device)
            batch["log_listed"] = batch["log_listed"].to(device)
            batch["log_status"] = batch["log_status"].to(device)
            batch["n_mentions"] = batch["n_mentions"].to(device)

            # text fields stay as Python lists of strings;
            # forward() will tokenize on the fly.

            logp = model(batch)              # [B, 2] log-probs
            preds = logp.argmax(dim=1)       # [B] in {0,1}
            all_preds.append(preds.cpu())

    if len(all_preds) == 0:
        raise RuntimeError("No predictions produced (empty test_loader?)")

    all_preds = torch.cat(all_preds).numpy()  # shape [N]

    # ---- build ID column ----
    # If you want tweet ids:
    #   ids = df_test["id_str"].values
    # If you want row index (0,1,2,...) like in your example:
    ids = df_test["challenge_id"].values

    out_df = pd.DataFrame({
        "ID": ids,
        "Prediction": all_preds.astype(int),
    })

    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions to: {out_path.resolve()}")


if __name__ == "__main__":
    predict()