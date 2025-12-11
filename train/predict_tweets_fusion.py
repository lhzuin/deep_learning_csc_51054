# predict_fusion.py

import json
from pathlib import Path

import hydra
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data.datamodule_fusion import FusionDataset, fusion_collate_fn
from data.meta_utils import parse_tweets_meta, build_meta_features

OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)


@hydra.main(config_path="../configs", config_name="fusion_v22", version_base="1.1")
def predict(cfg):
    # ----------------- device -----------------
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    acc_threshold = float(getattr(cfg, "acc_threshold", 0.5))
    print(f"Using decision threshold: {acc_threshold}")

    # ----------------- fusion checkpoint -----------------
    ckpt_path = Path(cfg.checkpoint_load_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}")

    # ----------------- meta tower checkpoint -----------------
    # we reuse scaler, feature_cols, stats, src2idx from meta_tower_best.pt
    meta_ckpt_path = Path(cfg.model.instance.meta_ckpt_path)
    if not meta_ckpt_path.exists():
        raise FileNotFoundError(f"Meta tower checkpoint not found: {meta_ckpt_path}")

    # Important: weights_only=False because the file contains sklearn objects
    meta_ckpt = torch.load(meta_ckpt_path, map_location="cpu", weights_only=False)
    scaler       = meta_ckpt["scaler"]
    feature_cols = meta_ckpt["feature_cols"]
    stats        = meta_ckpt["stats"]
    src2idx      = meta_ckpt["src2idx"]

    # ----------------- load test data -----------------
    test_path = Path(cfg.predict_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # parse_tweets_meta builds full_text, source_app, etc.
    # expect_label=False because test has no labels
    raw_df = parse_tweets_meta(str(test_path), expect_label=False)

    # build user_desc column from user.description (assuming you added it to keep_cols)
    if "user.description" in raw_df.columns:
        raw_df["user_desc"] = raw_df["user.description"].fillna("")
    else:
        # fallback, shouldn't happen if keep_cols is fixed
        raw_df["user_desc"] = ""

    # ----------------- build meta features for test -----------------
    # reuse the same stats/src2idx as the meta tower training
    meta_df, _, _ = build_meta_features(
        raw_df,
        fit_stats=stats,
        src2idx=src2idx,
        K=len(src2idx),
    )

    # project to feature_cols and scale with stored scaler
    X_meta = scaler.transform(meta_df[feature_cols].values.astype(np.float32))

    # FusionDataset expects a 'label' column, but we don't need it for prediction.
    # Just create a dummy one so we can reuse the same class.
    meta_df = meta_df.reset_index(drop=True)
    meta_df["label"] = 0  # dummy labels

    # ----------------- build test loader -----------------
    batch_size   = cfg.datamodule.batch_size
    num_workers  = cfg.datamodule.num_workers

    test_ds = FusionDataset(meta_df, X_meta)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fusion_collate_fn,
    )

    # ----------------- instantiate fusion model -----------------
    # MultiTowerFusion will itself reload the three towers from the paths
    # provided in cfg.model.instance.*
    model = hydra.utils.instantiate(cfg.model.instance).to(device)

    # load trained fusion head weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ----------------- run prediction -----------------
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch["meta"] = batch["meta"].to(device)
            logp = model(batch)  # (B, 2) log-probs or logits

            probs = torch.softmax(logp, dim=1)[:, 1]  # (B,)

            preds = (probs >= acc_threshold).long()   # (B,) in {0,1}

            all_preds.append(preds.cpu())

    if len(all_preds) == 0:
        raise RuntimeError("No predictions produced (empty test_loader?)")

    all_preds = torch.cat(all_preds).numpy()  # shape [N]

    # ----------------- build ID column -----------------
    # 'challenge_id' should be present in meta_df (coming from parse_tweets_meta)
    if "challenge_id" not in meta_df.columns:
        raise KeyError("challenge_id column not found in meta_df; "
                       "ensure your test jsonl has 'challenge_id' and "
                       "parse_tweets_meta keeps it.")

    ids = meta_df["challenge_id"].values

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