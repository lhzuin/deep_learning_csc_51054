import pandas as pd
import numpy as np
import re, ast, json
from pathlib import Path

def parse_tweets(path, expect_label=True):
    # Load & flatten
    path = Path(path)  # <- ensure it's a Path object
    #print(">>> parse_tweets path:", path, "exists:", path.exists())  # optional debug
    df = pd.read_json(path, lines=True)
    df = pd.json_normalize(df.to_dict(orient="records"), sep=".")

    # Ensure expected nested columns exist
    for col in [
        "text", "extended_tweet.full_text", "source",
        "entities.hashtags", "entities.user_mentions", "entities.urls",
        "extended_entities.media",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Full text (vectorized, avoids apply/axis=1)
    df["full_text"] = df["extended_tweet.full_text"].fillna(df["text"]).fillna("")

    # Engagement (create if missing)
    for col in ["retweet_count", "favorite_count", "reply_count", "quote_count"]:
        if col not in df.columns:
            df[col] = 0

    # Safe length for list-like fields (sometimes lists, sometimes stringified)
    def safe_len(x):
        if isinstance(x, list):
            return len(x)
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                return len(v) if isinstance(v, (list, tuple)) else 1
            except Exception:
                return 0
        return 0

    df["n_hashtags"] = df["entities.hashtags"].apply(safe_len)
    df["n_mentions"] = df["entities.user_mentions"].apply(safe_len)
    df["n_urls"]     = df["entities.urls"].apply(safe_len)

    # Media presence (avoid .get(...).apply on a scalar)
    df["has_media"] = df["extended_entities.media"].apply(lambda x: safe_len(x) > 0)

    # Source app (extract readable name from HTML anchor)
    def extract_source(x):
        if not isinstance(x, str):
            return "Unknown"
        m = re.search(r'>([^<]+)<', x)
        return m.group(1) if m else x

    df["source_app"] = df["source"].apply(extract_source)

    # User fields (create if missing)
    for col in [
        "user.description", "user.location",
        "user.favourites_count", "user.statuses_count", "user.listed_count"
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df["user.description"] = df["user.description"].fillna("")

    # Keep relevant columns (only those that exist)
    #"lang" -> always french
    keep_cols = [
        "challenge_id",
        "id_str", "full_text", "source_app",
        "n_hashtags", "n_mentions", "has_media",
        "user.description",
        "user.statuses_count", "user.listed_count",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy()

    # Attach label if expected and available
    if expect_label and "label" in df.columns:
        out["label"] = df["label"]
    elif expect_label and "label" not in df.columns:
        print("Warning: 'label' not found in this file; returning features only.")

    # Optional: show which expected columns were missing
    missing = sorted(set(keep_cols) - set(existing))
    if missing:
        print("Note: missing columns created or omitted:", missing)

    return out


def make_transformations(df, src2idx=None, K=15, stats=None):
    # --- text fields ---
    df["user.description"] = df["user.description"].fillna("").astype(str)
    df["full_text"] = df["full_text"].fillna("").astype(str)

    # --- numeric transforms ---
    # clip heavy tails before log1p
    if stats is None:
        # fit mode: compute from this df
        p99_listed = df["user.listed_count"].quantile(0.995)
        p99_status = df["user.statuses_count"].quantile(0.995)
        stats = {
            "p99_listed": float(p99_listed),
            "p99_status": float(p99_status),
        }
    else:
        # transform mode: reuse precomputed stats
        p99_listed = stats["p99_listed"]
        p99_status = stats["p99_status"]

    df["log_listed"]  = np.log1p(np.clip(df["user.listed_count"].fillna(0), 0, p99_listed))
    df["log_status"]  = np.log1p(np.clip(df["user.statuses_count"].fillna(0), 0, p99_status))
    df["n_mentions"]  = df["n_mentions"].fillna(0).astype(int).clip(0, 10)  # small cap helps

    # --- source_app buckets ---

    if src2idx is None:
        top_src = df["source_app"].fillna("Unknown").value_counts().head(K).index.tolist()
        src2idx = {s: i+1 for i, s in enumerate(top_src)}  # 0 reserved for "Other"

    df["source_idx"] = (
        df["source_app"].fillna("Unknown").map(src2idx).fillna(0).astype(int)
    )

    return df, src2idx, stats


def load_dataset(train_path="data/train.jsonl", expect_label=True, src2idx=None, K=15, stats=None):
    dataset = parse_tweets(train_path, expect_label=expect_label)
    df, src2idx, stats = make_transformations(dataset, src2idx=src2idx, K=K, stats=stats)
    return df, src2idx, stats