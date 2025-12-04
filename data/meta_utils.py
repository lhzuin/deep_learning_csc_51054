import numpy as np
import pandas as pd
import re, ast, hashlib
from sklearn.model_selection import train_test_split
from pathlib import Path


def parse_tweets_meta(path, expect_label=True):
    """
    Parse the raw train.jsonl and build a flat DataFrame with:
    - author_pseudo_id
    - basic tweet/user fields
    - simple structural flags (is_reply, has_media, etc.)
    """
    path = Path(path)
    df = pd.read_json(path, lines=True)
    df = pd.json_normalize(df.to_dict(orient="records"), sep=".")

    # Ensure some nested columns exist
    for col in [
        "text", "extended_tweet.full_text", "source",
        "entities.hashtags", "entities.user_mentions", "entities.urls",
        "extended_entities.media",
        "user.created_at", "user.description", "user.url", "user.location",
        "user.favourites_count", "user.statuses_count", "user.listed_count",
        "user.default_profile", "user.geo_enabled",
        "in_reply_to_status_id", "in_reply_to_user_id",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # -------- full text ----------
    df["full_text"] = df["extended_tweet.full_text"].fillna(df["text"]).fillna("")
    df["text_len"] = df["full_text"].str.len()

    # -------- counts / list lengths ----------
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

    # media flag
    df["has_media"] = df["extended_entities.media"].apply(lambda x: safe_len(x) > 0)

    # -------- author_pseudo_id ----------
    def make_user_key(row):
        key = (
            str(row.get("user.created_at", "")) + "|" +
            str(row.get("user.description", "")) + "|" +
            str(row.get("user.url", "")) + "|" +
            str(row.get("user.location", ""))
        )
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    df["author_pseudo_id"] = df.apply(make_user_key, axis=1)

    # -------- structural flags ----------
    df["is_reply"] = (
        df["in_reply_to_status_id"].notna() |
        df["in_reply_to_user_id"].notna()
    )

    # approximate retweet flag (not used here, but harmless)
    df["is_retweet"] = df["text"].fillna("").str.startswith("RT @")

    # ensure boolean columns exist (even if missing)
    for col in ["user.default_profile", "user.geo_enabled"]:
        if col not in df.columns:
            df[col] = np.nan

    # -------- source_app (HTML → readable name) ----------
    def extract_source(x):
        if not isinstance(x, str):
            return "Unknown"
        m = re.search(r'>([^<]+)<', x)
        return m.group(1) if m else x

    df["source_app"] = df["source"].apply(extract_source)

    # keep only metadata we care about:
    keep_cols = [
        "author_pseudo_id", "full_text",
        "challenge_id" if "challenge_id" in df.columns else None,
        "n_hashtags", "n_mentions", "n_urls", "has_media",
        "user.favourites_count", "user.statuses_count", "user.listed_count",
        "user.default_profile", "user.geo_enabled",
        "is_reply", "text_len", "source_app",
    ]
    keep_cols = [c for c in keep_cols if c is not None]

    out = df[keep_cols].copy()

    # attach label if present
    if expect_label and "label" in df.columns:
        out["label"] = df["label"]
    elif expect_label and "label" not in df.columns:
        print("Warning: 'label' not found in this file; returning features only.")

    return out



def build_meta_features(df, fit_stats=None, src2idx=None, K=15):
    """
    Build metadata features:
      - log_status (from user.statuses_count)
      - log_listed (from user.listed_count)
      - log_fav   (from user.favourites_count)
      - n_mentions, n_hashtags
      - booleans as 0/1
      - source_idx: bucketized from source_app with top-K on TRAIN
    """
    df = df.copy()

    # ensure numeric columns exist
    for col in ["user.statuses_count", "user.listed_count", "user.favourites_count"]:
        if col not in df.columns:
            df[col] = 0

    # fit_stats = dict with p99s, learned from train set only
    if fit_stats is None:
        fit_stats = {}
        for col in ["user.statuses_count", "user.listed_count", "user.favourites_count"]:
            fit_stats[f"{col}_p99"] = float(df[col].quantile(0.995))

    # log transforms with clipping
    df["log_status"] = np.log1p(
        np.clip(df["user.statuses_count"].fillna(0), 0, fit_stats["user.statuses_count_p99"])
    )
    df["log_listed"] = np.log1p(
        np.clip(df["user.listed_count"].fillna(0), 0, fit_stats["user.listed_count_p99"])
    )
    df["log_fav"] = np.log1p(
        np.clip(df["user.favourites_count"].fillna(0), 0, fit_stats["user.favourites_count_p99"])
    )

    # counts
    df["n_mentions"] = df["n_mentions"].fillna(0).astype(int)
    df["n_hashtags"] = df["n_hashtags"].fillna(0).astype(int)

    # booleans → 0/1
    for bcol in ["has_media", "is_reply", "user.default_profile", "user.geo_enabled"]:
        if bcol not in df.columns:
            df[bcol] = False
        df[bcol] = df[bcol].fillna(False).astype(int)

    # ---- source_idx from source_app ----
    if "source_app" not in df.columns:
        df["source_app"] = "Unknown"

    if src2idx is None:
        top_src = df["source_app"].fillna("Unknown").value_counts().head(K).index.tolist()
        src2idx = {s: i + 1 for i, s in enumerate(top_src)}  # 0 reserved for "Other"

    df["source_idx"] = (
        df["source_app"]
        .fillna("Unknown")
        .map(src2idx)
        .fillna(0)
        .astype(int)
    )

    return df, fit_stats, src2idx



def author_based_split(df, val_size=0.1, random_state=42):
    """
    Split df into train/val such that authors (author_pseudo_id)
    do not overlap between train and val.
    """
    assert "author_pseudo_id" in df.columns, "author_pseudo_id column missing"

    user_ids = df["author_pseudo_id"].astype(str)
    unique_users = user_ids.unique()

    train_users, val_users = train_test_split(
        unique_users, test_size=val_size, random_state=random_state
    )

    train_mask = user_ids.isin(train_users)
    val_mask   = user_ids.isin(val_users)

    train_df = df[train_mask].reset_index(drop=True)
    val_df   = df[val_mask].reset_index(drop=True)

    print("User-level split:")
    print("  #train tweets:", len(train_df))
    print("  #val tweets:  ", len(val_df))
    print("  #unique users train:", len(np.unique(user_ids[train_mask])))
    print("  #unique users val:  ", len(np.unique(user_ids[val_mask])))

    return train_df, val_df