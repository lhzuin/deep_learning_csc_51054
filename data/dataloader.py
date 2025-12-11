from torch.utils.data import Dataset, DataLoader
import torch

class TweetDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.ftxt  = df["full_text"].tolist()
        self.udesc = df["user.description"].fillna("").astype(str).tolist()

        # categorical
        self.src   = torch.tensor(df["source_idx"].values, dtype=torch.long)

        # numeric / boolean metadata
        self.l_l       = torch.tensor(df["log_listed"].values, dtype=torch.float32)
        self.l_s       = torch.tensor(df["log_status"].values, dtype=torch.float32)
        self.l_fav     = torch.tensor(df["log_fav"].values, dtype=torch.float32)
        self.nm        = torch.tensor(df["n_mentions"].values, dtype=torch.float32)
        self.nh        = torch.tensor(df["n_hashtags"].values, dtype=torch.float32)
        self.has_media = torch.tensor(df["has_media"].values, dtype=torch.float32)
        self.is_reply  = torch.tensor(df["is_reply"].values, dtype=torch.float32)
        self.def_prof  = torch.tensor(df["user.default_profile"].values, dtype=torch.float32)
        self.geo_en    = torch.tensor(df["user.geo_enabled"].values, dtype=torch.float32)

        self.y = torch.tensor(df["label"].values, dtype=torch.long) if is_train else None

    def __len__(self): 
        return len(self.ftxt)

    def __getitem__(self, i):
        item = {
            "full_text":  self.ftxt[i],
            "user_desc":  self.udesc[i],
            "source_idx": self.src[i],

            # numeric/boolean meta fields
            "log_listed":           self.l_l[i],
            "log_status":           self.l_s[i],
            "log_fav":              self.l_fav[i],
            "n_mentions":           self.nm[i],
            "n_hashtags":           self.nh[i],
            "has_media":            self.has_media[i],
            "is_reply":             self.is_reply[i],
            "user_default_profile": self.def_prof[i],
            "user_geo_enabled":     self.geo_en[i],
        }
        if self.y is not None:
            item["label"] = self.y[i]
        return item

def make_loader(df, bs=32, shuffle=True, is_train=True):
    return DataLoader(TweetDataset(df, is_train=is_train), batch_size=bs, shuffle=shuffle)



