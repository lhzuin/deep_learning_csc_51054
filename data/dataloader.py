from torch.utils.data import Dataset, DataLoader
import torch

class TweetDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.ftxt  = df["full_text"].tolist()
        self.udesc = df["user.description"].tolist()
        self.src   = torch.tensor(df["source_idx"].values, dtype=torch.long)
        self.l_l   = torch.tensor(df["log_listed"].values, dtype=torch.float32)
        self.l_s   = torch.tensor(df["log_status"].values, dtype=torch.float32)
        self.nm    = torch.tensor(df["n_mentions"].values, dtype=torch.int64)
        self.y     = torch.tensor(df["label"].values, dtype=torch.long) if is_train else None

    def __len__(self): 
        return len(self.ftxt)

    def __getitem__(self, i):
        item = {
            "full_text":  self.ftxt[i],
            "user_desc":  self.udesc[i],
            "source_idx": self.src[i],
            "log_listed": self.l_l[i],
            "log_status": self.l_s[i],
            "n_mentions": self.nm[i],
        }
        if self.y is not None:
            item["label"] = self.y[i]
        return item

def make_loader(df, bs=32, shuffle=True, is_train=True):
    return DataLoader(TweetDataset(df, is_train=is_train), batch_size=bs, shuffle=shuffle)



