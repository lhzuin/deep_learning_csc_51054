import torch, torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModel, CamembertTokenizer

#TODO: add a temp head for the text only stage

class InfluencerCamembertV2(nn.Module):
    def __init__(
        self,
        n_source_buckets: int,       # = 1 + len(top_src)  (0="Other")
        source_emb_dim: int = 16,
        num_proj_dim: int = 16,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.15,
        max_len_tweet: int = 128,
        max_len_desc: int = 96,
        meta_dropout: float = 0.0,
    ):
        super().__init__()
        self._param_groups_cache = None
        self.max_len_tweet = max_len_tweet
        self.max_len_desc  = max_len_desc

        base_model = "almanach/camembertv2-base"

        # ---------- Tweet text encoder (+LoRA adaptors) ----------
        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True) 
        self.tweet_enc = AutoModel.from_pretrained(base_model)
        self.desc_enc = AutoModel.from_pretrained(base_model)


        text_lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "key", "value"], lora_dropout=0.05,
            bias="none", task_type="FEATURE_EXTRACTION"
        )

        self.tweet_enc = get_peft_model(self.tweet_enc, text_lora_cfg)
        self.desc_enc = get_peft_model(self.desc_enc, text_lora_cfg)
        
        

        bert_dim = self.tweet_enc.config.hidden_size
        self.tweet_proj = nn.Linear(bert_dim, bert_dim)   

        # ---------- User description encoder (+LoRA) ----------
        self.desc_proj = nn.Linear(bert_dim, bert_dim)

        # ---------- Categorical: source_app (top-15 + other) ----------
        self.source_emb = nn.Embedding(n_source_buckets, source_emb_dim)

        # ---------- Numeric/Boolean ----------
        meta_in_dim = 9
        self.num_proj = nn.Sequential(
            nn.Linear(meta_in_dim, num_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(num_proj_dim),
        )

        self.meta_dropout = nn.Dropout(meta_dropout)

        # ---------- Head ----------
        joint_dim = (bert_dim   # tweet
                    + bert_dim  # desc
                    + source_emb_dim
                    + num_proj_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Dropout(head_dropout),
            nn.Linear(joint_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 2),
            nn.LogSoftmax(dim=1),
        )

        self.loss = nn.NLLLoss()
    
    def _dev(self):
        return next(self.parameters()).device

    # utility encoders (CLS pooling)
    def _encode_text(self, adapter_name: str, proj, texts, max_len):
        # choose which LoRA adapter to use
        self.text_enc.set_adapter(adapter_name)
        tok = self.tok(
            texts, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        ).to(self._dev())
        out = self.text_enc(**tok).last_hidden_state[:, 0]  # [CLS]
        return proj(out)

    def forward(self, batch):
        device = self._dev()
        # Text

        tok_tweet = self.tok(
            batch["full_text"],          # list of strings
            padding=True,
            truncation=True,
            max_length=self.max_len_tweet,
            return_tensors="pt"
        ).to(self._dev())

        out_tweet = self.tweet_enc(**tok_tweet).last_hidden_state[:, 0]
        t_f = self.tweet_proj(out_tweet)

        tok_desc = self.tok(
            batch["user_desc"],          # list of strings
            padding=True,
            truncation=True,
            max_length=self.max_len_desc,
            return_tensors="pt"
        ).to(self._dev())

        out_desc = self.desc_enc(**tok_desc).last_hidden_state[:, 0]
        d_f = self.desc_proj(out_desc)

        # Categorical
        source_idx = batch["source_idx"].to(device)            # LongTensor [B]
        s_f = self.source_emb(source_idx)                      # [B, source_emb_dim]

        # Numeric
        num = torch.stack([
            batch["log_listed"].to(device).view(-1),
            batch["log_status"].to(device).view(-1),
            batch["log_fav"].to(device).view(-1),
            batch["n_mentions"].to(device).float().view(-1),
            batch["n_hashtags"].to(device).float().view(-1),
            batch["has_media"].to(device).float().view(-1),
            batch["is_reply"].to(device).float().view(-1),
            batch["user_default_profile"].to(device).float().view(-1),
            batch["user_geo_enabled"].to(device).float().view(-1),
        ], dim=1)  # [B, 9]

        n_f = self.num_proj(num)

        meta = torch.cat([s_f, n_f], dim=-1)
        meta = self.meta_dropout(meta) 

        joint = torch.cat([t_f, d_f, meta], dim=-1)       # [B, joint_dim]
        logp = self.head(joint)                               # [B, 2]
        return logp


    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = self._build_param_groups()
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        """
        Generic, config-driven freezing:
        - group_names: list of strings, subset of keys in get_param_groups().
        """
        groups = self.get_param_groups()

        # default: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # enable only requested groups
        for g in group_names:
            assert g in groups, f"Unknown param group: {g}"
            for p in groups[g]:
                p.requires_grad = True
    
    def _build_param_groups(self):
        """
        Build a dict group_name -> list[Parameter].
        Called once and cached.
        """
        groups = {
            "head":       list(self.head.parameters()),
            "tweet_proj": list(self.tweet_proj.parameters()),
            "desc_proj":  list(self.desc_proj.parameters()),
            "source_emb": list(self.source_emb.parameters()),
            "num_proj":   list(self.num_proj.parameters()),
            "text_lora_all": [],
            "text_lora_tweet": [],
            "text_lora_desc": [],
            "enc_tweet_last2": [],
            "enc_desc_last2": [],
        }

        n_layers = self.tweet_enc.config.num_hidden_layers
        last_two = {n_layers - 1, n_layers - 2}

        # LoRA params live inside self.text_enc with names containing 'lora_'
        for name, p in self.tweet_enc.named_parameters():
            if "lora_" in name:
                groups["text_lora_all"].append(p)
                groups["text_lora_tweet"].append(p)
                continue
            if any(f"layer.{i}." in name for i in last_two):
                groups["enc_tweet_last2"].append(p)

        for name, p in self.desc_enc.named_parameters():
            if "lora_" in name:
                groups["text_lora_all"].append(p)
                groups["text_lora_desc"].append(p)
                continue
            if any(f"layer.{i}." in name for i in last_two):
                groups["enc_desc_last2"].append(p)

        return groups

