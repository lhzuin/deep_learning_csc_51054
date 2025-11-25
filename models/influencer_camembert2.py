import torch, torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModel

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
    ):
        super().__init__()
        self.max_len_tweet = max_len_tweet
        self.max_len_desc  = max_len_desc

        base_model = "cmarkea/distilcamembert-base"

        # ---------- Tweet text encoder (+LoRA adaptors) ----------
        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.base_enc = AutoModel.from_pretrained(base_model)
        text_lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.05,
            bias="none", task_type="FEATURE_EXTRACTION"
        )

        self.text_enc = get_peft_model(self.base_enc, text_lora_cfg)
        # By default the adapter is usually called "default"
        # Let’s just treat it as our "tweet" adapter:
        self.tweet_adapter = "default"

        # Add a second adapter for user descriptions
        self.desc_adapter = "desc"
        self.text_enc.add_adapter(self.desc_adapter, text_lora_cfg)
        # Freeze shared base model weights; only LoRA params train
        for p in self.text_enc.base_model.parameters():
            p.requires_grad = False

        bert_dim = self.text_enc.config.hidden_size
        self.tweet_proj = nn.Linear(bert_dim, bert_dim)   # (simple proj; keeps dims)

        # ---------- User description encoder (+LoRA) ----------
        self.desc_proj = nn.Linear(bert_dim, bert_dim)

        # ---------- Categorical: source_app (top-15 + other) ----------
        self.source_emb = nn.Embedding(n_source_buckets, source_emb_dim)

        # ---------- Numeric: [log_listed, log_status, n_mentions] ----------
        self.num_proj = nn.Sequential(
            nn.Linear(3, num_proj_dim),
            nn.ReLU(),
            nn.LayerNorm(num_proj_dim),
        )

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
        t_f = self._encode_text(
            self.tweet_adapter,
            self.tweet_proj,
            batch["full_text"],
            self.max_len_tweet,
        )
        d_f = self._encode_text(
            self.desc_adapter,
            self.desc_proj,
            batch["user_desc"],
            self.max_len_desc,
        )

        # Categorical
        source_idx = batch["source_idx"].to(device)            # LongTensor [B]
        s_f = self.source_emb(source_idx)                      # [B, source_emb_dim]

        # Numeric
        num = torch.stack([
            batch["log_listed"].to(device).view(-1),
            batch["log_status"].to(device).view(-1),
            batch["n_mentions"].to(device).float().view(-1),
        ], dim=1)                                             # [B, 3]
        n_f = self.num_proj(num)

        joint = torch.cat([t_f, d_f, s_f, n_f], dim=-1)       # [B, joint_dim]
        logp = self.head(joint)                               # [B, 2]
        return logp

    # --------- stage toggles ---------
    def set_stage(self, stage: str):
        """
        'text_only': train LoRA adapters (tweet+desc) + projections + head
        'mlp_only': freeze text encoders+adapters; train only head + shallow stuff
        'full':     train LoRA + head + numeric + source emb (+ projections)
        """
        assert stage in {"text_only", "full", "mlp_only"}

        # default: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        if stage == "text_only":
            # all LoRA params in shared encoder (both adapters live here)
            for n, p in self.text_enc.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

            # projections + head
            for m in [self.tweet_proj, self.desc_proj, self.head]:
                for p in m.parameters():
                    p.requires_grad = True

        elif stage == "mlp_only":
            for m in [self.source_emb, self.num_proj, self.head,
                      self.tweet_proj, self.desc_proj]:
                for p in m.parameters():
                    p.requires_grad = True

        else:  # 'full'
            # LoRA adapters
            for n, p in self.text_enc.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

            # shallow layers
            for m in [self.source_emb, self.num_proj, self.head,
                      self.tweet_proj, self.desc_proj]:
                for p in m.parameters():
                    p.requires_grad = True