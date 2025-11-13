import torch, torch.nn as nn
import transformers
from peft import get_peft_model, LoraConfig

#TODO: add a temp head for the text only stage

class InfluencerDistilBert(nn.Module):
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

        distilbert = "distilbert-base-uncased"

        # ---------- Tweet text encoder (+LoRA) ----------
        self.tok = transformers.DistilBertTokenizerFast.from_pretrained(distilbert)
        self.tweet_enc = transformers.DistilBertModel.from_pretrained(distilbert)
        text_lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_lin","v_lin"], lora_dropout=0.05,
            bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.tweet_enc = get_peft_model(self.tweet_enc, text_lora_cfg)
        for p in self.tweet_enc.base_model.parameters():
            p.requires_grad = False
        bert_dim = self.tweet_enc.config.hidden_size
        self.tweet_proj = nn.Linear(bert_dim, bert_dim)   # (simple proj; keeps dims)

        # ---------- User description encoder (+LoRA) ----------
        self.desc_enc = transformers.DistilBertModel.from_pretrained(distilbert)
        self.desc_enc = get_peft_model(self.desc_enc, text_lora_cfg)
        for p in self.desc_enc.base_model.parameters():
            p.requires_grad = False
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
    def _encode_text(self, encoder, proj, texts, max_len):
        tok = self.tok(
            texts, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt"
        ).to(self._dev())
        out = encoder(**tok).last_hidden_state[:, 0]  # [CLS]
        return proj(out)

    def forward(self, batch):
        device = self._dev()
        # Text
        t_f = self._encode_text(self.tweet_enc, self.tweet_proj, batch["full_text"], self.max_len_tweet)
        d_f = self._encode_text(self.desc_enc,  self.desc_proj,  batch["user_desc"],  self.max_len_desc)

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
        'text_only': train LoRA adapters (tweet+desc) + head; freeze numeric/source parts
        'full': freeze text encoders+adapters; train head + numeric + source emb (+ projections)
        """
        assert stage in {"text_only", "full", "mlp_only"}

        # default: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        if stage == "text_only":
            # train LoRA adapters & projections & head
            for m in [self.tweet_enc, self.desc_enc]:
                for n, p in m.named_parameters():
                    if "lora_" in n:     # only LoRA params
                        p.requires_grad = True
            for m in [self.tweet_proj, self.desc_proj, self.head]:
                for p in m.parameters():
                    p.requires_grad = True
        elif stage == "mlp_only":
            # Encoders (incl. LoRA) stay frozen.
            # We only train the cheap stuff on top.
            for m in [self.source_emb, self.num_proj, self.head,
                      self.tweet_proj, self.desc_proj]:
                for p in m.parameters():
                    p.requires_grad = True
        else:  # 'full'
            # 1) LoRA adapters in both text encoders
            for m in [self.tweet_enc, self.desc_enc]:
                for n, p in m.named_parameters():
                    if "lora_" in n:   # only LoRA params
                        p.requires_grad = True

            # 2) All shallow layers
            for m in [self.source_emb, self.num_proj, self.head,
                      self.tweet_proj, self.desc_proj]:
                for p in m.parameters():
                    p.requires_grad = True