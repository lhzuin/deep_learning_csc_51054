# models/multitower_fusion.py
import torch
import torch.nn as nn

from models.metadata_tower import MetaTower
from models.tweet_tower import InfluencerTextOnly
from models.description_tower import DescriptionTower


class MultiTowerFusion(nn.Module):
    """
    Late-fusion model:
      - MetaTower (metadata -> h_meta)
      - InfluencerTextOnly (tweet text -> h_tweet)
      - DescriptionTowerDistilcamembert (user description -> h_desc)

    Expects batch dict with keys:
      - "full_text": list[str]
      - "user_desc": list[str]
      - "meta": FloatTensor (B, meta_in_dim)  # already scaled
      - "label": LongTensor (B,)              # used by Trainer, not this class
    """

    def __init__(
        self,
        # --- paths to trained checkpoints ---
        meta_ckpt_path: str,
        tweet_ckpt_path: str,
        desc_ckpt_path: str,

        # --- meta tower architecture (must match how it was trained) ---
        meta_in_dim: int,
        meta_hidden_dim: int = 32,
        meta_out_dim: int = 16,
        meta_dropout: float = 0.0,   # you used 0.0 in best_config

        # --- tweet tower architecture (must match how it was trained) ---
        tweet_base_model: str = "cmarkea/distilcamembert-base",
        tweet_head_hidden_dim: int = 128,
        tweet_head_dropout: float = 0.1,
        tweet_max_len: int = 128,
        tweet_has_lora: bool = True,

        # --- description tower architecture (must match how it was trained) ---
        desc_base_model: str = "cmarkea/distilcamembert-base",
        desc_head_hidden_dim: int = 128,
        desc_head_dropout: float = 0.1,
        desc_max_len: int = 128,
        desc_has_lora: bool = True,

        # --- fusion head ---
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.1,

        # --- freezing ---
        freeze_towers: bool = True,
    ):
        super().__init__()
        self._param_groups_cache = None

        # 1) Rebuild + load MetaTower from *full* checkpoint
        meta_ckpt = torch.load(meta_ckpt_path, map_location="cpu", weights_only=False)
        self.meta_feature_cols = meta_ckpt["feature_cols"]   # useful for datamodule
        self.meta_stats        = meta_ckpt["stats"]
        self.meta_src2idx      = meta_ckpt["src2idx"]
        # scaler will be used in the datamodule, not here

        self.meta_tower = MetaTower(
            in_dim=meta_in_dim,
            hidden_dim=meta_hidden_dim,
            out_dim=meta_out_dim,
            dropout=meta_dropout,
        )
        self.meta_tower.load_state_dict(meta_ckpt["model_state_dict"])
        

        # 2) Rebuild + load tweet tower (plain state_dict)
        self.tweet_tower = InfluencerTextOnly(
            base_model=tweet_base_model,
            head_hidden_dim=tweet_head_hidden_dim,
            head_dropout=tweet_head_dropout,
            max_len=tweet_max_len,
            has_lora=tweet_has_lora,
        )
        self.tweet_tower.load_state_dict(torch.load(tweet_ckpt_path, map_location="cpu"))

        # 3) Rebuild + load description tower (plain state_dict)
        self.desc_tower = DescriptionTower(
            base_model=desc_base_model,
            head_hidden_dim=desc_head_hidden_dim,
            head_dropout=desc_head_dropout,
            max_len=desc_max_len,
            has_lora=desc_has_lora,
        )
        self.desc_tower.load_state_dict(torch.load(desc_ckpt_path, map_location="cpu"))

        # Optionally freeze towers (only train fusion head)
        if freeze_towers:
            for p in self.meta_tower.parameters():
                p.requires_grad = False
            for p in self.tweet_tower.parameters():
                p.requires_grad = False
            for p in self.desc_tower.parameters():
                p.requires_grad = False

        # total fusion input dim = concatenation of the hidden states of each tower
        fusion_in_dim = meta_out_dim + tweet_head_hidden_dim + desc_head_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 2),
            nn.LogSoftmax(dim=1),  # (B,2) log-probs for NLLLoss
        )

    # -------- plumbing utilities --------
    def _dev(self):
        return next(self.parameters()).device

    def forward(self, batch, return_parts=False):
        """
        batch must contain:
          - full_text: list[str]
          - user_desc: list[str]
          - meta: FloatTensor (B, meta_in_dim)
        """
        device = self._dev()

        # ---- Meta tower ----
        meta_x = batch["meta"].to(device)  # (B, meta_in_dim)
        h_meta, _ = self.meta_tower(meta_x, return_logits=True)   # (B, meta_out_dim)

        # ---- Tweet tower ----
        logp_tweet, h_tweet = self.tweet_tower(batch, return_logits=True)

        # ---- Description tower ----
        logp_desc, h_desc = self.desc_tower(batch, return_logits=True)

        # ---- Fusion ----
        fused = torch.cat([h_meta, h_tweet, h_desc], dim=1)
        logp_fused = self.fusion_head(fused)  # (B,2)

        if return_parts:
            return {
                "logp_fused": logp_fused,
                "h_meta": h_meta,
                "h_tweet": h_tweet,
                "h_desc": h_desc,
                "logp_tweet": logp_tweet,
                "logp_desc": logp_desc,
            }

        return logp_fused

    # -------- param groups API for your Trainer --------
    def _build_param_groups(self):
        # Only fusion head is trainable group "head"
        groups = {
            "head": list(self.fusion_head.parameters()),
            # if later you want to fine-tune towers, you can add more groups here
        }
        return groups

    def get_param_groups(self):
        if self._param_groups_cache is None:
            self._param_groups_cache = self._build_param_groups()
        return self._param_groups_cache

    def set_trainable_groups(self, group_names):
        """
        For compatibility with your Trainer/config-driven freezing.
        """
        groups = self.get_param_groups()

        # default: freeze everything
        for p in self.parameters():
            p.requires_grad = False

        for g in group_names:
            assert g in groups, f"Unknown param group: {g}"
            for p in groups[g]:
                p.requires_grad = True