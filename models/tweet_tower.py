import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, CamembertTokenizer
from peft import get_peft_model, LoraConfig


class InfluencerTextOnly(nn.Module):
    def __init__(
        self,
        base_model: str = "almanach/camembertv2-base" ,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.15,
        max_len: int = 128,
        has_lora: bool = True,
    ):
        super().__init__()
        self._param_groups_cache = None
        if base_model == "cmarkea/distilcamembert-base":
            self.tok = CamembertTokenizer.from_pretrained(base_model)
        else:
            self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True) 
        self.enc = AutoModel.from_pretrained(base_model)
        text_lora_cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["query", "key","value"], lora_dropout=0.05,
            bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.has_lora = has_lora
        if has_lora:
            self.enc = get_peft_model(self.enc, text_lora_cfg)
        dim = self.enc.config.hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(head_dropout),
            nn.Linear(dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(head_hidden_dim, 2),
            nn.LogSoftmax(dim=1),
        )

        self.max_len = max_len

    def _dev(self):
        return next(self.parameters()).device

    def forward(self, batch, return_logits=False):
        tok = self.tok(
            batch["full_text"],          # list of strings
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(self._dev())
        out = self.enc(**tok).last_hidden_state[:, 0]  # CLS / first token
        logits = self.mlp(out)
        out = self.classifier(logits)  # (N, 2) log-probs
        if return_logits:
            return out, logits
        return out
    
    def _build_param_groups(self):
        """
        Build a dict group_name -> list[Parameter].
        Called once and cached.
        """
        groups = {
            "mlp":       list(self.mlp.parameters()),
            "classifier": list(self.classifier.parameters()),
            "text_lora_all": [],
            "enc_last2": [],
        }

        n_layers = self.enc.config.num_hidden_layers
        last_two = {n_layers - 1, n_layers - 2}

        # LoRA params live inside self.text_enc with names containing 'lora_'
        for name, p in self.enc.named_parameters():
            # 1) collect all LoRA params as before
            if "lora_" in name:
                groups["text_lora_all"].append(p)
                continue

            # 2) collect params that belong to the last 2 layers
            # Works for names like "...layer.4....", "...layer.5...."
            if any(f"layer.{i}." in name for i in last_two):
                groups["enc_last2"].append(p)



        return groups

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