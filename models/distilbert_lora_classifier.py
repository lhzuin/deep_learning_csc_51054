# Idea: Use the encoder part of a encoder-decoder model as a feature extractor for tweets. Add a 2 layer mlp head
# Fine-tune the model with LoRa
# Implement cross validation
# First train only the text tower + head. Then save the model and train a new head + the other projections
import torch
import torch.nn as nn
import transformers
from peft import get_peft_model, LoraConfig


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class ClassifierDistilBert(nn.Module):
    def __init__(
        self,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.15,
        ):

        super().__init__()

        distilbert_name = "distilbert-base-uncased"
    

        # text tower
        self.text_encoder = transformers.DistilBertModel.from_pretrained(distilbert_name)
        self.tokenizer    = transformers.DistilBertTokenizerFast.from_pretrained(distilbert_name)
        text_lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_lin", "v_lin"],  # key and value proj
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.text_encoder = get_peft_model(self.text_encoder, text_lora_cfg)
        for p in self.text_encoder.base_model.parameters():
            p.requires_grad = False
        bert_dim = self.text_encoder.config.hidden_size

        self.loss = nn.NLLLoss()

        D = bert_dim
        joint_dim = D

        self.head = nn.Sequential(
            nn.LayerNorm(joint_dim),
            nn.Dropout(head_dropout),
            nn.Linear(joint_dim,head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 2),  
            nn.LogSoftmax(dim=1),
        )

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch):
        tok_tweet = self.tokenizer(batch['full_text'], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        tok_tweet = {k:v.to(device) for k,v in tok_tweet.items()}


        out_tweet = self.text_encoder(**tok_tweet)

        t_f = self.title_proj(out_tweet.last_hidden_state[:,0])

        joint_f = [t_f]
        joint = torch.cat(joint_f, dim=-1)
        return self.head(joint).squeeze(1)