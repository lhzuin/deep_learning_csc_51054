# Influencer Classification – README

This project trains **tweet**, **description**, and **fusion** (late-fusion) models to classify users as influencer vs observer.  
Training & inference are driven by **Hydra configs**.

---

## ⚙️ Repository Structure

Place these files in the **project root**:

train.jsonl
kaggle_test.jsonl

At the same level as:
- configs/   
- models/    
- data/    
- train/
- checkpoints/
- outputs/

---

## 🔹 Config Types

| Type | Naming | Purpose |
|---|---|---|
| Tweet | `tweet_only_*` | Train tweet text tower only |
| Description | `desc_only_*` | Train description tower only |
| Fusion / Submission | others (`fusion_*`, etc.) | Use multiple towers and generate predictions |

Older configs may require legacy scripts (`train_v0.py` … `train_v3.py`).

---

## 🚀 Training

Run from project root:

```bash
python -m train.train CONFIG_NAME
```

Examples:

```bash
python -m train.train tweet_only_v14
python -m train.train desc_only_v3
python -m train.train fusion_v22
```

Training is staged (freezing/unfreezing groups, LoRA phases, schedulers, early stopping) as defined in each config.

⸻

📦 Checkpoints

During training, best weights are saved to:
```bash
outputs/<date>/<name>.pt
```
and should be manually transferred to:
```bash
checkpoints/<name>.pt
```

Fusion models require pretrained tower checkpoints:
```bash
checkpoints/meta_tower_best.pt
checkpoints/tweet_only_lora.pt
checkpoints/desc_only_lora.pt
```

Paths must be set in the fusion config (meta_ckpt_path, tweet_ckpt_path, desc_ckpt_path).

⸻

🏁 Generating Submission CSVs

Single-tower (tweet/desc):
```bash
python -m train.predict_tweets CONFIG_NAME
```
Fusion models:
```bash
python -m train.predict_tweets_fusion CONFIG_NAME
```
Output will be written to the config-defined:
```bash
outputs/<submission>.csv
```

⸻

Notes
- Fusion prediction reloads all tower checkpoints and reuses metadata scalers/stats.
- Metadata tower is trained separately using the final cell of models/designing_metadata_tower.ipynb.
- Validation splits can be random or author-based depending on config.

⸻


