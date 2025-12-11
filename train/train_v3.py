# train_tweets.py
import os, sys, signal, torch, wandb, hydra
import random
import numpy as np
from omegaconf import OmegaConf
from torch.amp import autocast, GradScaler
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For more determinism on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_scheduler_for_stage(optimizer, *, stage_dict, cfg, num_training_steps):
    # stage_name is "stage1" or "stage2"
    scheduler_type = stage_dict["lr_scheduler"]
    warmup_fraction = stage_dict["warmup_fraction"]
    use_warmup = getattr(cfg, "use_warmup", True)

    num_warmup_steps = int(warmup_fraction * num_training_steps) if use_warmup else 0

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

def param_groups_for(model, cfg, stage_dict):
    groups = model.get_param_groups()

    # Expect a dict: {group_name: lr_key}
    stage_groups = stage_dict.get("groups", {})

    optim_cfg = cfg.optim

    # Safely get LR with fallback to lr / lr_class
    def get_lr(key, default_key="lr"):
        if hasattr(optim_cfg, key):
            return getattr(optim_cfg, key)
        if hasattr(optim_cfg, default_key):
            return getattr(optim_cfg, default_key)
        return getattr(optim_cfg, "lr_class")

    pg = []
    for group_name, lr_key in stage_groups.items():
        if group_name not in groups:
            continue
        params = [p for p in groups[group_name] if p.requires_grad]
        if not params:
            continue
        lr = get_lr(lr_key)
        # Special-case: no weight decay for LoRA-style groups
        weight_decay = 0.0 if "lora" in group_name else cfg.optim.weight_decay
        pg.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    return pg



class Trainer:
    def __init__(self, cfg, logger, device):
        self.model = None
        self.cfg = cfg
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None
        self.optimizer = None
        self.scaler = None
        self.logger = logger
        self.device = device
    
    def init_datamodule(self):
        # Data
        self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)
        self.datamodule.setup(self.cfg.checkpoint_save_path, type=self.cfg.get("val_type", "random"))
        self.train_loader = self.datamodule.train_class_dataloader()
        self.val_loader   = self.datamodule.val_dataloader()

    def init_model(self):
        try:
            self.model = hydra.utils.instantiate(
                self.cfg.model.instance,
                n_source_buckets=self.datamodule.n_source_buckets
            ).to(self.device)
        except Exception:
            self.model = hydra.utils.instantiate(
                self.cfg.model.instance
            ).to(self.device)

        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn)

        self.scaler = GradScaler(device="cuda") if self.device.type == "cuda" else None

        self.patience = self.cfg.early_stopping.patience
        self.min_epochs = self.cfg.early_stopping.min_epochs
        self.checkpoint_save_path = self.cfg.checkpoint_save_path
        self.acc_epsilon = self.cfg.acc_epsilon

        # Full optim config as dict
        opt_cfg_full = OmegaConf.to_container(self.cfg.optim, resolve=True, enum_to_str=True)

        # Extract target and keep only AdamW-supported args
        target = opt_cfg_full.pop("_target_")

        allowed_keys = {
            "lr", "betas", "eps", "weight_decay",
            "amsgrad", "foreach", "maximize",
            "capturable", "differentiable", "fused",
        }
        opt_kwargs = {k: v for k, v in opt_cfg_full.items() if k in allowed_keys}
        opt_kwargs["_target_"] = target

        self.opt_cfg = opt_kwargs

        self.optimizer = None

    def run_epoch(self, stage_name, scheduler):
        self.model.train()
        total_loss, total_n = 0.0, 0

        # tqdm progress bar over the train dataloader
        progress = tqdm(
            self.train_loader,
            desc=f"{stage_name} | training",
            leave=False  # set True if you want to keep all bars
        )

        for batch in progress:
            batch["label"] = batch["label"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler:
                with autocast(device_type=self.device.type):
                    logp = self.model(batch)
                    loss = self.loss_fn(logp, batch["label"])
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logp = self.model(batch)
                loss = self.loss_fn(logp, batch["label"])
                loss.backward()
                self.optimizer.step()

            if scheduler:
                scheduler.step()

            loss_val = loss.detach().cpu().item()
            total_loss += loss_val * len(batch["label"])
            total_n    += len(batch["label"])

            # tqdm: show current loss in the bar
            progress.set_postfix(loss=f"{loss_val:.4f}")

            if self.logger:
                wandb.log({
                    f"{stage_name}/loss_step": loss_val,
                    "aggregate/loss_step": loss_val,
                })

        return total_loss / total_n

    def run_validation(self):
        self.model.eval()
        val_loss, val_n = 0.0, 0
        correct, total = 0, 0
        all_labels = []
        all_probs  = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch["label"] = batch["label"].to(self.device)
                logp = self.model(batch)
                loss = self.loss_fn(logp, batch["label"])
                val_loss += loss.detach().cpu().item() * len(batch["label"])
                val_n    += len(batch["label"])

                preds = logp.argmax(dim=1)
                correct += (preds == batch["label"]).sum().item()
                total   += len(batch["label"])

                probs = torch.softmax(logp, dim=1)[:, 1]

                all_labels.append(batch["label"].detach().cpu())
                all_probs.append(probs.detach().cpu())
        val_loss /= max(1, val_n)
        val_acc = correct / max(1, total)

        all_labels = torch.cat(all_labels).numpy()
        all_probs  = torch.cat(all_probs).numpy()
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            # e.g. if only one class present in val
            val_auc = float("nan")
        return val_loss, val_acc, val_auc

    def run_stage(self, stage_dict):
        stage_name = stage_dict["name"]
        print(f"Running stage: {stage_dict['name']}")
        print(">>> Optim cfg :", self.cfg.optim)
        self.model.set_trainable_groups(list(stage_dict["groups"].keys()))

        self.optimizer = hydra.utils.instantiate(
            self.opt_cfg,
            params=param_groups_for(self.model, self.cfg, stage_dict),
            _convert_="all",
        )

        if self.cfg.use_warmup:
            num_training_steps_stage = stage_dict["epochs"] * len(self.train_loader)
            scheduler_stage = build_scheduler_for_stage(
                self.optimizer,
                stage_dict=stage_dict,
                cfg=self.cfg,
                num_training_steps=num_training_steps_stage,
            )
        else:
            scheduler_stage = None

        best_val_loss, best_val_acc, epochs_no_improve = float("inf"), 0, 0
        for epoch in range(stage_dict["epochs"]):
            train_loss = self.run_epoch(stage_name, scheduler_stage)
            if self.logger:
                wandb.log({
                    "epoch": epoch,
                    f"{stage_name}/loss_epoch": train_loss,
                    "aggregate/loss_epoch": train_loss,
                })

            val_loss, val_acc, val_auc = self.run_validation()

            if self.logger:
                wandb.log({
                    "epoch": epoch,
                    f"{stage_name}/val_loss_epoch": val_loss,
                    f"{stage_name}/val_acc_epoch": val_acc,
                    f"{stage_name}/val_auc_epoch": val_auc,
                    "aggregate/val_loss_epoch": val_loss,
                    "aggregate/val_acc_epoch": val_acc,
                    "aggregate/val_auc_epoch": val_auc,
                })
            print("Current AUC:", val_auc)
            improved = (val_acc > best_val_acc) or ((best_val_acc - val_acc < self.acc_epsilon) and (val_loss < best_val_loss))
            if improved:
                best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
                torch.save(self.model.state_dict(), self.checkpoint_save_path)
                print(f"[{stage_name}][Epoch {epoch}] best val loss {best_val_loss:.4f}, best val acc {best_val_acc:.4f} (saved)")
            else:
                print(f"[{stage_name}]No improve in epoch {epoch}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience and epoch >= self.min_epochs:
                    print(f"Early stopping {stage_dict['name']}.")
                    break

    def run_train(self):
        for i, stage_dict in enumerate(self.cfg.train_stages):
            # For stages beyond the first, reload best weights from previous stage
            if i > 0:
                print("Reloading best model from previous stage.")
                state_dict = torch.load(self.checkpoint_save_path, map_location=self.device)
                self.model.load_state_dict(state_dict)

            self.run_stage(stage_dict)


    def _build_cached_train_loader(self):
        """
        Run towers once on the train set, cache fused features, and replace
        self.train_loader by a lightweight loader that only feeds 'fused' + 'label'.
        """
        self.model.eval()
        device = self.device

        fused_list = []
        labels_list = []

        with torch.no_grad():
            progress = tqdm(self.train_loader, desc="Caching tower outputs", leave=False)
            for batch in progress:
                labels = batch["label"].to(device)
                out = self.model(batch, return_parts=True)

                # concat tower features: h_meta, h_tweet, h_desc
                h_meta  = out["h_meta"]
                h_tweet = out["h_tweet"]
                h_desc  = out["h_desc"]
                fused = torch.cat([h_meta, h_tweet, h_desc], dim=1)  # (B, fusion_in_dim)

                fused_list.append(fused.cpu())
                labels_list.append(labels.cpu())

        fused_all = torch.cat(fused_list, dim=0)   # (N, fusion_in_dim)
        labels_all = torch.cat(labels_list, dim=0) # (N,)

        class FusedDataset(Dataset):
            def __init__(self, fused, labels):
                self.fused = fused
                self.labels = labels
            def __len__(self):
                return self.fused.size(0)
            def __getitem__(self, idx):
                return {
                    "fused": self.fused[idx],
                    "label": self.labels[idx],
                }

        fused_dataset = FusedDataset(fused_all, labels_all)

        # try to reuse datamodule batch_size / num_workers
        batch_size = getattr(self.datamodule, "batch_size", 32)
        num_workers = getattr(self.datamodule, "num_workers", 0)

        self.train_loader = DataLoader(
            fused_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # go back to train mode for later
        self.model.train()

@hydra.main(config_path="../configs", config_name="tweet_only_v14", version_base="1.1")
def train(cfg):
    set_seed(cfg.seed)
    logger = wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name) if cfg.log else None

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")

    

    trainer = Trainer(
        cfg=cfg,
        logger=logger,
        device=device,
    )
    trainer.init_datamodule()
    trainer.init_model()
    

    # Early stop via signal
    def save_and_exit(*_):
        torch.save(trainer.model.state_dict(), cfg.checkpoint_save_path)
        print(f"Saved checkpoint → {cfg.checkpoint_save_path}")
        sys.exit(0)
    signal.signal(signal.SIGUSR1, save_and_exit)


    trainer.run_train()

    if logger: logger.finish()

if __name__ == "__main__":
    train()