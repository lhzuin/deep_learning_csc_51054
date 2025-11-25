# train_tweets.py
import os, sys, signal, torch, wandb, hydra
import random
import numpy as np
from omegaconf import OmegaConf
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from transformers.optimization import get_scheduler
from tqdm.auto import tqdm


OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For more determinism on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_scheduler_for_stage(optimizer, *, stage_name, cfg, num_training_steps):
    # stage_name is "stage1" or "stage2"
    scheduler_type = getattr(cfg, f"lr_scheduler_{stage_name}")
    warmup_fraction = getattr(cfg, f"warmup_fraction_{stage_name}")
    use_warmup = getattr(cfg, "use_warmup", True)

    num_warmup_steps = int(warmup_fraction * num_training_steps) if use_warmup else 0

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

def param_groups_for(model, lr_class, lr_lora, cfg):
    lora_params = [p for n,p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    return [
        {"params": head_params, "lr": lr_class, "weight_decay": cfg.optim.weight_decay},
        {"params": lora_params, "lr": lr_lora,  "weight_decay": 0.0},
    ]

def run_epoch(stage_name, train_loader, model, loss_fn, optimizer, scheduler, scaler, logger, device):
    model.train()
    total_loss, total_n = 0.0, 0

    # tqdm progress bar over the train dataloader
    progress = tqdm(
        train_loader,
        desc=f"{stage_name} | training",
        leave=False  # set True if you want to keep all bars
    )

    for batch in progress:
        batch["label"] = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with autocast(device_type=device.type):
                logp = model(batch)
                loss = loss_fn(logp, batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logp = model(batch)
            loss = loss_fn(logp, batch["label"])
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        loss_val = loss.detach().cpu().item()
        total_loss += loss_val * len(batch["label"])
        total_n    += len(batch["label"])

        # tqdm: show current loss in the bar
        progress.set_postfix(loss=f"{loss_val:.4f}")

        if logger:
            wandb.log({f"{stage_name}/loss_step": loss_val})

    return total_loss / total_n

def run_validation(val_loader, model, loss_fn, device):
    model.eval()
    val_loss, val_n = 0.0, 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch["label"] = batch["label"].to(device)
            logp = model(batch)
            loss = loss_fn(logp, batch["label"])
            val_loss += loss.detach().cpu().item() * len(batch["label"])
            val_n    += len(batch["label"])

            preds = logp.argmax(dim=1)
            correct += (preds == batch["label"]).sum().item()
            total   += len(batch["label"])
    val_loss /= max(1, val_n)
    val_acc = correct / max(1, total)
    return val_loss, val_acc

@hydra.main(config_path="../configs", config_name="train_v9_camembert2", version_base="1.1")
def train(cfg):
    set_seed(cfg.seed)
    logger = wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name) if cfg.log else None

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"PID={os.getpid()}  (kill -SIGUSR1 {os.getpid()} to checkpoint+exit)")

    # Data
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(cfg.checkpoint_save_path)

    # Model (override n_source_buckets from data)
    model = hydra.utils.instantiate(
        cfg.model.instance,
        n_source_buckets=datamodule.n_source_buckets
    ).to(device)

    # Early stop via signal
    def save_and_exit(*_):
        torch.save(model.state_dict(), cfg.checkpoint_save_path)
        print(f"Saved checkpoint → {cfg.checkpoint_save_path}")
        sys.exit(0)
    signal.signal(signal.SIGUSR1, save_and_exit)

    train_loader = datamodule.train_class_dataloader()
    val_loader   = datamodule.val_dataloader()
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    scaler = GradScaler(device="cuda") if device.type == "cuda" else None

    # ------- Stage 1: MLP-only warmup -------
    model.set_stage("mlp_only")
    opt_cfg = OmegaConf.to_container(cfg.optim, resolve=True, enum_to_str=True)
    lr_class = opt_cfg.pop("lr_class")
    lr_lora  = opt_cfg.pop("lr_lora")
    optimizer = hydra.utils.instantiate(opt_cfg, params=param_groups_for(model, lr_class, lr_lora, cfg), _convert_="all")

    if cfg.use_warmup:
        num_training_steps_stage1 = cfg.epochs_stage1 * len(train_loader)
        scheduler_stage1 = build_scheduler_for_stage(
            optimizer,
            stage_name="stage1",
            cfg=cfg,
            num_training_steps=num_training_steps_stage1,
        )
    else:
        scheduler_stage1 = None

    


    best_val_loss, best_val_acc, patience, epochs_no_improve = float("inf"), 0, cfg.early_stopping.patience, 0
    for epoch in range(cfg.epochs_stage1):
        train_loss = run_epoch("stage1", train_loader, model, loss_fn, optimizer, scheduler_stage1, scaler, logger, device)
        if logger: wandb.log({"epoch": epoch, "stage1/loss_epoch": train_loss})

        val_loss, val_acc = run_validation(val_loader, model, loss_fn, device)
        
        if logger: 
            wandb.log({"epoch": epoch, "stage1/val_loss_epoch": val_loss})
            wandb.log({"epoch": epoch, "stage1/val_acc_epoch": val_acc})

        if val_loss < best_val_loss and val_acc > best_val_acc:
            best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
            torch.save(model.state_dict(), cfg.checkpoint_save_path)
            print(f"[Stage1][Epoch {epoch}] best val loss {best_val_loss:.4f}, best val acc {best_val_acc:.4f} (saved)")
            
        else:
            print(f"[Stage1] No improve in epoch {epoch}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch >= cfg.early_stopping.min_epochs:
                print("Early stopping Stage 1.")
                break

    # ------- Stage 2: full (LoRA + all shallow layers) -------
    model.set_stage("full")
    optimizer = hydra.utils.instantiate(opt_cfg, params=param_groups_for(model, lr_class, lr_lora, cfg), _convert_="all")
    if cfg.use_warmup:
        num_training_steps_stage2 = cfg.epochs_stage2 * len(train_loader)
        scheduler_stage2 = build_scheduler_for_stage(
            optimizer,
            stage_name="stage2",
            cfg=cfg,
            num_training_steps=num_training_steps_stage2,
        )
    else:
        scheduler_stage2 = None

    best_val_loss, best_val_acc, epochs_no_improve = float("inf"), 0, 0
    for epoch in range(cfg.epochs_stage2):
        train_loss = run_epoch("stage2", train_loader, model, loss_fn, optimizer, scheduler_stage2, scaler, logger, device)
        if logger: wandb.log({"epoch": epoch, "stage2/loss_epoch": train_loss})

        val_loss, val_acc = run_validation(val_loader, model, loss_fn, device)

        if logger: 
            wandb.log({"epoch": epoch, "stage2/val_loss_epoch": val_loss})
            wandb.log({"epoch": epoch, "stage2/val_acc_epoch": val_acc})
            

        if val_loss < best_val_loss and val_acc > best_val_acc:
            best_val_loss, best_val_acc, epochs_no_improve = val_loss, val_acc, 0
            torch.save(model.state_dict(), cfg.checkpoint_save_path)
            print(f"[Stage2][Epoch {epoch}] best val loss {best_val_loss:.4f}, best val acc {best_val_acc:.4f} (saved)")
        else:
            print(f"[Stage2]No improve in epoch {epoch}: val loss {val_loss:.4f}, val acc {val_acc:.4f}")
            epochs_no_improve += 1
            if epochs_no_improve >= patience and epoch >= cfg.early_stopping.min_epochs:
                print("Early stopping Stage 2.")
                break

    if logger: logger.finish()

if __name__ == "__main__":
    train()