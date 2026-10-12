"""
Fine-tune SwinCVS (end-to-end or frozen variant) on the SAFE dataset.

    python3 train_safe.py --config config/SwinCVS_safe_finetune_e2e.yaml
    python3 train_safe.py --config config/SwinCVS_safe_finetune_frozen.yaml

The model is initialised from a FULL Endoscapes-trained SwinCVS checkpoint
(MODEL.INIT_WEIGHTS, strict load - any mismatch raises). Data are the SAFE 1fps
split csvs (labels/<FPS>/splits/{train,val,test}.csv): one sample per
`is_ds_keyframe` row = the 5 consecutive frames ending at the keyframe, targets
C1/C2/C3. Preprocessing is identical to SAFE inference (scripts/f_dataset_safe.py);
the train split additionally gets optional sequence-consistent augmentation.

Loss: BCEWithLogitsLoss(pos_weight = neg/pos per class on the SAFE train split).
For E2E with multiclassifier the loss is alpha * L(fc_swin) + (1 - alpha) * L(fc_lstm),
and the same mix is used for the validation loss. Metrics (mAP, per-criterion AP,
balanced accuracy) are computed on val only, from the LSTM head (the one used at
inference). The best (val mAP) and last checkpoints are plain state dicts written
to a fresh run directory <OUTPUT_DIR>/<run_name>_<timestamp>/ and can be passed to
`inference.py --weights`.

Partial fine-tuning: TRAIN.LAST_LAYERS / --train_last_layers N freezes the SwinV2 backbone
except its last N layers (see backbone_layer_groups()): the blocks and patch-merging layers
counted backwards from the output down to patch_embed, the final norm coming along with the
last block. The heads (LSTM, fc_lstm, fc_swin) always train; 0 = whole backbone frozen.
MODEL.FROZEN_BACKBONE_EVAL then puts the frozen backbone layers in eval mode.
null = the config's own setup (whole model for E2E, LSTM + fc_lstm for frozen).
"""

print("Importing libraries...")
# Standard library imports
import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
import warnings

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Local imports
from scripts.f_environment import get_config, set_deterministic_behaviour
from scripts.f_build import build_finetune_model
from scripts.f_dataset_safe import get_safe_datasets, get_safe_dataloader, compute_pos_weight
from scripts.f_training_utils import build_optimizer, update_params, NativeScalerWithGradNormCount
from scripts.f_metrics import get_map, get_balanced_accuracies

warnings.filterwarnings("ignore")


##############################################################################################
# CONFIG
##############################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SwinCVS on SAFE")
    parser.add_argument("--config", "--config_path", dest="config", type=str, required=True,
                        help="config/SwinCVS_safe_finetune_{e2e,frozen}.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE (e.g. cuda:0, cpu)")
    parser.add_argument("--data_root", type=str, default=None, help="Override DATA.ROOT")
    parser.add_argument("--output_dir", type=str, default=None, help="Override OUTPUT_DIR")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name (default EXPERIMENT_NAME)")
    parser.add_argument("--init_weights", type=str, default=None, help="Override MODEL.INIT_WEIGHTS")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="Train batch size")
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_last_layers", type=int, default=None,
                        help="Freeze the backbone except its last N layers (Swin blocks / patch merging, counted "
                             "from the output; 0 = all frozen); the heads always train. Overrides TRAIN.LAST_LAYERS")
    parser.add_argument("--max_train_iters", type=int, default=None, help="Limit train iterations per epoch")
    parser.add_argument("--max_val_iters", type=int, default=None, help="Limit val (and test) iterations")
    parser.add_argument("--eval_test", action="store_true", help="Evaluate the best checkpoint on the test split")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--opts", nargs="+", default=[],
                        help="Extra yacs overrides as KEY VALUE pairs, e.g. --opts TRAIN.OPTIMIZER.CLASSIFIER_LR 5e-5")
    return parser.parse_args()


def load_config(args):
    config = get_config(args.config, mode="inference")  # plain yacs load (no Endoscapes naming/validation)
    config.defrost()
    overrides = {
        "DEVICE": args.device,
        "DATA.ROOT": args.data_root,
        "OUTPUT_DIR": args.output_dir,
        "WANDB.RUN_NAME": args.run_name,
        "MODEL.INIT_WEIGHTS": args.init_weights,
        "TRAIN.EPOCHS": args.epochs,
        "TRAIN.BATCH_SIZE": args.batch_size,
        "VAL.BATCH_SIZE": args.val_batch_size,
        "TRAIN.NUM_WORKERS": args.num_workers,
        "TRAIN.LAST_LAYERS": args.train_last_layers,
        "SEED": args.seed,
        "TRAIN.MAX_ITERS_PER_EPOCH": args.max_train_iters,
        "VAL.MAX_ITERS": args.max_val_iters,
        "WANDB.PROJECT": args.wandb_project,
        "WANDB.ENTITY": args.wandb_entity,
    }
    for key, value in overrides.items():
        if value is not None:
            node = config
            *parents, leaf = key.split(".")
            for p in parents:
                node = node[p]
            node[leaf] = value
    if args.eval_test:
        config.TEST.ENABLE = True
    if args.no_wandb:
        config.WANDB.ENABLE = False
    for key, value in zip(args.opts[0::2], args.opts[1::2]):
        node = config
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key in --opts: {key}")
        value = yaml.safe_load(value)
        if isinstance(value, str):  # YAML 1.1 reads '1e-3' (no dot) as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[leaf] = value

    if not config.MODEL.LSTM:
        raise ValueError("SAFE fine-tuning supports SwinCVS only (MODEL.LSTM=True)")
    config.MODEL.INFERENCE = False
    if config.TRAIN.get("LAST_LAYERS") is not None and config.TRAIN.LAST_LAYERS < 0:
        raise ValueError(f"TRAIN.LAST_LAYERS must be >= 0 or null, got {config.TRAIN.LAST_LAYERS}")
    if not config.WANDB.RUN_NAME:
        config.WANDB.RUN_NAME = config.EXPERIMENT_NAME
        if config.TRAIN.get("LAST_LAYERS") is not None:
            config.WANDB.RUN_NAME += f"_last{config.TRAIN.LAST_LAYERS}"
    config.freeze()
    return config


def config_to_dict(config):
    return yaml.safe_load(config.dump())


##############################################################################################
# TRAIN / EVAL
##############################################################################################


def uses_multiclassifier(config):
    return bool(config.MODEL.E2E and config.MODEL.MULTICLASSIFIER)


def compute_losses(config, criterion, outputs, targets, alpha):
    """Returns (total, swin or None, lstm). Same mix for train and val."""
    if uses_multiclassifier(config):
        outputs_swin, outputs_lstm = outputs
        loss_swin = criterion(outputs_swin.float(), targets)
        loss_lstm = criterion(outputs_lstm.float(), targets)
        return alpha * loss_swin + (1 - alpha) * loss_lstm, loss_swin, loss_lstm
    loss_lstm = criterion(outputs.float(), targets)
    return loss_lstm, None, loss_lstm


def lstm_logits(config, outputs):
    return outputs[1] if uses_multiclassifier(config) else outputs


def backbone_layer_groups(model):
    """(name, params, modules) per SwinV2 layer, output first: the blocks / patch merging in
    reverse forward order, patch_embed last. The final norm comes with the last block.
    `modules` are the ones whose train/eval mode follows the layer."""
    swin = model.swinv2_model
    groups = []
    for i in reversed(range(len(swin.layers))):
        stage = swin.layers[i]
        blocks = [(f"swinv2_model.layers.{i}.blocks.{j}", b) for j, b in enumerate(stage.blocks)]
        if stage.downsample is not None:
            blocks.append((f"swinv2_model.layers.{i}.downsample", stage.downsample))
        for name, module in reversed(blocks):
            params, modules = list(module.parameters()), [module]
            if not groups:
                params += list(swin.norm.parameters())
                modules.append(swin.norm)
            groups.append((name, params, modules))
    params, modules = list(swin.patch_embed.parameters()), [swin.patch_embed, swin.pos_drop]
    if swin.ape:
        params.append(swin.absolute_pos_embed)
    groups.append(("swinv2_model.patch_embed", params, modules))
    assert sum(len(g[1]) for g in groups) == len(list(swin.parameters())), "backbone_layer_groups misses parameters"
    return groups


def apply_last_layers(config, model):
    """Train the heads (LSTM, fc_lstm, fc_swin) and only the last TRAIN.LAST_LAYERS backbone layers;
    the rest of the backbone is frozen. Returns the frozen backbone modules (to be put in eval mode
    when MODEL.FROZEN_BACKBONE_EVAL), or None if LAST_LAYERS is unset."""
    n = config.TRAIN.get("LAST_LAYERS")
    if n is None:
        return None
    groups = backbone_layer_groups(model)
    if n > len(groups):
        print(f"TRAIN.LAST_LAYERS={n} > {len(groups)} backbone layers: training all of them")
    trained, frozen = groups[:n], groups[n:]
    for p in model.parameters():
        p.requires_grad = True  # heads; frozen backbone layers are switched off below
    for _, params, _ in frozen:
        for p in params:
            p.requires_grad = False
    # fc_swin shares the ENCODER_LR param group with the backbone (build_optimizer)
    if (trained or getattr(model, "fc_swin", None) is not None) and config.TRAIN.OPTIMIZER.ENCODER_LR <= 0:
        raise ValueError(f"TRAIN.LAST_LAYERS={n} trains backbone layers / fc_swin, but TRAIN.OPTIMIZER.ENCODER_LR "
                         f"is {config.TRAIN.OPTIMIZER.ENCODER_LR}: set it > 0 (e.g. --opts TRAIN.OPTIMIZER.ENCODER_LR 5e-6)")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Training the heads + the last {len(trained)} of {len(groups)} backbone layers, "
          f"output first: {[g[0] for g in trained]}")
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,}")
    return [m for _, _, modules in frozen for m in modules]


def build_lr_scheduler(config, optimizer, steps_per_epoch):
    """Per-optimizer-step multiplier on every param group's lr: linear warmup then cosine to MIN_LR_RATIO."""
    sched = config.TRAIN.LR_SCHEDULER
    if sched.NAME == "none":
        return None
    if sched.NAME != "cosine":
        raise NotImplementedError(sched.NAME)
    total = max(1, config.TRAIN.EPOCHS * steps_per_epoch)
    warmup = int(sched.WARMUP_EPOCHS * steps_per_epoch)
    min_ratio = sched.MIN_LR_RATIO

    def lr_lambda(step):
        if step < warmup:
            return min_ratio + (1 - min_ratio) * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(config, model, loader, criterion, optimizer, loss_scaler, lr_scheduler,
                    device, epoch, alpha, global_step, wandb_run, frozen_backbone=None):
    model.train()
    if config.MODEL.FROZEN_BACKBONE_EVAL:
        if frozen_backbone is not None:  # TRAIN.LAST_LAYERS: only the frozen backbone layers
            for m in frozen_backbone:
                m.eval()
        elif not config.MODEL.E2E:
            model.swinv2_model.eval()

    use_amp = config.TRAIN.AMP and device.type == "cuda"
    accum = config.TRAIN.ACCUMULATION_STEPS
    max_iters = config.TRAIN.MAX_ITERS_PER_EPOCH
    n_iters = len(loader) if not max_iters else min(len(loader), max_iters)
    trainable = [p for p in model.parameters() if p.requires_grad]

    sums = {"loss": 0.0, "loss_swin": 0.0, "loss_lstm": 0.0}
    n_samples = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, total=n_iters, desc=f"Train {epoch + 1:02}/{config.TRAIN.EPOCHS:02}", dynamic_ncols=True)
    for idx, (samples, targets) in enumerate(pbar):
        if idx >= n_iters:
            break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(samples)
        loss, loss_swin, loss_lstm = compute_losses(config, criterion, outputs, targets, alpha)

        # Gradient accumulation: step (and zero grads) only every `accum` iterations / at the end
        update = ((idx + 1) % accum == 0) or (idx + 1 == n_iters)
        grad_norm = loss_scaler(loss / accum, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=trainable, update_grad=update)
        if update:
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        bs = targets.shape[0]
        n_samples += bs
        sums["loss"] += loss.item() * bs
        sums["loss_lstm"] += loss_lstm.item() * bs
        if loss_swin is not None:
            sums["loss_swin"] += loss_swin.item() * bs
        global_step += 1

        log = {"train/loss": loss.item(), "train/loss_lstm": loss_lstm.item(),
               "train/lr_encoder": optimizer.param_groups[0]["lr"],
               "train/lr_classifier": optimizer.param_groups[-1]["lr"], "epoch": epoch + 1}
        if loss_swin is not None:
            log["train/loss_swin"] = loss_swin.item()
        if grad_norm is not None:
            log["train/grad_norm"] = float(grad_norm)
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_log = {"train/epoch_loss": sums["loss"] / max(1, n_samples),
                 "train/epoch_loss_lstm": sums["loss_lstm"] / max(1, n_samples)}
    if uses_multiclassifier(config):
        epoch_log["train/epoch_loss_swin"] = sums["loss_swin"] / max(1, n_samples)
    return epoch_log, global_step


@torch.inference_mode()
def evaluate(config, model, loader, criterion, device, alpha, prefix="val"):
    """fp32 forward (as in inference.py); loss is the training criterion averaged per sample."""
    model.eval()
    max_iters = config.VAL.MAX_ITERS
    n_iters = len(loader) if not max_iters else min(len(loader), max_iters)

    sums = {"loss": 0.0, "loss_swin": 0.0, "loss_lstm": 0.0}
    n_samples = 0
    probs, preds, trues = [], [], []
    for idx, (samples, targets) in enumerate(tqdm(loader, total=n_iters, desc=prefix.capitalize(), dynamic_ncols=True)):
        if idx >= n_iters:
            break
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(samples)
        loss, loss_swin, loss_lstm = compute_losses(config, criterion, outputs, targets, alpha)

        bs = targets.shape[0]
        n_samples += bs
        sums["loss"] += loss.item() * bs
        sums["loss_lstm"] += loss_lstm.item() * bs
        if loss_swin is not None:
            sums["loss_swin"] += loss_swin.item() * bs

        probability = torch.sigmoid(lstm_logits(config, outputs).float())
        probs.append(probability.cpu())
        preds.append(torch.round(probability).cpu())
        trues.append(targets.cpu())

    C1_bacc, C2_bacc, C3_bacc, mean_recall = get_balanced_accuracies(trues, preds)
    C1_ap, C2_ap, C3_ap, mAP = get_map(trues, probs)
    baccs = [C1_bacc, C2_bacc, C3_bacc]
    metrics = {
        f"{prefix}/loss": sums["loss"] / max(1, n_samples),
        f"{prefix}/loss_lstm": sums["loss_lstm"] / max(1, n_samples),
        f"{prefix}/mAP": mAP,
        f"{prefix}/AP_C1": C1_ap,
        f"{prefix}/AP_C2": C2_ap,
        f"{prefix}/AP_C3": C3_ap,
        f"{prefix}/bacc_C1": C1_bacc,
        f"{prefix}/bacc_C2": C2_bacc,
        f"{prefix}/bacc_C3": C3_bacc,
        # mean of the three per-criterion balanced accuracies
        f"{prefix}/bacc_mean": float(np.nanmean(baccs)) if not np.all(np.isnan(baccs)) else float("nan"),
        # f_metrics' 4th return value (reported as "avg_bal_acc" by SwinCVS.py) is the mean RECALL
        f"{prefix}/recall_mean": mean_recall,
    }
    if uses_multiclassifier(config):
        metrics[f"{prefix}/loss_swin"] = sums["loss_swin"] / max(1, n_samples)
    outputs = {"probs": torch.cat(probs).numpy(), "targets": torch.cat(trues).numpy()}
    return metrics, outputs


def fmt(metrics):
    return " | ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())


##############################################################################################
# MAIN
##############################################################################################


def init_wandb(config, run_dir):
    if not config.WANDB.ENABLE:
        print("wandb disabled")
        return None
    import wandb  # raise if missing and logging was requested

    run = wandb.init(
        project=config.WANDB.PROJECT,
        entity=config.WANDB.ENTITY,
        name=config.WANDB.RUN_NAME,
        tags=list(config.WANDB.TAGS) if config.WANDB.TAGS else None,
        config=config_to_dict(config),
        dir=str(run_dir),
    )  # mode comes from the WANDB_MODE env var (online/offline/disabled)
    run.define_metric("val/mAP", summary="max")
    return run


def main():
    args = parse_args()
    config = load_config(args)
    set_deterministic_behaviour(config.SEED)

    device = torch.device(config.DEVICE if (not config.DEVICE.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    run_name = config.WANDB.RUN_NAME
    run_dir = Path(config.OUTPUT_DIR) / f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never reuse / overwrite an existing run dir
    with open(run_dir / "config.yaml", "w") as f:
        f.write(config.dump())
    print(f"Run directory: {run_dir}")

    # DATA
    split_names = ("train", "val", "test") if config.TEST.ENABLE else ("train", "val")
    datasets = get_safe_datasets(config, split_names)
    train_dataset = datasets["train"][0]
    train_loader = get_safe_dataloader(config, train_dataset, shuffle=True)
    val_loader = get_safe_dataloader(config, datasets["val"][0], shuffle=False, batch_size=config.VAL.BATCH_SIZE)

    # MODEL
    model = build_finetune_model(config)
    frozen_backbone = apply_last_layers(config, model)
    model.to(device)

    # LOSS: pos_weight = neg/pos per class on the SAFE train split
    if config.TRAIN.POS_WEIGHT is None:
        pos_weight = compute_pos_weight(train_dataset)
    else:
        pos_weight = torch.tensor(config.TRAIN.POS_WEIGHT, dtype=torch.float32)
    print(f"BCE pos_weight (C1, C2, C3): {[round(w, 3) for w in pos_weight.tolist()]}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # OPTIMIZER / SCHEDULER / AMP
    optimizer = build_optimizer(config, model)
    loss_scaler = NativeScalerWithGradNormCount(enabled=config.TRAIN.AMP and device.type == "cuda")
    max_iters = config.TRAIN.MAX_ITERS_PER_EPOCH
    iters_per_epoch = len(train_loader) if not max_iters else min(len(train_loader), max_iters)
    steps_per_epoch = math.ceil(iters_per_epoch / config.TRAIN.ACCUMULATION_STEPS)
    lr_scheduler = build_lr_scheduler(config, optimizer, steps_per_epoch)

    wandb_run = init_wandb(config, run_dir)
    if wandb_run is not None:
        wandb_run.config.update({"pos_weight": pos_weight.tolist(),
                                 "n_train": len(train_dataset), "n_val": len(datasets["val"][0])})

    alpha = config.TRAIN.MULTICLASSIFIER_ALPHA
    best_map = -math.inf
    best_epoch = None
    global_step = 0
    results = {}
    time_list = []

    print(f"Beginning fine-tuning: {run_name}")
    for epoch in range(config.TRAIN.EPOCHS):
        start_time = time.time()
        if uses_multiclassifier(config) and config.TRAIN.MULTICLASSIFIER_ALPHA_DECAY:
            alpha, _ = update_params(alpha, 1 - alpha, epoch)

        train_log, global_step = train_one_epoch(
            config, model, train_loader, criterion, optimizer, loss_scaler, lr_scheduler,
            device, epoch, alpha, global_step, wandb_run, frozen_backbone,
        )
        val_metrics, val_outputs = evaluate(config, model, val_loader, criterion, device, alpha, prefix="val")
        print(f"Epoch {epoch + 1}: {fmt(train_log)}")
        print(f"Epoch {epoch + 1}: {fmt(val_metrics)}")

        val_map = val_metrics["val/mAP"]
        is_best = not math.isnan(val_map) and val_map > best_map
        if is_best:
            best_map, best_epoch = val_map, epoch + 1
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"New best val mAP {val_map:.4f} (epoch {epoch + 1}) -> {run_dir / 'best.pt'}")
        torch.save(model.state_dict(), run_dir / "last.pt")

        if wandb_run is not None:
            log = {**train_log, **val_metrics, "epoch": epoch + 1}
            if uses_multiclassifier(config):
                log["train/alpha"] = alpha
            wandb_run.log(log, step=global_step)
            if best_epoch is not None:
                wandb_run.summary["best/val_mAP"] = best_map
                wandb_run.summary["best/epoch"] = best_epoch

        results[f"Epoch {epoch + 1}"] = {**train_log, **val_metrics, "alpha": alpha,
                                         "preds_prob": val_outputs["probs"].tolist()}
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        time_list.append(time.time() - start_time)
        print(f"Epoch duration: {int(time_list[-1])}s | "
              f"estimated remaining: {int(np.mean(time_list) * (config.TRAIN.EPOCHS - epoch - 1))}s")

    print(f"Best val mAP: {best_map:.4f} at epoch {best_epoch}" if best_epoch else "No valid val mAP (best.pt not written)")

    # OPTIONAL TEST EVALUATION ON THE BEST CHECKPOINT
    if config.TEST.ENABLE:
        ckpt = run_dir / ("best.pt" if best_epoch is not None else "last.pt")
        print(f"Evaluating {ckpt.name} on the test split")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
        test_dataset, test_meta = datasets["test"]
        test_loader = get_safe_dataloader(config, test_dataset, shuffle=False, batch_size=config.VAL.BATCH_SIZE)
        test_metrics, test_outputs = evaluate(config, model, test_loader, criterion, device, alpha, prefix="test")
        print(f"Test ({ckpt.name}): {fmt(test_metrics)}")
        with open(run_dir / "test_metrics.json", "w") as f:
            json.dump({"checkpoint": ckpt.name, "epoch": best_epoch, **test_metrics}, f, indent=2)
        n = len(test_outputs["probs"])
        result_df = test_meta.iloc[:n].copy()
        for i, c in enumerate(["C1", "C2", "C3"]):
            result_df[f"Conf_{c}"] = test_outputs["probs"][:, i]
        result_df.to_csv(run_dir / "test_result.csv", index=False)
        if wandb_run is not None:
            for k, v in test_metrics.items():
                wandb_run.summary[k] = v

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
