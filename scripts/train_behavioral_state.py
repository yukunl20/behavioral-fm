import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import Dataset, Subset, DataLoader

from src.data.movielens import MovieLensSequenceDataset, collate_sequences
from src.models.behavioral_state_model import BehavioralStateModel


def file_hash(path):
    # Content fingerprint for split/checkpoint provenance in metrics.json.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Keep only the last max_length events so long users fit a fixed window.
class RecentSequences(Dataset):
    def __init__(self, base, max_length):
        self.base, self.max_length = base, max_length

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        original = self.base[index]
        # How many events to skip from the start so we keep only the last max_length
        # (0 if the sequence is already short enough). Window = original[offset:].
        offset = max(0, len(original["movie_id"]) - self.max_length)
        item = {k: original[k][offset:].clone() for k in
                ("movie_id", "rating", "time_gap_hours", "genres")}
        item["user_id"] = original["user_id"]
        item["row_id"] = index  # dataset row (for scoring export)
        item["offset"] = offset  # absolute start index of this window in the full sequence
        return item


def collate_recent_sequences(items):
    # Pad via shared collator, then attach row/offset metadata for eval dumps.
    batch = collate_sequences(items)
    if "is_real_event" not in batch:
        raise ValueError("collate_sequences must provide is_real_event [B, T] bool mask")
    batch["row_id"] = torch.tensor([x["row_id"] for x in items])
    batch["offset"] = torch.tensor([x["offset"] for x in items])
    return batch


def get_split(path, fingerprint, n, create=False, split_seed=2026):
    # Load a frozen train/val index split; optionally create a one-shot fallback.
    if not path.exists():
        if not create:
            raise FileNotFoundError("provide an adapted prior split or explicitly create a fallback")
        if n < 2:
            raise ValueError("need both training and validation data")
        order = np.random.default_rng(split_seed).permutation(n)
        n_val = max(1, int(0.15 * n))
        manifest = dict(
            data_hash=fingerprint, policy="random-row exploratory fallback",
            split_seed=split_seed,
            train=order[n_val:].tolist(), validation=order[:n_val].tolist(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:  # fail if someone else created it concurrently
            json.dump(manifest, f, indent=2)
    manifest = json.loads(path.read_text())
    if manifest["data_hash"] != fingerprint:
        raise ValueError("split/data fingerprint mismatch")
    train, val = manifest["train"], manifest["validation"]
    if not train or not val:
        raise ValueError("empty split")
    joined = train + val
    if any(type(i) is not int or not 0 <= i < n for i in joined):
        raise ValueError("invalid dataset index")
    if len(set(joined)) != len(joined):
        raise ValueError("duplicate or overlapping split rows")
    return manifest


def load_encoder(model, path):
    # Copy pretrained EventEncoder weights into model; tolerate a few ckpt layouts.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload)  # {"model": sd, ...} or a bare state_dict
    candidates = []
    # Strip module prefix so keys match EventEncoder.state_dict() names.
    for prefix in ("event_encoder.", "encoder."):  # BehavioralStateModel vs MaskedEventModel
        selected = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if selected:
            candidates.append(selected)
    # Already saved as a raw EventEncoder state_dict (no prefix).
    if not candidates and set(state) == set(model.event_encoder.state_dict()):
        candidates = [state]
    if len(candidates) != 1:  # 0 = unknown layout; >1 = don't guess which prefix
        raise ValueError("ambiguous/unsupported checkpoint schema; write an explicit adapter")
    model.event_encoder.load_state_dict(candidates[0], strict=True)  # every key must match


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    # Make wall-clock timing honest on async accelerators.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def validate_batch(batch):
    # Runner assumptions: bool mask, left-contiguous reals + right pad, finite fields.
    real = batch["is_real_event"]
    if real.dtype != torch.bool:
        raise ValueError("is_real_event must be Boolean")
    expected = torch.arange(real.size(1), device=real.device)[None] < real.sum(1)[:, None]
    if not torch.equal(real, expected):
        raise ValueError("runner requires contiguous real events then right padding")
    for key in ("rating", "time_gap_hours", "genres"):
        # Reject nonfinite padding too: masking is not NaN sanitization.
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"nonfinite {key}")
    if (batch["time_gap_hours"][real] < 0).any():
        raise ValueError("negative elapsed time")


def active_pairs(batch):
    # Supervise only real→real adjacent steps (skip pad and trailing lone events).
    real = batch["is_real_event"]
    return real[:, :-1] & real[:, 1:]


def loss_and_count(output, batch):
    # Next-genre BCE: logits at t predict genres at t+1 on active pairs.
    active = active_pairs(batch)
    count = int(active.sum())
    if count == 0:
        return None, 0
    loss = F.binary_cross_entropy_with_logits(
        output["logits"][:, :-1][active], batch["genres"][:, 1:][active],
    )  # mean over (pairs × genres)
    return loss, count


def quality(targets, scores):
    # Micro AP / F1 over flattened (event, genre) labels.
    return dict(
        micro_ap=float(average_precision_score(targets, scores, average="micro")),
        micro_f1=float(f1_score(targets, scores >= 0.5, average="micro", zero_division=0)),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    # Validation metrics + optional dual-gate diagnostics aligned to supervised pairs.
    model.eval()
    scores, targets, ids, users = [], [], [], []
    gates, gaps, distances = [], [], []
    for batch in loader:
        batch = move(batch, device)
        validate_batch(batch)
        active = active_pairs(batch)
        if not active.any():
            continue
        output = model(batch)
        scores.append(output["logits"][:, :-1].sigmoid()[active].cpu())
        targets.append(batch["genres"][:, 1:][active].cpu())
        row, position = active.nonzero(as_tuple=True)  # position = source index t
        ids.append(torch.stack([
            batch["row_id"][row], batch["offset"][row] + position + 1,  # target event in full seq
        ], dim=1).cpu())
        users.append(batch["user_id"][row].cpu())
        if output["gate"] is not None:
            gates.append(output["gate"][:, :-1][active].cpu())
            # This is the observed gap INTO t, not the future gap into t+1.
            gaps.append(batch["time_gap_hours"][:, :-1][active].cpu())
            distances.append((output["fast_z"] - output["slow_z"]).square()
                             .mean(-1).sqrt()[:, :-1][active].cpu())
    if not scores:
        raise ValueError("no eligible validation transitions")
    arrays = dict(scores=torch.cat(scores).numpy(), targets=torch.cat(targets).numpy(),
                  row_ids=torch.cat(ids).numpy(), user_ids=torch.cat(users).numpy())
    if not np.isfinite(arrays["scores"]).all():
        raise ValueError("nonfinite validation scores")
    result = quality(arrays["targets"], arrays["scores"])
    if gates:
        gate = torch.cat(gates)
        if not torch.isfinite(gate).all():
            raise ValueError("nonfinite gate")
        result.update(
            gate_mean=float(gate.mean()), gate_std=float(gate.std(unbiased=False)),
            gate_p05=float(torch.quantile(gate, 0.05)),
            gate_p95=float(torch.quantile(gate, 0.95)),
        )
        arrays.update(gate_mean_per_event=gate.mean(-1).numpy(),
                      gap_hours=torch.cat(gaps).numpy(),
                      branch_rms_distance=torch.cat(distances).numpy())
    return result, arrays


@torch.no_grad()
def prevalence_baseline(loader, num_genres):
    # Constant predictor = train-set genre frequencies (sanity floor for micro-AP).
    total = torch.zeros(num_genres)
    count = 0
    for batch in loader:
        validate_batch(batch)
        y = batch["genres"][:, 1:][active_pairs(batch)]
        total += y.sum(0)
        count += len(y)
    if count == 0:
        raise ValueError("no training transitions")
    return (total / count).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["event_only", "single", "dual"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--split-file", required=True)
    p.add_argument("--create-split", action="store_true")
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--tau-single-hours", type=float, default=24)
    p.add_argument("--tau-fast-hours", type=float, default=2)
    p.add_argument("--tau-slow-hours", type=float, default=720)
    p.add_argument("--learnable-tau", action="store_true")
    p.add_argument("--fixed-half-gate", action="store_true")
    p.add_argument("--status-mode", choices=["ignore", "status_embed", "mask_token"], default="ignore")
    p.add_argument("--event-checkpoint")
    p.add_argument("--freeze-event-encoder", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        p.error("positive epochs/batch size and max-length >= 2 required")
    # Main protocol: shared frozen encoder so backbone ablations are comparable.
    if not args.smoke and not (args.event_checkpoint and args.freeze_event_encoder):
        p.error("main runs require a shared checkpoint and frozen EventEncoder")
    if args.fixed_half_gate and args.backbone != "dual":
        p.error("fixed-half-gate requires dual")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        p.error("run-id must be one directory name")

    # Reproducibility for python / numpy / torch CPU RNG (dataloader shuffle uses generator below).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    processed = Path(args.processed_dir)
    # Fingerprint data + window length so splits stay tied to the exact view of each user.
    fingerprint = hashlib.sha256((file_hash(processed / "sequences.npy") +
        file_hash(processed / "metadata.json") + f":T={args.max_length}").encode()).hexdigest()
    base = MovieLensSequenceDataset(processed)
    dataset = RecentSequences(base, args.max_length)
    split_path = Path(args.split_file)
    manifest = get_split(split_path, fingerprint, len(dataset), args.create_split, args.split_seed)
    train_set = Subset(dataset, manifest["train"])
    val_set = Subset(dataset, manifest["validation"])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_recent_sequences, num_workers=0,
    )
    train_eval = DataLoader(train_set, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_recent_sequences)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_recent_sequences)
    meta = base.metadata
    model = BehavioralStateModel(
        num_movies=int(meta["num_movies"]), num_genres=int(meta["num_genres"]),
        backbone=args.backbone, width=args.width, status_mode=args.status_mode,
        tau_single_hours=args.tau_single_hours, tau_fast_hours=args.tau_fast_hours,
        tau_slow_hours=args.tau_slow_hours, learnable_tau=args.learnable_tau,
        head_seed=args.seed,  # same genre-head init across backbones (see fork_rng in model)
    )
    if args.event_checkpoint:
        load_encoder(model, args.event_checkpoint)
    if args.freeze_event_encoder:
        model.freeze_event_encoder()
    if args.fixed_half_gate:
        # Last layer is already zero; freeze whole gate so sigmoid stays 0.5.
        model.temporal.gate.requires_grad_(False)
    device = choose_device()
    model.to(device)
    # Decay log_tau without weight decay; everything else with 0.01.
    regular, taus = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (taus if name.endswith("log_tau") else regular).append(parameter)
    param_groups = [{"params": regular, "weight_decay": 0.01}]
    if taus:
        param_groups.append({"params": taus, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
    out = Path("artifacts/day10") / args.run_id / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite smoke/main/ablation results.
    baseline = prevalence_baseline(train_eval, int(meta["num_genres"]))
    history = []
    events = updates = transitions = 0
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()  # Frozen encoder remains in eval mode via override.
        total_loss, total_pairs = 0.0, 0
        for batch in train_loader:
            batch = move(batch, device)
            validate_batch(batch)
            output = model(batch)
            loss, count = loss_and_count(output, batch)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regular + taus, 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * count
            total_pairs += count
            transitions += count
            events += int(batch["is_real_event"].sum())
            updates += 1
        if not total_pairs:
            raise ValueError("epoch had no supervised transitions")
        # Log effective τ (exp of clamped log_tau) for each DecayMemory bank.
        tau_values = {name: float(module.tau_hours.detach().cpu())
                      for name, module in model.named_modules()
                      if hasattr(module, "tau_hours")}
        record = dict(epoch=epoch + 1, loss=total_loss / total_pairs, tau_hours=tau_values)
        history.append(record)
        print(record)
    synchronize(device)
    elapsed = time.perf_counter() - start
    metrics, arrays = evaluate(model, val_loader, device)
    metrics["constant_baseline"] = quality(
        arrays["targets"], np.broadcast_to(baseline, arrays["targets"].shape),
    )
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(
        config=vars(args), data_hash=fingerprint, split_hash=file_hash(split_path),
        split_policy=manifest["policy"],
        encoder_hash=file_hash(args.event_checkpoint) if args.event_checkpoint else None,
        torch_version=str(torch.__version__), device=str(device), git_revision=revision,
        git_dirty=dirty,
    )
    metrics.update(metadata)
    metrics.update(
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        seconds=elapsed, events_per_second=events / elapsed,
        transitions_per_second=transitions / elapsed,
        updates=updates, supervised_transitions=transitions, history=history,
    )
    torch.save({"model": model.state_dict(), "metadata": metadata}, out / "model.pt")
    np.savez(out / "scores.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
