"""Train BehavioralSequenceModel (causal / masked / prefix) and save a checkpoint.

Example (frozen EventEncoder reusable by train_behavioral_state):
  python -m scripts.train_behavioral_sequence --objective causal --experiment zero \\
    --mask-probability 0.15 --epochs 3 --probe-epochs 3 \\
    --split-file artifacts/day10/split.json --create-split
→ artifacts/day08/causal_zero_p0.15.pt
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.config.experiment_modes import experiment_spec
from src.data.movielens import MovieLensSequenceDataset, collate_sequences
from src.models.sequence import BehavioralSequenceModel
from src.objectives.sequence_tasks import (
    causal_next_genre_loss,
    clean_status,
    masked_item_loss,
    strict_prefix,
    strict_prefix_loss,
)
from scripts.train_behavioral_state import RecentSequences, file_hash, get_split


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def training_loss(model, batch, objective, experiment, mask_probability):
    if objective == "causal":
        return causal_next_genre_loss(model, batch)  # corruption / mask_prob unused
    if objective == "masked":
        return masked_item_loss(model, batch, experiment, mask_probability)
    if objective == "prefix":
        return strict_prefix_loss(model, batch, random_cut=True)
    raise ValueError(objective)


@torch.no_grad()
def prefix_embeddings(model, batch, direction):
    # Fixed final-event target for probes (not random cut).
    prefix, cuts, target = strict_prefix(batch, random_cut=False)
    hidden = model(prefix, clean_status(prefix), direction=direction)
    row = torch.arange(len(cuts), device=cuts.device)
    return hidden[row, cuts - 1], target  # [B, D], [B, G]


def micro_f1(logits, target):
    prediction = torch.sigmoid(logits) >= 0.5
    truth = target.bool()
    tp = (prediction & truth).sum().float()
    fp = (prediction & ~truth).sum().float()
    fn = (~prediction & truth).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


@torch.no_grad()
def evaluate_probe(model, probe, loader, device, direction):
    model.eval()
    probe.eval()
    scores, labels = [], []
    for batch in loader:
        batch = move(batch, device)
        z, y = prefix_embeddings(model, batch, direction)
        scores.append(probe(z))
        labels.append(y)
    if not scores:
        raise ValueError("empty validation loader")
    return micro_f1(torch.cat(scores), torch.cat(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=["causal", "masked", "prefix"], required=True)
    parser.add_argument(
        "--experiment",
        choices=["zero", "status", "token", "vime"],
        default="zero",
        help="Ablation label → status_mode / corruption (causal ignores corruption).",
    )
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split-file", default="artifacts/day10/split.json")
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="artifacts/day08")
    args = parser.parse_args()

    if args.epochs < 1 or args.probe_epochs < 0 or args.batch_size < 1 or args.max_length < 2:
        parser.error("need epochs>=1, probe_epochs>=0, batch_size>=1, max_length>=2")
    if not 0.0 < args.mask_probability < 1.0:
        parser.error("mask-probability must be in (0, 1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device()
    spec = experiment_spec(args.experiment)

    processed = Path(args.processed_dir)
    fingerprint = hashlib.sha256(
        (
            file_hash(processed / "sequences.npy")
            + file_hash(processed / "metadata.json")
            + f":T={args.max_length}"
        ).encode()
    ).hexdigest()
    base = MovieLensSequenceDataset(processed)
    dataset = RecentSequences(base, args.max_length)
    split_path = Path(args.split_file)
    manifest = get_split(
        split_path, fingerprint, len(dataset), args.create_split, args.split_seed,
    )
    train_set = Subset(dataset, manifest["train"])
    val_set = Subset(dataset, manifest["validation"])
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    meta = base.metadata
    num_movies = int(meta["num_movies"])
    num_genres = int(meta["num_genres"])
    model = BehavioralSequenceModel(
        num_movies=num_movies,
        num_genres=num_genres,
        status_mode=spec.status_mode,
        d_model=args.d_model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move(batch, device)
            loss = training_loss(
                model, batch, args.objective, args.experiment, args.mask_probability,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"pretrain epoch={epoch + 1} loss={np.mean(losses):.4f}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()  # freeze dropout before any probe forward

    probe = nn.Linear(args.d_model, num_genres).to(device)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    direction = "causal" if args.objective == "causal" else "bidirectional"
    score = None
    for epoch in range(args.probe_epochs):
        probe.train()
        losses = []
        for batch in train_loader:
            batch = move(batch, device)
            with torch.no_grad():
                z, target = prefix_embeddings(model, batch, direction)
            logits = probe(z)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            probe_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            probe_optimizer.step()
            losses.append(float(loss.detach()))
        score = evaluate_probe(model, probe, val_loader, device, direction)
        print(
            f"probe epoch={epoch + 1} "
            f"loss={np.mean(losses):.4f} val_micro_f1={score:.4f}"
        )
    if score is None and args.probe_epochs == 0:
        score = evaluate_probe(model, probe, val_loader, device, direction)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Filename bookkeeping: p{mask} even when causal ignores corruption.
    name = f"{args.objective}_{args.experiment}_p{args.mask_probability}"
    ckpt_path = output / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "probe": probe.state_dict(),
            "args": vars(args),
            "status_mode": spec.status_mode,
            "data_hash": fingerprint,
            "split_file": str(split_path),
            "split_hash": file_hash(split_path),
        },
        ckpt_path,
    )
    with open(output / f"{name}.json", "w") as handle:
        json.dump(
            {
                **vars(args),
                "status_mode": spec.status_mode,
                "val_probe_micro_f1": score,
                "data_hash": fingerprint,
                "checkpoint": str(ckpt_path),
            },
            handle,
            indent=2,
        )
    print("saved", ckpt_path, "device", device)


if __name__ == "__main__":
    main()
