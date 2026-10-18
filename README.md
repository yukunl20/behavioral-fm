# behavioral-fm

Self-supervised pretraining for a **behavioral foundation model** on MovieLens-1M user rating sequences.

Each interaction is treated as a multi-field event (`movie_id`, `rating`, `time_gap`, `genres`). The model learns representations by reconstructing fields after controlled corruption — comparing how the encoder is told about masks (`ignore` / `status_embed` / `mask_token`) and whether corruption is zero-fill or VIME-style donor replacement.

This repo is an early research scaffold: data pipeline, corruption/encoder, and a short pretrain + linear-probe loop are in place. A temporal backbone and fuller downstream eval are still ahead.

---

## Status

| Area | Status | Notes |
|------|--------|-------|
| MovieLens download & inspect | ✅ Done | `scripts/download_data.py`, `scripts/inspect_data.py` |
| Per-user sequence prep | ✅ Done | Chronological sequences → `data/processed/` |
| Dataset + pad collate | ✅ Done | `src/data/movielens.py` |
| Field corruption (zero / donor) | ✅ Done | `src/objectives/corruption.py` |
| Event encoder (base) | ✅ Done | Per-event field fusion |
| Status modes + recon heads | ✅ Done | `ignore` / `status_embed` / `mask_token` + mask heads in `MaskedEventModel` |
| Reconstruction decoder | ✅ Done | `--decoder linear|mlp` (default `mlp` residual trunk; probe still uses encoder `z`) |
| Pretrain losses + train loop | ✅ Done | `scripts/pretrain.py` (4 ablation modes) + `scripts/compare.py` |
| Smoke / unit checks | ✅ Done | `scripts/smoke_test.py`, `scripts/test.py` |
| Temporal backbone over events | ⬜ Not started | Transformer / sequence model |
| Downstream eval | ⬜ Not started | Beyond next-genre linear probe |
| Configs | ⬜ Not started | — |

### Ablations

| Name | Corruption | Encoder `status_mode` | Idea |
|------|------------|------------------------|------|
| `zero` | zero-fill | `ignore` | Masked values only; no explicit mask signal |
| `status` | zero-fill | `status_embed` | Zero-fill + learned status embedding |
| `token` | zero-fill | `mask_token` | MAE/BERT-style learned mask tokens |
| `vime` | donor replace | `ignore` | Hidden corruption label; predict which fields were corrupted |

### Results

Latest: `--max-steps 3000` with `--decoder mlp` (Exp 4 in [EXPERIMENTS.md](EXPERIMENTS.md)):

| mode | val_objective ↓ | val_mask_accuracy | next_genre_probe_micro_ap ↑ | embedding_mean_dim_std | effective_rank |
|------|----------------:|------------------:|----------------------------:|-----------------------:|---------------:|
| `zero` | **7.7867** | — | 0.3339 | 0.0765 | **24.10** |
| `status` | 7.7958 | — | 0.3329 | 0.0762 | 18.01 |
| `token` | 7.7990 | — | **0.3347** | 0.0757 | 22.18 |
| `vime` | 8.1114 | 0.7497 | 0.3158 | 0.0550 | 7.30 |

Lower `val_objective` is better for reconstruction; higher `next_genre_probe_micro_ap` is better for transfer. `val_mask_accuracy` only applies to VIME.

Compared with the earlier linear-head run @ 3000 steps, the MLP decoder raises effective rank (VIME 1.57→7.30) and VIME probe AP (0.284→0.316), while MAE probe scores stay ~0.33–0.335. Full history: [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Quick start

```bash
# Use the project env (has torch, etc.)
.behavior-fm/bin/python scripts/download_data.py
.behavior-fm/bin/python scripts/prepare_sequences.py

# Pretrain one ablation (writes artifacts/day06_<mode>_*)
# Default --decoder mlp (richer); use --decoder linear for the old heads-only baseline.
.behavior-fm/bin/python scripts/pretrain.py --mode zero --max-steps 3000
.behavior-fm/bin/python scripts/pretrain.py --mode status --max-steps 3000
.behavior-fm/bin/python scripts/pretrain.py --mode token --max-steps 3000
.behavior-fm/bin/python scripts/pretrain.py --mode vime --max-steps 3000

# Print a metrics table across modes
.behavior-fm/bin/python scripts/compare.py

# Bootstrap uncertainty on probe AP (all four modes)
.behavior-fm/bin/python scripts/bootstrap.py --prefix day06_mlp
```

