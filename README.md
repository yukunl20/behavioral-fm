# behavioral-fm

Self-supervised pretraining for a **behavioral foundation model** on MovieLens-1M user rating sequences.

Each interaction is treated as a multi-field event (`movie_id`, `rating`, `time_gap`, `genres`). The model learns representations by reconstructing fields after controlled corruption — comparing how the encoder is told about masks (none / status embedding / learned mask token) and whether corruption is zero-fill or VIME-style donor replacement.

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
| Status modes + recon heads | ✅ Done | `none` / `add` / `token` + mask heads in `MaskedEventModel` |
| Pretrain losses + train loop | ✅ Done | `scripts/pretrain.py` (4 ablation modes) + `scripts/compare.py` |
| Smoke / unit checks | ✅ Done | `scripts/smoke_test.py`, `scripts/test.py` |
| Temporal backbone over events | ⬜ Not started | Transformer / sequence model |
| Downstream eval | ⬜ Not started | Beyond next-genre linear probe |
| Configs | ⬜ Not started | — |

### Ablations

| Name | Corruption | Encoder `status_mode` | Idea |
|------|------------|------------------------|------|
| `zero` | zero-fill | `none` | Masked values only; no explicit mask signal |
| `status` | zero-fill | `add` | Zero-fill + learned status embedding |
| `token` | zero-fill | `token` | MAE/BERT-style learned mask tokens |
| `vime` | donor replace | `none` | Hidden corruption label; predict which fields were corrupted |

### Results

`--max-steps 3000` (~20 epochs over the train split) via `scripts/pretrain.py` + `scripts/compare.py`:

| mode | val_objective ↓ | val_mask_accuracy | next_genre_probe_micro_ap ↑ | embedding_mean_dim_std | effective_rank | steps_per_second |
|------|----------------:|------------------:|----------------------------:|-----------------------:|---------------:|-----------------:|
| `zero` | 7.8116 | — | **0.3337** | 0.0914 | **22.46** | 19.7 |
| `status` | 7.8014 | — | 0.3249 | 0.0878 | 10.79 | 20.3 |
| `token` | **7.7904** | — | 0.3249 | 0.0938 | 12.32 | 19.8 |
| `vime` | 8.1149 | 0.7506 | 0.2843 | 0.0809 | 1.57 | 19.9 |

Lower `val_objective` is better for reconstruction; higher `next_genre_probe_micro_ap` is better for transfer. `val_mask_accuracy` only applies to VIME.

At this budget, `token` wins reconstruction, `zero` still leads the next-genre probe, and VIME’s very low effective rank suggests representation collapse rather than a fair loss of the ablation.

---

## Quick start

```bash
# Use the project env (has torch, etc.)
.behavior-fm/bin/python scripts/download_data.py
.behavior-fm/bin/python scripts/prepare_sequences.py

# Pretrain one ablation (writes artifacts/day06_<mode>_*)
.behavior-fm/bin/python scripts/pretrain.py --mode zero
.behavior-fm/bin/python scripts/pretrain.py --mode status
.behavior-fm/bin/python scripts/pretrain.py --mode token
.behavior-fm/bin/python scripts/pretrain.py --mode vime

# Print a metrics table across modes
.behavior-fm/bin/python scripts/compare.py
```
