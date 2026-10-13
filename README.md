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
