# behavioral-fm

Self-supervised pretraining for a **behavioral foundation model** on MovieLens-1M user rating sequences.

Each interaction is treated as a multi-field event (`movie_id`, `rating`, `time_gap`, `genres`). The model learns representations by reconstructing fields after controlled corruption — comparing how the encoder is told about masks (none / status embedding / learned mask token) and whether corruption is zero-fill or VIME-style donor replacement.

This repo is an early research scaffold: data pipeline and corruption/encoder pieces are in place; full pretraining and evaluation are still ahead.

---

## Status

| Area | Status | Notes |
|------|--------|-------|
| MovieLens download & inspect | ✅ Done | `scripts/download_data.py`, `scripts/inspect_data.py` |
| Per-user sequence prep | ✅ Done | Chronological sequences → `data/processed/` |
| Dataset + pad collate | ✅ Done | `src/data/movielens.py` |
| Field corruption (zero / donor) | ✅ Done | `src/objectives/corruption.py` |
| Event encoder (base) | ✅ Done | Per-event field fusion |
| Status modes + recon heads | 🚧 In progress | `none` / `add` / `token` + mask heads |
| Pretrain losses + train loop | ⬜ Not started | — |
| Temporal backbone over events | ⬜ Not started | Transformer / sequence model |
| Downstream eval | ⬜ Not started | `src/eval/` stub |
| Configs & tests | ⬜ Not started | — |

### Planned ablations

| Name | Corruption | Encoder `status_mode` | Idea |
|------|------------|------------------------|------|
| `zero` | zero-fill | `none` | Masked values only; no explicit mask signal |
| `status` | zero-fill | `add` | Zero-fill + learned status embedding |
| `token` | zero-fill | `token` | MAE/BERT-style learned mask tokens |
| `vime` | donor replace | `none` | Hidden corruption label; predict which fields were corrupted |
