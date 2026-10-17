# Experiment tracker

Log of pretraining / ablation runs for the MovieLens-1M behavioral foundation model scaffold.

Shared defaults unless noted: `batch_size=32`, `mask_prob=0.25`, `d_model=64`, `probe_epochs=200`, AdamW `lr=3e-4`, seed `7`, device CPU for the runs below. Artifacts land under `artifacts/` (local only).

---

## Exp 1 — Mode ablation @ 300 steps

**Date:** 2026-10-12  
**Goal:** Sanity-check `zero` / `status` / `token` / `vime` with the default short budget (~2 epochs).  
**Command:** `python scripts/pretrain.py --mode <mode>` (default `--max-steps 300`)

| mode | val_objective | val_mask_accuracy | next_genre_probe_micro_ap | effective_rank | notes |
|------|--------------:|------------------:|--------------------------:|---------------:|-------|
| zero | 7.9333 | — | **0.3380** | 19.00 | best probe |
| status | 7.9441 | — | 0.3282 | 21.05 | |
| token | 7.9457 | — | 0.3299 | 19.25 | |
| vime | 8.2684 | 0.7499 | 0.3155 | 14.53 | mask head already ~0.75 |

**Takeaway:** Gaps are tiny; 300 steps is too short to rank modes. VIME already trails on probe AP.

---

## Exp 2 — Mode ablation @ 3000 steps

**Date:** 2026-10-17  
**Goal:** Fairer comparison at ~20 epochs.  
**Command:** `python scripts/pretrain.py --mode <mode> --max-steps 3000`  
**Artifacts:** `artifacts/day06_{zero,status,token,vime}_*`

| mode | val_objective | val_mask_accuracy | next_genre_probe_micro_ap | embedding_mean_dim_std | effective_rank |
|------|--------------:|------------------:|--------------------------:|-----------------------:|---------------:|
| zero | 7.8116 | — | **0.3337** | 0.0914 | **22.46** |
| status | 7.8014 | — | 0.3249 | 0.0878 | 10.79 |
| token | **7.7904** | — | 0.3249 | 0.0938 | 12.32 |
| vime | 8.1149 | 0.7506 | 0.2843 | 0.0809 | **1.57** |

**Takeaway:** `token` wins reconstruction; `zero` still leads the next-genre probe. VIME probe drops further and effective rank collapses (~1.6), suggesting representation collapse rather than a mild loss of the ablation.

---

## Exp 3 — VIME mask-loss weight sweep @ 3000 steps

**Date:** 2026-10-17  
**Goal:** Test whether the VIME auxiliary mask loss weight causes collapse.  
**Command:**  
`python scripts/pretrain.py --mode vime --max-steps 3000 --mask-loss-weight <w> --run-name day06_vime_w<tag>`  
**Weights:** `0.0, 0.1, 0.25, 0.5, 1.0`  
**Artifacts:** `artifacts/day06_vime_w{0,0p1,0p25,0p5,1}_*`

| mask_loss_weight | val_objective | val_mask_accuracy | next_genre_probe_micro_ap | effective_rank |
|-----------------:|--------------:|------------------:|--------------------------:|---------------:|
| 0.0 | 7.8349 | 0.2522 | **0.2925** | 1.63 |
| 0.1 | 7.8911 | 0.7506 | 0.2897 | 1.62 |
| 0.25 | 7.9752 | 0.7506 | 0.2859 | 1.59 |
| 0.5 | 8.1149 | 0.7506 | 0.2843 | 1.57 |
| 1.0 | 8.3978 | 0.7506 | 0.2865 | 1.62 |

**Takeaway:** Effective rank stays ~1.6 even at weight `0.0` (reconstruction-only VIME). Mask accuracy jumps to ~0.75 for any weight ≥ 0.1. Lowering the mask-loss weight does **not** fix collapse; donor corruption / representation geometry is the likelier cause. Best probe in this sweep is still well below `zero` @ 3000 steps (0.33).

---

## Open follow-ups

- [ ] Add a shared diversity regularizer (e.g. VICReg-style variance / covariance) and re-run VIME vs `zero`
- [ ] Diagnose whether collapse is in per-event `z` or only after mean-pooling for the probe
- [ ] Longer budget (e.g. 10k steps) for non-VIME modes only, if rankings remain noisy
- [ ] Temporal backbone over events (next major model change)

---

## How to log a new experiment

1. Run with a distinct `--run-name` when sweeping hyperparameters.
2. Append a section here with date, goal, command, key metrics, and takeaway.
3. Point at artifact paths under `artifacts/` (do not commit large `.pt` / `.npy` files).
