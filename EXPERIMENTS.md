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

## Exp 4 — Mode ablation @ 3000 steps with MLP decoder

**Date:** 2026-10-17  
**Goal:** Test whether a richer residual MLP reconstruction decoder improves transfer vs Exp 2 (linear heads only).  
**Command:**  
`python scripts/pretrain.py --mode <mode> --max-steps 3000 --decoder mlp --run-name day06_mlp_<mode>`  
**Artifacts:** `artifacts/day06_mlp_{zero,status,token,vime}_*`

| mode | val_objective | val_mask_accuracy | next_genre_probe_micro_ap | embedding_mean_dim_std | effective_rank |
|------|--------------:|------------------:|--------------------------:|-----------------------:|---------------:|
| zero | **7.7867** | — | 0.3339 | 0.0765 | **24.10** |
| status | 7.7958 | — | 0.3329 | 0.0762 | 18.01 |
| token | 7.7990 | — | **0.3347** | 0.0757 | 22.18 |
| vime | 8.1114 | 0.7497 | 0.3158 | 0.0550 | 7.30 |

**vs Exp 2 (linear heads @ 3000):**
- MAE modes: probe AP nearly flat (`zero` 0.3337→0.3339; `status`/`token` rise to ~0.33). Reconstruction slightly better for `zero`.
- Effective rank improves for `status`/`token` (10.8/12.3 → 18.0/22.2) and especially VIME (1.57 → **7.30**).
- VIME probe AP improves (0.284 → **0.316**) but still trails MAE modes.

**Takeaway:** The MLP decoder helps representation geometry (higher effective rank, less VIME collapse) and lifts VIME transfer somewhat, but does not change the overall ranking much: MAE modes still lead the next-genre probe around 0.33–0.335.

---

## Bugfix — VIME donor entity consistency

**Issue:** `movie_id` and `genres` share one corruption decision, but `donor_corrupt` used to call `_replace_with_donors` separately per field, each with a fresh donor permutation. A corrupted movie could get genres from a different donor event.

**Fix:** sample one shared `entity_donors` permutation for movie+genres; rating and time_gap keep independent donor perms. Covered by `test_donor_corrupt_keeps_movie_genre_entity`.

**Follow-up:** re-run VIME (Exp 2/4) after this fix; prior VIME numbers may be slightly optimistic/pessimistic due to inconsistent entities.

---

## Exp 5 — VIME after donor fix: linear vs MLP decoder @ 3000 steps

**Date:** 2026-10-17  
**Goal:** Re-measure VIME with entity-consistent donors; compare `--decoder linear` vs `--decoder mlp`.  
**Command:**  
`python scripts/pretrain.py --mode vime --max-steps 3000 --decoder <linear|mlp> --run-name day06_vime_fixed_<decoder>`  
**Artifacts:** `artifacts/day06_vime_fixed_{linear,mlp}_*`

| decoder | val_objective | val_mask_accuracy | next_genre_probe_micro_ap | embedding_mean_dim_std | effective_rank |
|---------|--------------:|------------------:|--------------------------:|-----------------------:|---------------:|
| linear | 8.1161 | 0.7490 | 0.2877 | 0.0836 | 1.56 |
| mlp | 8.1123 | 0.7496 | **0.3146** | 0.0544 | **7.21** |

**vs pre-fix VIME (same budget):**
- linear: probe 0.2843→0.2877, rank ~1.57→1.56 (essentially unchanged; still collapsed)
- mlp: probe 0.3158→0.3146, rank 7.30→7.21 (essentially unchanged)

**Takeaway:** Fixing movie/genre donor consistency does **not** materially change VIME metrics. The MLP decoder still helps VIME a lot vs linear heads (probe ~0.29→0.31, rank ~1.6→7.2), but VIME remains behind MAE `zero`/`token` (~0.33–0.335 probe, rank ~22–24).

---

## Exp 6 — Bootstrap probe-AP comparison (all four modes, MLP @ 3000)

**Date:** 2026-10-17  
**Goal:** Uncertainty on next-genre probe AP gaps for Exp 4 (`day06_mlp_*`).  
**Command:** `python scripts/bootstrap.py --prefix day06_mlp --baseline zero --samples 1000`

Point-estimate micro AP: token 0.3347, zero 0.3339, status 0.3329, vime 0.3158.

Pairwise vs baseline `zero` (treatment − baseline):

| mode | mean | 95% lo | 95% hi | P(diff > 0) |
|------|-----:|-------:|-------:|------------:|
| status | -0.0011 | -0.0063 | 0.0040 | 0.323 |
| token | +0.0008 | -0.0035 | 0.0052 | 0.624 |
| vime | -0.0179 | -0.0252 | -0.0104 | 0.000 |

P(mode has highest AP on a bootstrap sample): token 0.535, zero 0.318, status 0.147, vime 0.000.

**Takeaway:** Among MAE modes, `token`’s small edge over `zero` is **not** significant (95% CI covers 0). `vime` is significantly worse than `zero`. Do not over-claim a MAE winner from point estimates alone.

---

## Open follow-ups

- [x] Re-run mode ablation with `--decoder mlp` at 3000 steps (Exp 4); optional: explicit `--decoder linear` re-run for paired wall-clock compare
- [x] Re-run VIME after donor entity-consistency fix (Exp 5)
- [x] Bootstrap four-mode probe comparison (Exp 6)
- [ ] Add a shared diversity regularizer (e.g. VICReg-style variance / covariance) and re-run VIME vs `zero`
- [ ] Diagnose whether collapse is in per-event `z` or only after mean-pooling for the probe
- [ ] Longer budget (e.g. 10k steps) for non-VIME modes only, if rankings remain noisy
- [ ] Temporal backbone over events (next major model change)

---

## How to log a new experiment

1. Run with a distinct `--run-name` when sweeping hyperparameters.
2. Append a section here with date, goal, command, key metrics, and takeaway.
3. Point at artifact paths under `artifacts/` (do not commit large `.pt` / `.npy` files).
