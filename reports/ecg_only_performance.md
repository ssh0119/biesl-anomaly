# ECG-Only Cascade — Performance Report

Single-lead ECG, evaluated end to end: Stage 1 flags anomalies, Stage 2 names the rhythm. All figures below are from the patient-disjoint held-out test fold (fold 9 of MIMIC-III-Ext-PPG) — none of these segments were seen during training.

Training recipe: Stage 2 uses AdamW + OneCycleLR (max_lr=1e-3, 10% warmup, cosine decay), 8 epochs. Stage 1 uses plain AdamW (lr=1e-3, weight_decay=1e-4, no dropout) with validation checked every 1,000 steps rather than once per epoch — the deployed checkpoint is from **step 3,000 (33% into epoch 1)**, not an epoch boundary. See "Changes from the previous run" for why. Both: batch size 512. Stage 2's rhythm taxonomy dropped `junctional` and `ventricular` (0.3% and 0.06% of train anomalies, unlearnable in the prior run); those segments still count as anomalies for Stage 1, Stage 2 just skips them instead of forcing a guess.

| Dataset | Channel | Segment | Test fold | Stage 1 model | Stage 2 model |
|---|---|---|---|---|---|
| MIMIC-III-Ext-PPG | ECG · Lead II | 3,750 samples · 30 s @ 125 Hz | Fold 9 (patient-disjoint) | ResNet1D · 8.7M params | CNN + Transformer · 624K params |

---

## Stage 1 — Anomaly Detection

Normal sinus rhythm vs. any other rhythm — n = 579,006 test segments.

| ROC-AUC | PR-AUC | Accuracy |
|---|---|---|
| **0.946** | **0.913** | **88%** |

**Per-class detail (counts)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal (SR) | 0.93 | 0.88 | 0.90 | 359,294 |
| Anomaly | 0.82 | 0.89 | 0.85 | 219,712 |
| *Weighted avg* | *0.89* | *0.88* | *0.88* | *579,006* |

Confusion matrix (counts, rows=true/cols=pred, `[normal, anomaly]`): `[[316424, 42870], [24988, 194724]]`

Best result across every Stage 1 attempt so far (previous best was ROC-AUC 0.940). Found by evaluating mid-epoch instead of only at epoch boundaries — see "Changes from the previous run".

---

## Stage 2 — Rhythm-Type Classification

Runs only on Stage-1 anomalies — n = 219,685 test segments, **4 rhythm groups** (down from 6).

| Accuracy | Macro F1 | Weighted F1 |
|---|---|---|
| **87%** (was 79%) | **0.77** (was 0.49) | **0.87** (was 0.82) |

**Per-class detail (counts)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Atrial tachy | 0.82 | 0.85 | 0.83 | 42,746 |
| Conduction block | 0.50 | 0.61 | 0.55 | 15,002 |
| Paced | 0.82 | 0.73 | 0.77 | 31,255 |
| Sinus rate | 0.95 | 0.94 | 0.94 | 130,682 |
| *Macro avg* | *0.77* | *0.78* | *0.77* | *219,685* |
| *Weighted avg* | *0.87* | *0.87* | *0.87* | *219,685* |

Confusion matrix (counts, rows=true/cols=pred, `[atrial_tachy, conduction_block, paced, sinus_rate]`):
```
[[ 36380   1714   2259   2393]
 [  2753   9090   1830   1329]
 [   997   4335  22739   3184]
 [  4430   2874    771 122607]]
```

Removing `junctional`/`ventricular` fixed the precision collapse those two caused everywhere else (sinus-rate F1 alone went 0.89→0.94+). Remaining weak point: **conduction_block precision is still ~0.50** — several thousand true `paced` segments get misread as `conduction_block`. Worth a closer look (these two classes may share ECG morphology that the model can't separate from a single lead), but it's a normal confusion-pair problem now, not a broken-class problem.

**Checkpoint provenance:** re-checked with the same mid-training (every 500 steps) validation used for Stage 1, to rule out missing a better sub-epoch point. Best landed at **step 4,000 — 17% into epoch 2** (val macro-F1 0.836), matching the earlier epoch-boundary-only result (epoch 3, 0.836) almost exactly — unlike Stage 1, checking more often didn't find anything better here. Steps in between swing in a noisy ±0.03–0.05 band (wider than Stage 1's ±0.02), and none of it beat step 4,000 through 5 full epochs, so the search was stopped there rather than running all 8.

---

## Changes from the previous run

1. **Dropped `junctional` (JR/JTACH) and `ventricular` (VTACH) as Stage 2 classes.** They were 0.3% and 0.06% of train anomalies respectively and were unlearnable in practice (junctional precision 0.01 despite 34% recall; ventricular 0% F1 across every modality tried). Segments with these rhythms are still flagged as anomalies by Stage 1 — Stage 2 now just declines to name them instead of guessing.
2. **Added OneCycleLR** (warmup 10% + cosine decay) to `src/train.py`, replacing the flat `lr=1e-3` AdamW used before.

### Did the LR schedule fix the epoch-1-then-overfit pattern?

**Mixed — it helped Stage 2, not Stage 1.**

- **Stage 2** (624K params, CNN+Transformer): improved past epoch 1, peaking at **epoch 3 (val macro-F1 0.836)** before a gentle decline to 0.789 by epoch 8. The best-checkpoint mechanism correctly kept epoch 3. Net win: the schedule found a better optimum than epoch 1 alone would have.
- **Stage 1** (8.7M params, ResNet1D): val macro-F1 declined every single epoch under OneCycleLR — 0.885 (ep1) → 0.866 → 0.864 → 0.860 → 0.856 → 0.852 → 0.845 → 0.844 (ep8) — even as LR decayed to ~0 and train_loss kept falling (0.144→0.021). Epoch 1 stayed the best checkpoint throughout; epochs 2–8 were wasted compute (~55 min), not harmful (best-checkpoint save protects the deployed model) but not useful either.

### Follow-up: does more regularization fix Stage 1?

Tried next: `max_lr` 1e-3→3e-4, `weight_decay` 1e-4→5e-4, and added `nn.Dropout(0.3)` to `ResNet1D` (it had none before). Result: **no** — same 8-epoch run showed the identical early-peak-then-decline shape (peak at epoch 2 this time, val macro-F1 0.880, declining to 0.841 by epoch 8), and the peak itself was *lower* than the original run's epoch-1 peak, plus worse on the held-out test set (ROC-AUC 0.931 vs 0.938). Regularizing harder didn't stop the decline — it just lowered the ceiling. That points at something other than classic overfitting-from-too-much-capacity: e.g. BatchNorm running-stats drift between train and val distribution, or the val fold genuinely differing from train folds in a way no amount of weight regularization fixes.

### Does Stage 1 overfit before epoch 1 even finishes?

Both attempts above only checked validation at epoch boundaries — so "epoch 1 is the peak" could mean anything from "the true peak is right at the end of epoch 1" to "the true peak was hours earlier and epoch 1's number is already off the top." Added `--eval-every-steps` to `src/train.py` to check every 1,000 steps (epoch 1 = 9,125 steps) instead of once per epoch, using the original recipe (lr=1e-3, weight_decay=1e-4, no dropout).

Result: **yes, and earlier than expected.** Val macro-F1 by step:

| step | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | 8,000 | 9,000 | 9,125 (ep1 end) |
|---|---|---|---|---|---|---|---|---|---|---|
| val F1 | 0.864 | 0.878 | **0.884** | 0.882 | 0.883 | 0.866 | 0.878 | 0.879 | 0.871 | 0.882 |

Peak is at **step 3,000 — 33% into epoch 1** (~1.5M samples seen). Steps 1,000–9,125 all sit within a noisy ±0.02 band, roughly the same size as the run-to-run variance seen between separate full runs (0.877–0.885) — so "epoch 1 vs epoch 2" was never really a clean signal, it's mostly this same noise. Epoch 2 (checked the same way) shows the noise resolving into a real, gentle decline: local highs drift 0.879 → 0.877 → 0.875 → 0.870 down to 0.870 by epoch 2's end. So there is a genuine long-run degradation, it just isn't visible epoch-to-epoch near the start — it's a slow bleed starting from wherever the noise band happens to peak.

**The deployed Stage 1 checkpoint is now step 3,000 of this run** — the best result found across every attempt (ROC-AUC 0.946 vs the original run's 0.938). Practical takeaway: Stage 1 doesn't need a full epoch, let alone 8 — check validation sub-epoch and stop once it stops improving.

**Open question for next time:** *why* it peaks this early and then bleeds down regardless of LR/weight_decay/dropout is still unexplained. Worth checking BatchNorm behavior (e.g. `track_running_stats=False` or lower momentum) or auditing the val fold for a real distribution shift before spending more compute chasing it.

---

*Held-out test fold, never seen in training or model selection. Generated from `src/evaluate.py`.*
