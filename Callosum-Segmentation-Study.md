# Corpus Callosum Segmentation from Diffusion MRI

**Does anatomical masking or fibre-orientation input improve a callosum segmenter?**
A 15-run factorial over five input sets and three white-matter mask settings, on 144 HCP subjects.

---

## Overview

The task is to segment the corpus callosum on the mid-sagittal slice of a diffusion MRI volume. A 2D U-Net trained on fractional anisotropy alone already reaches Dice ≈ 0.955, so the question is not whether the problem is solvable but **what additional information actually helps**.

Two ideas were tested against a control, as a full factorial:

1. **Anatomical masking** — restrict the network input to white matter, segmented from the subject's own T1 with FSL FAST. The network currently searches the whole slice, including grey matter and CSF, where callosum cannot be.
2. **Fibre orientation** — feed the diffusion tensor, rather than only scalar metrics. FA, MD, AD and RD are all functions of the tensor eigenvalues, so all four are blind to direction, and the structures most easily confused with the callosum (fornix, cingulum) differ from it mainly by direction: callosal fibres cross the midline, out of the sagittal plane, while fornix and cingulum run within it.

Both were tested at 500 epochs per run, with the binarisation threshold selected on validation and applied unchanged to a held-out test split.

---

## Data

A 144-subject HCP-derived cohort at 1.25 mm isotropic, volumes of 145 × 174 × 145, split **at subject level** so no subject appears in more than one split:

| Split | Subjects |
|---|---|
| Train | 115 |
| Validation | 14 |
| Test | 15 |

Each subject provides FA, MD, AD, RD, the full diffusion tensor as eigenvalues and eigenvectors, a mean b0, a skull-stripped T1 already resampled to the diffusion grid, and an MRIcloud corpus callosum mask as ground truth.

### One data-preparation finding worth reporting

The diffusion volumes are uniformly in `LAS` orientation, but **the T1 and the ground-truth mask are not** — across the first 20 subjects they appear in **ten different orientations** (`SAL`, `IPR`, `LPS`, `SPR`, `IAR`, …). Every scalar volume is therefore reoriented to `LAS` before use.

This is not a cosmetic detail. Loading the T1 as-is drops its correlation with FA inside the brain from ≈0.40 to ≈0.13, and would mirror any T1-derived mask onto the wrong hemisphere. An early version of this pipeline had exactly that bug, and its T1 results were meaningless until it was fixed.

---

## Method

### Pipeline

1. **Preprocess** — load the requested channels, normalise each on its own fixed scale, extract the mid-sagittal slice, optionally apply the white-matter mask, cache to `.npz`.
2. **Train** — 2D U-Net, residual double-conv blocks, four downsampling stages, `init_channels=32`. Loss is Dice + BCE-with-logits. Adam at `lr=5e-5`, 500 epochs, checkpoint selected on validation loss.
3. **Evaluate** — sweep the binarisation threshold on validation, report Dice, Jaccard and Hausdorff distance.
4. **Test** — apply the validation-selected threshold to the held-out split.

Augmentation is a ±10° rotation in the sagittal plane plus a random 64×64 crop. When the raw tensor is an input, the rotation also rotates the tensor components themselves (`D' = R D Rᵀ`) — a tensor field does not transform like a scalar field, and omitting this trains the network on fibres pointing in directions that do not exist in the image.

### White-matter mask

```bash
fast -t 1 -n 3 -N -o wm/t1  T1_brain_1.25.nii   # t1_pve_2.nii.gz = white matter
```

`pve_2` is a **partial-volume fraction** in [0, 1], not a label map — 40% of its non-zero voxels lie strictly between 0 and 1, at tissue boundaries — so it needs a cut. At 0.5 the mask reproduces FAST's own hard segmentation (Dice 0.982 against `seg == 3`) and discards 2.2% of callosum voxels; at 0.3 it discards 0.8%. Voxels the mask removes are targets the network can never predict, so each threshold sets a ceiling on Dice before training starts.

Masking is applied **after** slice extraction, so it cannot change which slice is selected. Segmentations are cached per subject, computed once for the cohort in ~30 minutes at four parallel jobs, and reused across all experiments.

### Input components

| Component | Channels | Carries orientation? |
|---|---|---|
| `fa` | 1 | No — eigenvalue function |
| `md`, `ad`, `rd` | 1 each | No — eigenvalue functions |
| `t1` | 1 | No — non-diffusion contrast |
| `tensor` | 6 | Yes, in the scanner frame |
| `tensor_inv` | 5 | Yes, as rotation invariants |

`tensor_inv` recodes the same six numbers as five quantities invariant under the augmentation's rotation — through-plane fraction `Dxx/tr(D)`, trace, in-plane/through-plane coupling, and two shape terms. The raw tensor forces the network to learn that invariance from examples; the invariants hand it over by construction.

---

## Results

Five input sets × three mask settings = 15 runs, all completed. Test-set means, threshold selected on validation.

| Input | Mask | Threshold | Val Dice | **Test Dice** | SD | Test Jaccard | Test HD |
|---|---|---|---|---|---|---|---|
| `fa tensor_inv t1` | — | 0.05 | 0.9620 | **0.9648** | 0.0114 | 0.9323 | 1.12 |
| `fa tensor_inv t1` | 0.5 | 0.95 | 0.9591 | 0.9636 | 0.0104 | 0.9299 | 1.09 |
| `fa tensor_inv` | 0.5 | 0.10 | 0.9594 | 0.9628 | 0.0100 | 0.9285 | 1.09 |
| `fa` | 0.5 | 0.20 | 0.9583 | 0.9623 | 0.0093 | 0.9275 | 1.09 |
| `fa tensor_inv t1` | 0.3 | 0.40 | 0.9600 | 0.9621 | 0.0096 | 0.9272 | 1.09 |
| `fa` | 0.3 | 0.65 | 0.9587 | 0.9615 | 0.0100 | 0.9260 | 1.07 |
| `tensor_inv` | 0.5 | 0.50 | 0.9589 | 0.9608 | 0.0110 | 0.9248 | 1.09 |
| `tensor_inv` | 0.3 | 0.15 | 0.9576 | 0.9607 | 0.0088 | 0.9246 | 1.09 |
| `tensor` | 0.5 | 0.45 | 0.9583 | 0.9606 | 0.0106 | 0.9244 | 1.16 |
| `fa tensor_inv` | 0.3 | 0.55 | 0.9591 | 0.9601 | 0.0095 | 0.9234 | 1.07 |
| `tensor` | 0.3 | 0.15 | 0.9588 | 0.9580 | 0.0114 | 0.9196 | 1.09 |
| `tensor` | — | 0.05 | 0.9549 | 0.9565 | 0.0125 | 0.9169 | 1.19 |
| `fa` | — | 0.10 | 0.9551 | 0.9550 | 0.0129 | 0.9142 | 1.08 |
| `fa tensor_inv` | — | 0.05 | 0.9570 | 0.9550 | 0.0123 | 0.9142 | 1.06 |
| `tensor_inv` | — | 0.05 | 0.9553 | 0.9540 | 0.0128 | 0.9123 | 1.24 |

### Main effects

| Mask | Mean test Dice | | Input | Mean test Dice |
|---|---|---|---|---|
| none | 0.9571 | | `fa` | 0.9596 |
| 0.3 | 0.9605 | | `tensor` | 0.9584 |
| **0.5** | **0.9620** | | `tensor_inv` | 0.9585 |
| | | | `fa tensor_inv` | 0.9593 |
| | | | **`fa tensor_inv t1`** | **0.9635** |

---

## Findings

**1. White-matter masking helps, and it helps consistently.** Averaged over inputs, masking gains +0.005 Dice, and the sign is the same for four of the five input sets (+0.004 to +0.008). It also makes performance *more uniform across subjects*: per-subject SD drops from 0.0114–0.0129 unmasked to 0.0088–0.0114 masked. Being consistent in sign across independent input sets makes this the most trustworthy result here, more so than any single run's margin.

**2. The tighter mask won, contrary to expectation.** The 0.5 threshold beat 0.3 (0.9620 vs 0.9605), even though it discards nearly three times as many callosum voxels (2.2% vs 0.8%). The Dice ceiling that discarding imposes is evidently smaller than the benefit of removing more irrelevant tissue.

**3. Fibre orientation did not help — the second hypothesis is not supported.** `tensor` (0.9584) and `tensor_inv` (0.9585) both land *below* plain `fa` (0.9596), and adding `tensor_inv` to `fa` changes nothing (0.9593). The rotation-invariant recoding performed no better than the raw tensor. The reasoning behind the hypothesis still holds — scalar metrics genuinely cannot distinguish callosum from fornix by direction — but at this resolution a U-Net with enough spatial context evidently resolves that ambiguity from shape and position alone, without needing the orientation made explicit.

**4. The one input that mattered was T1, and it makes the mask redundant.** `fa tensor_inv t1` is the strongest input set (0.9635 mean) and produced the single best run (**0.9648**). It is also the *only* input set where masking made things worse (−0.001 to −0.003). That is coherent rather than contradictory: the mask is *derived from* the T1, so once the network sees the T1 directly it can learn the tissue boundary itself — with soft, graded intensities — while the mask can only impose a hard, lossy version of the same information.

The practical recommendation that follows: **give the network the T1, or give it the mask, but there is no gain in doing both.**

---

## Limitations

- **One run per cell.** Repeated runs of identical configurations earlier in this project spread up to 0.0135 Dice — larger than most differences in the table. Individual row-to-row gaps should not be over-read; only the mask effect (consistent in sign across four input sets) and the T1 effect (the largest single margin) are argued here.
- **Fifteen test subjects**, with a per-subject SD around 0.010, giving a standard error near 0.003.
- **The threshold sweep operates on logits.** Inference omits the sigmoid, so a nominal sweep over 0.05–0.95 explores only ≈0.51–0.72 in probability. The sweep is consequently near-flat and its selected threshold is erratic — note the values ranging from 0.05 to 0.95 in the table, with no effect on Dice. Reported Dice is therefore measured at a suboptimal cut, equally for every run, so comparisons hold while absolute values are pessimistic.
- **Hausdorff distance is uninformative here** (1.06–1.24 voxels across all runs) and does not separate the configurations.

---

## Reproducing

```bash
# 1. White-matter masks, once for the whole cohort (~30 min, cached)
python extract_wm.py --jobs 4

# 2. A single configuration
python brainhack_challenge.py --input fa tensor_inv t1 --epochs 500

# 3. The full 15-run factorial
python brainhack_challenge_tests.py
```

Every run writes to its own experiment directory under `logs/`, keyed by input set and mask setting, containing the metric report, TensorBoard curves, the checkpoint, and prediction figures (input, ground truth, continuous prediction, thresholded prediction).

| File | Role |
|---|---|
| `brainhack_challenge.py` | 2D pipeline: preprocess, train, evaluate, test |
| `brainhack_challenge_3d.py` | 3D variant — full 128³ volume, patch-trained U-Net |
| `extract_wm.py` | Parallel, cached FSL FAST white-matter extraction |
| `brainhack_challenge_tests.py` | Factorial runner over inputs × mask thresholds |

---

## Next steps

- **Repeat the leading configurations across seeds.** Run-to-run variance currently exceeds most of the effects measured, and this is the cheapest way to firm up the ranking.
- **Apply the sigmoid at inference** so the threshold sweep operates on probabilities and the reported Dice is measured at a genuinely optimal cut.
- **Move to 3D.** The callosum is a volumetric structure and the single-slice formulation discards most of it; a patch-trained 3D U-Net over the cropped 128³ volume is implemented and ready to evaluate.
