# Multi-Agent RL for Cephalometric Landmark Detection

One RL agent per landmark (19 on ISBI 2015, 29 on Aariz). All agents share one network,
start at the mean skull shape, see a 64×64 patch plus every agent's position, and walk a
4-level image pyramid (0.8 → 0.1 mm/px), zooming in by rule when they settle.
DQN, MAPPO and COMA are compared with a U-Net heatmap baseline, on ISBI 2015, on Aariz,
and ISBI → Aariz (unseen devices). A calibrated uncertainty score flags landmarks for review.

The design rationale (what was kept, dropped and why) is in the project's architecture doc.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install a CUDA build of torch for GPU training
bash scripts/smoke_test.sh               # ~5-10 min on CPU: tests + every method end to end on fake data
```

The smoke test only proves every step runs: its budget is tiny, so its errors (20–35 mm) mean nothing.
To check that agents actually learn, use a longer synthetic run (CPU, roughly 20–30 min each):

```bash
python scripts/train.py --config configs/synthetic_smoke.yaml --out runs/learn_dqn \
    --set algo=dqn agent_steps=300000 eval_every=30000 dqn.buffer=30000 dqn.warmup=2000
python scripts/train.py --config configs/synthetic_smoke.yaml --out runs/learn_mappo \
    --set algo=mappo agent_steps=60000 eval_every=15000
```
Reference results on the synthetic set (validation MRE): placing the mean shape gives 3.3 mm;
DQN reached 2.0 mm after 300k agent-steps; MAPPO went from 19.5 to 5.7 mm in 60k agent-steps.
Policy-gradient methods need several updates per sample: keep `minibatch` small (32).

## Data

Both datasets are converted once into a common cache: grayscale `.npy` images resampled to
**0.1 mm/px** plus a `meta.json` with landmarks (junior, senior and their mean).

**ISBI 2015** (400 images, 19 landmarks, two doctors). Expected layout of the official release:

```
ISBI2015/RawImage/{TrainingData,Test1Data,Test2Data}/001.bmp ...
ISBI2015/.../400_junior/001.txt, .../400_senior/001.txt   (folders are searched for; extra nesting is fine)
```
```bash
python scripts/prepare_data.py isbi --raw /path/to/ISBI2015 --out data/isbi
```
Splits written: `train` (130), `val` (20, held out from the 150 training images, seed 0), `test1` (150), `test2` (100).

**Aariz** (1000 images, 7 devices, 29 landmarks, junior + senior). Download from
[figshare](https://doi.org/10.6084/m9.figshare.27986417.v1); layout as in the
[official repo](https://github.com/manwaarkhd/aariz):

```
Aariz/{train,valid,test}/Cephalograms/<id>.jpg      (folder-name case does not matter)
Aariz/{train,valid,test}/Annotations/Cephalometric Landmarks/{Senior,Junior} Orthodontists/<id>.json
```
Images come at different resolutions, so each needs its pixel spacing (mm/px). The release ships a CSV
with the imaging device and pixel size (mm) of every cephalogram; pass it with `--spacing-csv`
(id, pixel-size and device columns are detected by name, and printed so you can check them):
```bash
python scripts/prepare_data.py aariz --raw /path/to/Aariz --out data/aariz   # finds cephalogram_machine_mappings.csv itself
```
> **Aariz pixel-size correction.** For 4 of the 7 devices the CSV pixel size does not fit the released
> image files (Sella–Nasion came out at 24–118 mm instead of ~72 mm). `prepare_data.py` corrects them
> per device (`AARIZ_SPACING_RULES` in `cephmarl/data.py`): ART Plus and both ProMax use the CSV;
> Hyperion X5 images were downscaled by varying factors, so the CSV size is scaled by native/actual
> height; Smart3D, Veraviewepocs 2D and Rotograph EVO are calibrated from Sella–Nasion on the train
> split (an estimate, about ±10%). Evaluation reports errors both ways: corrected, and with the
> official CSV pixel sizes (`model_official_csv_pixel_size`) for comparison with published Aariz results.
> Section 9 of the dataset analysis should show every device near the same Sella–Nasion length.

Aariz ground truth: the official junior/senior mean, except in train/val where the two differ by more
than 10 mm (Aariz seniors corrected the juniors, so the senior mark is used). Aariz agents start from the
mean shape of their own imaging device (`env.init: mean_device`), because fields of view differ a lot.

All images (both datasets) are converted to 8-bit grayscale (16-bit files are rescaled, not clipped)
and contrast-stretched so the 1st–99th percentile of non-black pixels spans 0–255.

Landmarks shared by both datasets are matched by name in `cephmarl/landmarks.py`
(all 19 ISBI landmarks have an Aariz counterpart). Check definitions such as
ISBI "Upper Lip" vs Aariz "Labrale superius" before reporting cross-dataset results.

## Analyse the datasets (do this before training)

One command prepares both datasets, runs every analysis and packs the results into `eda_results.zip`:
```bash
bash scripts/run_eda.sh /path/to/ISBI2015 /path/to/Aariz
```
Landmark-only analysis of ISBI from the annotation folder alone (no images needed):
```bash
python scripts/analyze_dataset.py --isbi-annotations /path/to/AnnotationsByMD --out analysis/isbi_ann
```
Individual runs:

```bash
python scripts/analyze_dataset.py --data-root data/isbi  --out analysis/isbi
python scripts/analyze_dataset.py --data-root data/aariz --out analysis/aariz
python scripts/analyze_dataset.py --data-root data/isbi  --compare data/aariz --out analysis/isbi_to_aariz
```
Each run writes `report.md`, `summary.json` and `figures/`: inventory, intensity per split/device,
landmark spread, junior–senior agreement and bias, Procrustes shape PCA and outlier skulls,
mean-shape start error (the baseline every method must beat) and step-budget reach, augmentation
fallback rate, train/test shift (KS tests), Sella–Nasion scale check per device, and cross-dataset
start error.

## Train, evaluate, flag, compare

```bash
python scripts/train.py --config configs/isbi_mappo.yaml --seed 0          # -> runs/isbi_mappo_mappo_s0/best.pt
python scripts/evaluate.py --checkpoint runs/isbi_mappo_mappo_s0/best.pt --split test1
python scripts/evaluate.py --checkpoint runs/isbi_mappo_mappo_s0/best.pt --data-root data/aariz --split test  # cross-dataset
python scripts/evaluate.py --checkpoint runs/isbi_mappo_mappo_s0/best.pt --split val   # needed for flagging
python scripts/flag.py --val runs/isbi_mappo_mappo_s0/eval_isbi_val.npz --test runs/isbi_mappo_mappo_s0/eval_isbi_test1.npz
python scripts/compare.py --split isbi_test1 --method MAPPO "runs/isbi_mappo_mappo_s*" --method DQN "runs/isbi_dqn_dqn_s*"
```
`bash scripts/run_experiments.sh` runs the whole plan (≈30 training runs).
Any config key can be overridden: `--set agent_steps=1000000 env.reward=team`.

| Config | What it is |
| --- | --- |
| `isbi_{dqn,mappo,coma,unet}.yaml`, `aariz_*.yaml` | main methods (COMA uses the team reward) |
| `ablation_noshare.yaml` | agents don't see each other's positions |
| `ablation_randinit.yaml` | random start instead of mean shape |
| `ablation_singlescale.yaml` | finest level only, same 115-step budget |
| `ablation_team_reward.yaml` | MAPPO on the team reward (fairness check vs COMA) |
| `synthetic_smoke.yaml` | tiny CPU run for checking the pipeline |

Ablations are built on MAPPO; if another RL method wins on ISBI, change `base:` in those files.

## Outputs

- `runs/<run>/best.pt` – best-on-validation checkpoint (also stores mean shape and shape model)
- `runs/<run>/log.jsonl` – validation MRE over training
- `runs/<run>/eval_<dataset>_<split>.json` – MRE ± SD, SDR at 2/2.5/3/4 mm, per-landmark MRE,
  and the junior-vs-senior human reference
- `runs/<run>/eval_<dataset>_<split>.npz` – predictions, errors, uncertainty features
- `..._flags_<signals>.json` – AUROC/AUPRC for errors > 2 and > 4 mm, catch rate at a 10 % review
  budget, risk–coverage curve, correlation with doctor disagreement
- `results/compare_<split>.md` – table over seeds + Wilcoxon signed-rank with Holm correction

## Code map

```
cephmarl/
  landmarks.py      landmark names, ISBI <-> Aariz mapping
  data.py           dataset preparation (ISBI, Aariz, synthetic) and the cached dataset
  augment.py        GPU affine + intensity augmentation (image and landmarks together)
  env.py            vectorised B images x K agents environment, 4-level pyramid, zoom rule, reward
  models.py         shared trunk, dueling Q head, actor, MAPPO critic, COMA critic, U-Net
  algos/dqn.py      Double + dueling DQN, shared weights, replay
  algos/mappo.py    MAPPO with centralised critic and GAE
  algos/coma.py     COMA with counterfactual baseline and TD(lambda) critic
  unet_baseline.py  heatmap U-Net training and decoding
  flagging.py       shape model, flag scorer, flag metrics
  metrics.py        radial error, MRE, SDR
scripts/            prepare_data, analyze_dataset, run_eda, diagnose_aariz, show_images, train, evaluate, flag, compare, run_experiments, smoke_test
tests/              oracle-agent, reward, patch, augmentation, shape-model and metric checks
```

## Environment in one table

| Level | mm/px | Patch covers | Step | Step cap |
| --- | --- | --- | --- | --- |
| L0 | 0.8 | 51.2 mm | 1.6 mm | 60 |
| L1 | 0.4 | 25.6 mm | 0.8 mm | 40 |
| L2 | 0.2 | 12.8 mm | 0.2 mm | 20 |
| L3 | 0.1 | 6.4 mm | 0.1 mm | 15 |

Reward = (previous distance − new distance) / step size, so it lies in [−1, 1] at every level;
leaving the image costs −1. An agent zooms in when it revisits one of its last 4 positions or hits
the level cap, and stops the same way at L3. Its answer is the mean of its last 4 L3 positions.

## Compute notes

- Budget is `agent_steps` (alive agent-steps, default 3 M). On a single T4/P100, measure the
  first run's speed from `log.jsonl` and scale `agent_steps` or seeds to fit your time.
- DQN replay is stored on CPU as uint8 patches (~8.5 KB per transition); `dqn.buffer` sets its size.
- Aariz images are loaded from disk per batch, so RAM use stays small.

## Implementation choices that differ from a literal reading of the design

- GroupNorm instead of BatchNorm in the patch encoder: BatchNorm's train/eval statistics interact
  badly with acting, replay and target networks in RL.
- The U-Net input is resized to *fit* 800×640 keeping the aspect ratio (≈0.3 mm/px on ISBI), so
  no landmark is cropped on Aariz images with a larger field of view.
- If validation errors all fall on one side of the flag threshold (an untrained model), the flag
  scorer falls back to an unweighted sum of z-scored signals and prints a warning.
