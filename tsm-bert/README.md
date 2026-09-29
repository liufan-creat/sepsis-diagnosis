# Sepsis TSM-BERT — In-Hospital Mortality from Pre-Admission Vitals/Labs Trajectories

A compact BERT-style Transformer (~0.3M params) that predicts **in-hospital
death of the current admission** (label = `hospital_expire_flag`, MIMIC
scheme) from a patient's **pre-admission ICU trajectory** of 15 vitals/lab
channels, over three input observation windows: **w7 / w14 / w30 days**.

Three released checkpoints (one per window) + the full reproducible
training/evaluation code + the aggregate scaling statistics (`scalers.json`)
needed to score any conforming external cohort.

> **Terminology (important).** The `w7/w14/w30` names are the *input
> observation windows* (how many pre-admission days the model reads). They
> are **not** outcome horizons: the label is in-hospital death of the
> admission, independent of window.

## Checkpoints

| file | window | seq length (L=w*C) | AUC (source inner val, seed 42) | Brier val |
|------|-------:|-------------------:|--------------------------------:|----------:|
| `checkpoints/bert_trm_w7.pt`  | 7  | 105 | 0.8849 | — |
| `checkpoints/bert_trm_w14.pt` | 14 | 210 | 0.9388 | — |
| `checkpoints/bert_trm_w30.pt` | 30 | 450 | 0.9501 | — |

Each checkpoint is self-describing (`config` + `state_dict` + train metrics);
load with `model.model.build_model(ckpt["config"])` (see `infer.py`).
All three share one backbone geometry: d_model=64, 4 heads, 6 layers,
ffn=256, dropout=0.3, CLS readout, RoPE (day-major index), miss_mode=embed,
value_mode=linear, pool=cls, trained seed 42.

## Architecture & token design (one paragraph)

Day-major flattened cells, index = day*C + channel. Token vocabulary:
`0=PAD` (post-censorship or out-of-source-range; excluded from attention,
zero injection), `1=OBS` (observed in-range; token = `W_v[ch]*val +
E_day[day] + E_ch[ch]`, val = robust-standardized), `2=MISS` (missing at/
before that admission's right-censorship boundary — the last day with ANY
observation; the slot becomes `e_miss[day] + e_ch[ch]` and **stays in
attention** — missingness is structural "censoring-as-signal" input, never
imputed). Values are standardized with per-channel median/IQR fitted on the
source TRAIN split only; observed values outside the source per-channel range
become PAD (cross-cohort zero-leakage discipline). Binary logit head
(`BCEWithLogitsLoss`, pos_weight auto from the train ratio; sigmoid at
inference — softmax appears only inside attention).

## Repository layout

```
model/model.py            <-- THE model definition (single source of truth)
train.py                  train w7/w14/w30 from a source tokens npz
eval.py                   cross-dataset evaluation of released checkpoints
infer.py                  minimal inference (tokens npz -> per-hadm probabilities)
prepare_tsm.py            trajectories+mask h5 -> tsm_split.npz (split + robust scaling)
prepare_tsm_bert.py       tsm_split.npz -> tokenized npz (PAD/OBS/MISS)
make_bounds_npz.py        scalars.json -> minimal source bounds npz (for prepare_tsm_bert.py)
checkpoints/              bert_trm_w{7,14,30}.pt
scalars.json              aggregate source statistics (median/iqr/tok_edges)
```

## Requirements

Python >=3.10, CUDA optional (CPU works for source-scale eval; GPU strongly
recommended for larger cohorts). See `requirements.txt`.

## Reproducibility

Two verifications are documented for this release (script `verify_release.py`
in the original project logs; results below were produced at packaging time):

1. **Deterministic data pipeline** — rerunning `prepare_tsm.py` +
   `prepare_tsm_bert.py` from the raw source `trajectory.h5`
   (`--drop_channels crp fibrinogen`) reproduces `tsm_split.npz` and
   `tsm_bert_tokens.npz` **bit-for-bit** (all arrays element-equal).
2. **Weights x code parity** — loading the released checkpoints with the
   released `model/model.py` (strict state dict) reproduces the recorded
   source validation AUCs on the inner val split:
   w7 0.8849 / w14 0.9388 / w30 0.9501 (|diff| < 1e-5, CPU).
3. **End-to-end released scripts** — `eval.py` and `infer.py` on the source
   tokenized cohort (full cohort, in-sample): AUC w7 0.9471 / w14 0.9689 /
   w30 0.9774, bit-identical between the two scripts.

Re-training a from-scratch copy is NOT bit-exact across torch
versions/CUDA (expected; float non-determinism), and the released weights
are the reference artifacts.

### Training (from a source tokens npz)

```
python train.py --npz <tsm_bert_tokens.npz> --out_dir <out> --windows 7 14 30 --miss_mode embed
```

### Building tokens for an EXTERNAL cohort (no patient data leaves your disk
beyond what you already hold; only `scalars.json` is required from this repo)

```
# 1) robust-standardize with SOURCE stats (never fit on the target)
python prepare_tsm.py --data_dir <cohort_dir> --scalars_json scalers.json \
      --drop_channels crp fibrinogen --out_dir <out>/data
# 2) tokenize with SOURCE token edges + the embed MISS mode the checkpoints expect
python prepare_tsm_bert.py --npz <out>/data/tsm_split.npz \
      --bounds_npz <source_tokens.npz> --out_npz <out>/data/tsm_bert_tokens.npz \
      --miss_mode embed
# 3) score
python eval.py --npz <out>/data/tsm_bert_tokens.npz --model_dir checkpoints \
      --out_dir <out>/eval --device cuda
```

Note: for step 2 a source tokens npz is needed only for its `tok_edges` —
`scalars.json` in this repo carries the same edges, so a small helper can
construct the bounds npz if the original file is unavailable.
The cohort input contract is `trajectory.h5` (keys: trajectories,
observation_mask, hadm_ids, channel_names — 17-channel MIMIC layout from
which `crp`/`fibrinogen` are dropped) plus `mimi3_diag.csv`
(`hadm_id, hospital_expire_flag`).

## Quick inference (any conforming cohort)

```python
import numpy as np, torch
from model.model import build_model

ckpt = torch.load("checkpoints/bert_trm_w30.pt", map_location="cpu", weights_only=False)
model = build_model(ckpt["config"], ckpt["state_dict"])  # strict-load, .eval() ready

d = np.load("cohort_tokens.npz", allow_pickle=False)
w, C = 30, 15
tok = torch.from_numpy(d[f"bert_tok_{w}"])
val = torch.from_numpy(d[f"bert_val_{w}"]).float()
km = torch.cat([torch.zeros(tok.shape[0], 1, dtype=torch.bool), tok == 0], 1)
with torch.no_grad():
    p = torch.sigmoid(model(
        tok, val,
        torch.from_numpy(d[f"bert_daypos_{w}"]).long(),
        torch.from_numpy(d[f"bert_chpos_{w}"]).long(),
        km))
# p = P(in-hospital death | w30 pre-admission trajectory)
```

## What is deliberately NOT included

Patient-level data (tokens npz, split npz, labels, hadm ids) is NOT
distributed with this repository: the released artifacts above reproduce
the model on any conforming external cohort using only `scalars.json`
(aggregate statistics, no patient info). MIMIC-derived training data
remains subject to Physionet terms of use.

## License

MIT (code + weights).
