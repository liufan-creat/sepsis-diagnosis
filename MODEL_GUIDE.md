# Model Guide — Sepsis In-Hospital Mortality (two model families)

This repository ships **two model families** that predict the same outcome
— **in-hospital death of the current admission** (label
`hospital_expire_flag`, MIMIC scheme) — from two different input modalities.
Pick by what data you have at hand:

| family | input | models | where the weights live |
|---|---|---|---|
| **ICD text BERT** (`src/`) | discharge ICD diagnosis codes, tokenized text | 6L / 36L, CSL mode (2 checkpoints) | HuggingFace `fansen/BERT-6L-CSL-Sepsis-Diag`, `fansen/BERT-36L-CSL-Sepsis-Diag` |
| **TSM trajectory BERT** (`tsm-bert/`) | pre-admission vitals/lab daily trajectory, 15 channels | w7 / w14 / w30 (3 checkpoints) | `tsm-bert/checkpoints/` in this repo (+ HF `fansen/BERT-TSM-Sepsis-Diag` once uploaded) |

Both are BERT-style encoder Transformers; both output a single binary logit
(sigmoid -> probability of death). They are NOT comparable runs: different
inputs, different cohorts, different tokens. Use each on its own modality.

---

## 1. ICD text BERT (6L / 36L, CSL mode)

**Input**: one admission's ICD diagnosis text (see `unified_eval.py` for the
exact `long_title` column contract).
**Models**: 6L = 1 x 6-layer encoder branch; 36L = 6 x 6-layer branches
(`encoder_num=6`). CSL mode = diagnosis-text input.

### Download
```python
from huggingface_hub import snapshot_download
d = snapshot_download("fansen/BERT-36L-CSL-Sepsis-Diag")   # or fansen/BERT-6L-CSL-Sepsis-Diag
```

### Load + predict (verified snippet)
```python
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import models as tk_models
from src.bert_classifier import BertClassifier          # repo root on sys.path

repo = d                                                 # snapshot dir
# tokenizer filename matches the model name inside each HF repo:
#   6L-CSL -> BERT-6L-CSL_tokenizer.json, 36L-CSL -> BERT-36L-CSL_tokenizer.json, ...
tok = PreTrainedTokenizerFast(tokenizer_file=f"{repo}/BERT-36L-CSL_tokenizer.json")
tok.add_special_tokens({"pad_token": "[PAD]", "unk_token": "[UNK]",
                        "cls_token": "[CLS]", "sep_token": "[SEP]",
                        "mask_token": "[MASK]"})
tok._tokenizer.model = tk_models.WordPiece(vocab=tok.get_vocab(),
                                           unk_token="[UNK]")   # required fix

MAX_LENGTH = 45                                          # config.py
ckpt = torch.load(f"{repo}/pytorch_model.bin", map_location="cpu", weights_only=False)
model = BertClassifier(
    vocab_size=ckpt["model_state_dict"]["tok_embed.weight"].shape[0],
    label_dim=128, hidden_size=256, num_layers=6, num_heads=8,
    class_num=2, encoder_num=1 if "6L" in repo else 6,   # 6L -> 1 branch, 36L -> 6
    dropout=0.1).eval()
model.load_state_dict(ckpt["model_state_dict"])

inp = tok("Sepsis of unspecified origin ...", truncation=True,
          padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
logits = model(inp["input_ids"], (inp["input_ids"] == 0).bool())
prob = torch.softmax(logits, -1)[0, 1].item()            # P(in-hospital death)
```
Full cross-dataset evaluation harness: `unified_eval.py`
(`--model-type cs_l|all`; `python unified_eval.py --help`).

---

## 2. TSM trajectory BERT (w7 / w14 / w30)

**Input**: per-admission pre-admission daily series of 15 vitals/lab
channels (MAP, urine, lactate, pH, po2, ALT/AST/albumin, bilirubin,
creatinine, urea nitrogen, platelet, PT, PTT, WBC — the 17-channel MIMIC
layout minus `crp`/`fibrinogen`).
**Windows are INPUT observation horizons** — w7/w14/w30 = the model reads
the 7/14/30 days *before* admission. The label stays in-hospital death,
independent of window. Do not read w7/w14/w30 as follow-up periods.

Architecture: d64 / 4 heads / 6 layers, RoPE on the day-major token index,
CLS readout, ~0.3M params, binary logit head. Token vocabulary
`0=PAD, 1=OBS, 2=MISS`: missing cells before the admission's
right-censorship boundary become attended [MISS] tokens
("censoring as signal", `miss_mode=embed`); values are standardized with
per-channel median/IQR and OOV-clipped using **source-train-only**
statistics (`tsm-bert/scalars.json`). Cross-cohort use therefore needs ONLY
`scalars.json` — no patient-level file leaves this repository.

### Load + predict (single window, Python)
```python
import torch
from model.model import build_model            # tsm-bert/ on sys.path

ckpt = torch.load("tsm-bert/checkpoints/bert_trm_w30.pt",
                  map_location="cpu", weights_only=False)
model = build_model(ckpt["config"], ckpt["state_dict"])   # strict-load + .eval()

# tok/val/daypos/chpos: (B, w*15) tensors from a tokenized cohort npz
# (bert_tok_w / bert_val_w / bert_daypos_w / bert_chpos_w)
km = torch.cat([torch.zeros(tok.shape[0], 1, dtype=torch.bool), tok == 0], 1)
p = torch.sigmoid(model(tok, val, daypos.long(), chpos.long(), km))
# p = P(in-hospital death | w30 pre-admission trajectory)
```
Note: `daypos`/`chpos` must be **long** (npz stores int8; index ops fail
otherwise). All three windows share one backbone geometry; w7/w14 are
day-0-aligned prefixes of w30 (index = day*15 + channel).

### Scoring a NEW conforming cohort (shell, 4 commands)
```bash
cd tsm-bert
python prepare_tsm.py      --data_dir <cohort_dir> --scalars_json scalars.json \
      --drop_channels crp fibrinogen --out_dir data
python make_bounds_npz.py  --scalars_json scalars.json --out_npz data/bounds.npz
python prepare_tsm_bert.py --npz data/tsm_split.npz --bounds_npz data/bounds.npz \
      --out_npz data/tokens.npz --miss_mode embed
python infer.py            --npz data/tokens.npz --windows 7 14 30 --out_csv probs.csv
```
Cohort directory contract: `trajectory.h5` (keys: trajectories,
observation_mask, hadm_ids, channel_names) + `mimi3_diag.csv`
(hadm_id, hospital_expire_flag). `--miss_mode embed` is REQUIRED — the
checkpoints were trained in that mode and `infer.py` refuses mismatches.

### Reproducibility (verified at packaging, 2026-09-29)
- Prepare chain from raw h5 reproduces both npz **bit-for-bit**
  (`--drop_channels crp fibrinogen` + source scalars + `--miss_mode embed`).
- Released checkpoints recompute the recorded inner-val AUCs exactly:
  w7 0.8849 / w14 0.9388 / w30 0.9501.
- `eval.py` == `infer.py` on the source cohort (full cohort, in-sample):
  w7 0.9471 / w14 0.9689 / w30 0.9774.
- From-scratch retraining is not bit-exact across torch versions; the
  released weights are the reference artifacts.

---

## File map

```
src/            ICD text BERT: model def (bert_classifier.py), trainer,
                unified_eval.py, data prep, embeddings/SHAP scripts
hf_upload/      staging copies of the 4 ICD HF repo payloads
tsm-bert/       TSM trajectory BERT: model/model.py (single source),
                train.py, eval.py, infer.py, prepare_*.py, make_bounds_npz.py,
                checkpoints/bert_trm_w{7,14,30}.pt, scalars.json, README.md
requirements.txt  ICD-family deps; tsm-bert/requirements.txt = TSM deps
```

## Data & licenses

- Code + weights: MIT.
- **No patient-level data** is distributed with this repository;
  `scalars.json` holds aggregate statistics only. MIMIC-derived training
  data remains subject to PhysioNet terms of use.
