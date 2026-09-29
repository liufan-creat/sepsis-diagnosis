# Sepsis Diagnosis Prediction

Two BERT-style Transformer families for in-hospital mortality prediction
in sepsis patients. See **`MODEL_GUIDE.md`** for complete usage of both.

| Family | Input | Checkpoints | Where |
|---|---|---|---|
| ICD text BERT (6L / 36L-CSL) | discharge ICD diagnosis text | 2 | [HuggingFace `fansen`](https://huggingface.co/fansen) (`BERT-6L-CSL-Sepsis-Diag`, `BERT-36L-CSL-Sepsis-Diag`) |
| TSM trajectory BERT (w7 / w14 / w30) | pre-admission daily vitals/lab trajectory, 15 channels | 3 | `tsm-bert/checkpoints/` |

## Repo layout

```
src/               ICD text BERT: config, tokenizer, model, train, unified eval,
                   embeddings + clustering analysis
hf_upload/         ICD model mirrors (tokenizers + checkpoints + per-model README)
tsm-bert/          TSM trajectory BERT (self-contained): prepare -> train -> eval -> infer
                   + weights w7/w14/w30 + scalars.json
MODEL_GUIDE.md     usage guide for both families (verified snippets)
```

## TSM trajectory BERT (w7 / w14 / w30)

Self-contained; no dependency on `src/`.

```bash
cd tsm-bert
pip install -r requirements.txt

# 1. scale + tokenise (aggregate scalars.json shipped; no patient data needed)
python prepare_tsm.py --h5 your_cohort/trajectory.h5 --scalars_json scalars.json \
    --bounds_npz ybounds.npz --miss_mode embed
python prepare_tsm_bert.py --scalars_json scalars.json --scale_npz ybounds.npz \
    --h5 your_cohort/trajectory.h5 --windows 7 14 30

# 2. eval / inference (weights included)
python eval.py  --window 14
python infer.py --window 14 --h5 your_cohort/trajectory.h5
```

Label: the single binary logit is sigmoided to P(death). Reproduction
numbers and the full external-cohort pipeline (incl. building your own
`trajectory.h5`) are documented in [`tsm-bert/README.md`](tsm-bert/README.md).

## ICD text BERT (6L / 36L-CSL)

```python
import torch
from src.bert_classifier import BertClassifier
from src.tokenizer_builder import load_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = load_tokenizer("hf_upload/BERT-36L-CSL-Sepsis-Diag/BERT-36L-CSL_tokenizer.json")

model = BertClassifier(
    vocab_size=len(tokenizer), label_dim=128, hidden_size=256,
    num_layers=6, num_heads=8, class_num=2, encoder_num=6, dropout=0.1,
).to(device)
state = torch.load("hf_upload/BERT-36L-CSL-Sepsis-Diag/BERT-36L-CSL.pt",
                   map_location=device, weights_only=False)["model_state_dict"]
model.load_state_dict(state)
model.eval()
# encoder_num: 1 for 6L / 6 for 36L
```

Training: `python src/run_experiment.py`; evaluation across external datasets:
`python src/unified_eval.py`.
