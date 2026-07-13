# Sepsis Diagnosis Prediction from ICD Text

BERT-style Transformer classifier that predicts in-hospital mortality for sepsis patients based on their ICD diagnosis codes.

| Model | Layers | Params | Mode | Input Format |
|---|---|---|---|---|
| **BERT-6L-CSL** | 6 (encoder_num=1×6) | ~4M | diag | `[CLS] + long_title` |
| **BERT-30L-CSL** | 36 (encoder_num=6×6) | ~22ML | diag | `[CLS] + long_title` |
| **BERT-6L-Recon** | 6 (encoder_num=1×6) | ~4M | dig_lab | `classification_label + long_title` |
| **BERT-30L-Recon** | 36 (encoder_num=6×6) | ~22ML | dig_lab | `classification_label + long_title` |

---

## Installation & Usage Guide

### Prerequisites
```bash
# Install conda env if you want
conda create -n sepsis python=3.10
conda activate sepsis
pip installed torch transformers scikit-learn shap pandas numpy matplotlib seaborn tokenizers
```

### Project Structure
```
sensitive-sequence/
├── src/                       # core source codes
│   ├── config.py              # all configurations, paths, hyperparameters, etc.
│   └── *.py                   # modules for data, training, evaluating
├── scripts/
│   └── convert_to_hf.py       # convert local checkpoints → HuggingFace format
├── notebooks/                 # interactive analysis examples
├── hf_upload/                 # (generated) HF-ready model directories
└── README.md                  # this file
```

### Data Preparation

1. Download MIMIC-IV data (requires credential at physiological.org):
```bash
# Place raw data in BASE_DIR (config.py):
BASE_DIR=/home/liufan/AI_biology/RNA/sepsis/mimic_iv_3.1/
  <your_path>/hosp/diagnoses_icd.csv.gz
  <your_path>/icu/icustays.csv.gz
```

2. Extract & prepare:
```bash
python src/extraction.py        # Step 1: extract sepsis patient data
python src/preparation.py       # Step 2: organ failure detection + classification labels
```

3. Build tokenizer (first time only):
```python
from src.tokenizer_builder import build_tokenizer
train_df = pd.read_csv("src/config.py's DATA_T_DIAG")
build_tokenizer(train_df, output_file, max_length=100)
```

### Training from scratch

```bash
# diag mode (5K steps, 8 clusters):
python src/run_experiment.py \
    --encoder-num 6 --train-steps 5000 --n-clusters 8 \
    --checkpoint outputs/BERT-36L-CSL.pth

# dig_lab mode (10K steps, 6 clusters):
python src/run_experiment.py --lab-input path/to/patient_summary_labeled.csv \
    --enable-lab-analysis \
    --encoder-num 6 --train-steps 10000 --n-clusters 6 \
    --checkpoint outputs/BERT-36L-Recon.pth
```

### Evaluation (all 4 models)

```bash
# Evaluate all 4 checkpoints on everything:
python src/unified_eval.py

# Specific model + dataset:
python src/unified_eval.py \
    --model-type cs_l \
    --checkpoint outputs/BERT-36L-CSL.pth \
    --input outputs/eicu_diag.csv

# Save detail predictions to dir:
python src/unified_eval.py --detail-output-dir my_preds/
```

### Extract Embeddings (downstream clustering)

```bash
# All 4 models × all datasets → pure embedding CSV:
python src/extract_embeddings.py --output-dir outputs/embeddings/

# Single model + one dataset:
python src/extract_embeddings.py \
    --checkpoint outputs/BERT-36L-CSL.pth \
    --input outputs/eicu_diag.csv \
    -o my_embs/
```

### Clustering (TSNE + KMeans)

```bashpython src/cluster_embeddings.py \
    --embeddings-dir <dir_with_emb*.csv> \
    --n-clusters 8
```

### SHAP Analysis

```bashpython src/compute_shap.py \
    -c outputs/BERT-36L-CSL.pth \
    -t outputs/sepsis_diagnoses_tokenizer.json \
    -x outputs/data_t_diag.csv \
    --n-samples 30 -o shap_results.csv
```

### Lab Cluster Analysis (dig_lab mode only)

```bashpython src/analyze_lab_clusters.py \
    --clustered outputs/lab_clustered.csv \
    --lab outputs/patient_summary_labeled.csv \
    --checkpoint outputs/BERT-36L-Recon.pth \
    --tokenizer outputs/sepsis_diagnoses_lab_tokenizer.json \
    --n-clusters 6
```

### eICU Label Enrichment (eICU-CRD external dataset)

```bashpython src/eicu_add_labels.py \
    -i <raw_eicu_csv> -o eicu_label_enriched.csv
```

---

## Model Weights (HuggingFace Hub)

Download pretrained models from HF:

| Repo | Checkpoint | Tokenizer |
|---|---|---|
| `liufan/BERT-36L-CSL-Sepsis-Diag` | BERT-36L-CSL.pth → [CLS] + long_title` | sepsis_diagnoses_tokenizer.json |
| `liufan/BERT-36L-Recon-Sepsis-Diag` | BERT-36L-Recon.pth | sepsis_diagnoses_lab_tokenizer.json |

```python
import torch
from src.bert_classifier import BertClassifier

# Example: load HF model
ckpt = torch.load("pytorch_model.bin")
model = BertClassifier(vocab_size=4805, encoder_num=6)
model.load_state_dict(ckpt["model_state_dict"])
```

### Convert checkpoints → HF format (local)

```bashcd /home/liufan/AI_biology/hermes_coding/sepsis_diag
python ../sepsis-diagnosis/scripts/convert_to_hf.py --convert-all
# or for a single model:
python ../sepsis-diagnosis/scripts/convert_to_hf.py -c outputs/BERT-36L-CSL.pth -t outputs/sepsis_diagnoses_tokenizer.json -n BERT-36L-CSL
```

Outputs are in `hf_upload/<model_name>/` with config + pytorch_model.bin.

---

## License & Data Use

MIMIC-IV requires credentialed access through PhysioNet. The code is released for research purposes only.

**Citation**: (pending)
