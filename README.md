# Sepsis Diagnosis Prediction from ICD Text

BERT-style Transformer classifier that predicts in-hospital mortality for sepsis patients based on their ICD diagnosis codes.

**Pretrained models available on [HuggingFace Hub](https://huggingface.co/fansen) (user: `fansen`)**:

| Model | Repo | Layers | Mode | Input Format |
|---|---|---|---|---|
| **BERT-6L-CSL** | [`fansen/BERT-6L-CSL-Sepsis-Diag`](https://huggingface.co/fansen/BERT-6L-CSL-Sepsis-Diag) | 6 (encoder_num=1×6) | diag | `[CLS] + long_title` |
| **BERT-36L-CSL** | [`fansen/BERT-36L-CSL-Sepsis-Diag`](https://huggingface.co/fansen/BERT-36L-CSL-Sepsis-Diag) | 36 (encoder_num=6×6) | diag | `[CLS] + long_title` |
| **BERT-6L-Recon** | [`fansen/BERT-6L-Recon-Sepsis-Diag`](https://huggingface.co/fansen/BERT-6L-Recon-Sepsis-Diag) | 6 (encoder_num=1×6) | dig_lab | `classification_label + long_title` |
| **BERT-36L-Recon** | [`fansen/BERT-36L-Recon-Sepsis-Diag`](https://huggingface.co/fansen/BERT-36L-Recon-Sepsis-Diag) | 36 (encoder_num=6×6) | dig_lab | `classification_label + long_title` |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/liufan-creat/sepsis-diagnosis.git
cd sepsis-diagnosis

# Install dependencies
pip install torch transformers scikit-learn shap pandas numpy matplotlib seaborn tokenizers
```

### Load a pretrained model

```python
import torch
from src.bert_classifier import BertClassifier
from src.tokenizer_builder import load_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and build model
tokenizer = load_tokenizer("hf_upload/BERT-36L-CSL-Sepsis-Diag/BERT-36L-CSL_tokenizer.json")
vocab_size = len(tokenizer)

model = BertClassifier(
    vocab_size=vocab_size,
    label_dim=128,
    hidden_size=256,
    num_layers=6,
    num_heads=8,
    class_num=2,
    encoder_num=6,  # 36 layers total = 6 branches × 6 layers each
    dropout=0.1,
).to(device)

# Load weights from checkpoint
ckpt = torch.load("path_to_model/pytorch_model.bin", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Project Structure

```
sepsis-diagnosis/
├── src/                           # Core source modules
│   ├── config.py                  # Global configuration (paths, hyperparameters)
│   ├── bert_classifier.py         # BERT-style Transformer model definition
│   ├── trainer.py                 # Training loop and evaluation functions
│   ├── run_experiment.py          # Main training entry point (diag / dig_lab modes)
│   ├── unified_eval.py            # Evaluate all 4 models on downstream datasets
│   ├── extract_embeddings.py      # Extract CLS embeddings for downstream clustering
│   ├── cluster_embeddings.py      # TSNE + KMeans clustering on embeddings
│   ├── compute_shap.py            # SHAP permutation importance analysis
│   ├── analyze_lab_clusters.py    # Lab-specific cluster analysis (dig_lab mode)
│   ├── eicu_add_labels.py         # eICU label enrichment for external validation
│   ├── extraction.py              # MIMIC-IV data extraction pipeline
│   ├── preparation.py             # Organ failure detection + classification labeling
│   ├── tokenizer_builder.py       # WordPiece tokenizer build / load utilities
│   └── notebook_script.py         # Jupyter-style interactive analysis example
├── scripts/
│   └── convert_to_hf.py           # Convert .pth checkpoints → HuggingFace format
├── hf_upload/                     # Pretrained models (HF-ready format)
│   ├── BERT-6L-CSL-Sepsis-Diag/
│   ├── BERT-36L-CSL-Sepsis-Diag/
│   ├── BERT-6L-Recon-Sepsis-Diag/
│   └── BERT-36L-Recon-Sepsis-Diag/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Data Preparation

### Prerequisites

This project uses **MIMIC-IV** clinical data, which requires credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/). Training on this data is only available to approved researchers.

```bash
# Place raw data in your project directory:
<your_path>/hosp/diagnoses_icd.csv.gz
<your_path>/hosp/admissions.csv.gz
<your_path>/icu/icustays.csv.gz

# Update BASE_DIR in src/config.py accordingly
```

### Pipeline steps

```bash
python src/extraction.py           # Step 1: extract sepsis patient diagnoses from MIMIC-IV
python src/preparation.py          # Step 2: organ failure detection + classification labels
python src/eicu_add_labels.py      # Step 3: (optional) enrich eICU labels for external validation
```

### Tokenizer

```bash
# Build tokenizer from training data (first time only):
python -c "
import pandas as pd
from src.tokenizer_builder import build_and_save_tokenizer
train_df = pd.read_csv('path/to/data_t_diag.csv')
words = train_df['long_title'].str.replace(' ', '_', regex=False).unique().tolist()
tokenizer = build_and_save_tokenizer(words, [], 'outputs/sepsis_diagnoses_tokenizer.json')
"
```

---

## Training from scratch

### diag mode (dignosis codes only)

```bash
python src/run_experiment.py \
    --encoder-num 6 \
    --train-steps 5000 \
    --n-clusters 8 \
    --checkpoint outputs/BERT-36L-CSL.pth
```

### dig_lab mode (diagnosis + classification labels)

```bash
python src/run_experiment.py \
    --mode dig_lab \
    --lab-input path/to/patient_summary_labeled.csv \
    --enable-lab-analysis \
    --encoder-num 6 \
    --train-steps 5000 \
    --n-clusters 6 \
    --checkpoint outputs/BERT-36L-Recon.pth
```

---

## Evaluation (all 4 models)

### Full evaluation across datasets

```bash
python src/unified_eval.py                      # All 4 models × all matching datasets
```

### Targeted evaluation

```bash
# Specific model type + dataset:
python src/unified_eval.py \
    --model-type cs_l \
    --checkpoint outputs/BERT-36L-CSL.pth \
    --input outputs/eicu_diag.csv

# Save per-sample predictions:
python src/unified_eval.py \
    --detail-output-dir my_preds/
```

---

## Embedding extraction + Clustering

### Extract CLS embeddings

```bash
# All 4 models × all datasets → pure embedding CSVs:
python src/extract_embeddings.py --output-dir outputs/embeddings/

# Single model + one dataset:
python src/extract_embeddings.py \
    --checkpoint outputs/BERT-36L-CSL.pth \
    --input outputs/eicu_diag.csv \
    -o my_embs/
```

### TSNE + KMeans clustering

```bash
python src/cluster_embeddings.py \
    --embeddings-dir <dir_with_emb*.csv> \
    --n-clusters 8
```

---

## SHAP Analysis

```bash
python src/compute_shap.py \
    -c outputs/BERT-36L-CSL.pth \
    -t outputs/sepsis_diagnoses_tokenizer.json \
    -x outputs/data_t_diag.csv \
    --n-samples 30 -o shap_results.csv
```

---

## Lab Cluster Analysis (dig_lab mode)

```bash
python src/analyze_lab_clusters.py \
    --clustered outputs/lab_clustered.csv \
    --lab outputs/patient_summary_labeled.csv \
    --checkpoint outputs/BERT-36L-Recon.pth \
    --tokenizer outputs/sepsis_diagnoses_lab_tokenizer.json \
    --n-clusters 6
```

---

## eICU Label Enrichment (external dataset)

```bash
python src/eicu_add_labels.py \
    -i <raw_eicu_csv> \
    -o eicu_label_enriched.csv
```

---

## License & Data Use

This repository is released under the **MIT License** (see [LICENSE](LICENSE)). It covers all source code and documentation.

**Data access**: Training data from MIMIC-IV requires credential approval through [PhysioNet](https://physionet.org/content/mimiciv/). The pretrained models are provided under the MIT License but their training involved MIMIC-IV derived features, which remain subject to PhysioNet's Data Use Agreement.

**eICU-CRD** data also has its own licensing requirements ([see here](https://eicu-crd.mit.edu/criticalcare/)). The eICU label enrichment script (`src/eicu_add_labels.py`) does not distribute or reproduce raw eICU-CRD records; it only restructures publicly available diagnosis strings.

### Citation

```bibtex
@misc{liufan2026sepsisdiagnosis,
  author       = {Liufan},
  title        = {Sepsis Diagnosis Prediction from ICD Text},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/liufan-creat/sepsis-diagnosis}}
}
```

---

## HuggingFace Model Repositories

| Model | HF Repo | Download |
|---|---|---|
| BERT-6L-CSL | [`fansen/BERT-6L-CSL-Sepsis-Diag`](https://huggingface.co/fansen/BERT-6L-CSL-Sepsis-Diag) | [Files](https://huggingface.co/fansen/BERT-6L-CSL-Sepsis-Diag/tree/main) |
| BERT-36L-CSL | [`fansen/BERT-36L-CSL-Sepsis-Diag`](https://huggingface.co/fansen/BERT-36L-CSL-Sepsis-Diag) | [Files](https://huggingface.co/fansen/BERT-36L-CSL-Sepsis-Diag/tree/main) |
| BERT-6L-Recon | [`fansen/BERT-6L-Recon-Sepsis-Diag`](https://huggingface.co/fansen/BERT-6L-Recon-Sepsis-Diag) | [Files](https://huggingface.co/fansen/BERT-6L-Recon-Sepsis-Diag/tree/main) |
| BERT-36L-Recon | [`fansen/BERT-36L-Recon-Sepsis-Diag`](https://huggingface.co/fansen/BERT-36L-Recon-Sepsis-Diag) | [Files](https://huggingface.co/fansen/BERT-36L-Recon-Sepsis-Diag/tree/main) |
