"""
Global configuration for sepsis diagnosis prediction pipeline.
All paths, hyperparameters, and model settings are centralized here.
"""

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = "/home/liufan/AI_biology/RNA/sepsis/mimic_iv_3.1"
OUTPUT_DIR = "/home/liufan/AI_biology/hermes_coding/sepsis_diag/outputs"

# MIMIC-IV raw data
MIMIC_DIAGNOSES_ICD = f"{BASE_DIR}/hosp/diagnoses_icd.csv.gz"
MIMIC_D_ICD_DIAGNOSES = f"{BASE_DIR}/hosp/d_icd_diagnoses.csv.gz"
MIMIC_ICUSTAYS = f"{BASE_DIR}/icu/icustays.csv.gz"
MIMIC_ADMISSIONS = f"{BASE_DIR}/hosp/admissions.csv.gz"
MIMIC_SEPSIS_CSV = f"{BASE_DIR}/sepsis.csv"

# Intermediate files
CLEANED_WITH_TRANSLATION = f"{BASE_DIR}/cleaned_with_translation.csv"
PT_DIAG = f"{BASE_DIR}/pt_diag.csv"

# Pipeline outputs (saved to OUTPUT_DIR)
PATIENT_SUMMARY = f"{BASE_DIR}/patient_summary.csv"
MIMI3_DIAG = f"{OUTPUT_DIR}/mimi3_diag.csv"
PT_DIAG_CLEAN = f"{OUTPUT_DIR}/pt_diag.csv"

# Intermediate dataset outputs
DATA_S_DIAG = f"{OUTPUT_DIR}/data_s_diag.csv"
S_DATA_DIAG = f"{OUTPUT_DIR}/s_data_diag.csv"
DATA_T_DIAG = f"{OUTPUT_DIR}/data_t_diag.csv"

# Tokenizer (lives in OUTPUT_DIR)
TOKENIZER_FILE_DIAG = f"{OUTPUT_DIR}/sepsis_diagnoses_tokenizer.json"

# Model checkpoints (BERT-36L = encoder_num=6, 36 transformer layers)
CHECKPOINT_DIAG = f"{OUTPUT_DIR}/BERT-36L-CSL.pth"

# ── Model Hyperparameters ────────────────────────────────────────────────
LABEL_DIM = 128          # embedding dimension for token embeddings
NUM_LAYERS = 6           # transformer encoder layers
HIDDEN_SIZE = 256        # linear layer hidden size
NUM_HEADS = 8
CLASS_NUM = 2
BATCH_SIZE = 64
MAX_LENGTH = 45
VOCAB_SIZE_DIAG = None   # computed at runtime

# ── Training ─────────────────────────────────────────────────────────────
TRAIN_EPOCHS_DIAG = 5000
TRAIN_RATIO = 0.75
LR = 1e-6
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
LOSS_WEIGHTS = [1.0, 2.0]  # class weights for [0, 1]
RANDOM_STATE = 42

# ── Evaluation ───────────────────────────────────────────────────────────
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 120
KMEANS_N_CLUSTERS = 8
KMEANS_RANDOM_STATE = 30


# ── Special tokens ───────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
