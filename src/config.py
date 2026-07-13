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
PATIENT_SUMMARY_LABELED = f"{BASE_DIR}/patient_summary_labeled.csv"
MIMI3_DIAG = f"{OUTPUT_DIR}/mimi3_diag.csv"
PT_DIAG_CLEAN = f"{OUTPUT_DIR}/pt_diag.csv"

# Intermediate dataset outputs
DATA_S_DIAG = f"{OUTPUT_DIR}/data_s_diag.csv"
S_DATA_DIAG = f"{OUTPUT_DIR}/s_data_diag.csv"
DATA_T_DIAG = f"{OUTPUT_DIR}/data_t_diag.csv"

# Tokenizer (both live in OUTPUT_DIR)
TOKENIZER_FILE_DIAG = f"{OUTPUT_DIR}/sepsis_diagnoses_tokenizer.json"
TOKENIZER_FILE_LAB = f"{OUTPUT_DIR}/sepsis_diagnoses_lab_tokenizer.json"

# Model checkpoints (BERT-36L = encoder_num=6, 36 transformer layers)
CHECKPOINT_DIAG = f"{OUTPUT_DIR}/BERT-36L-CSL.pth"
CHECKPOINT_LAB = f"{OUTPUT_DIR}/BERT-36L-Recon.pth"

# ── Model Hyperparameters ────────────────────────────────────────────────
LABEL_DIM = 128          # embedding dimension for token embeddings
NUM_LAYERS = 6           # transformer encoder layers
HIDDEN_SIZE = 256        # linear layer hidden size
NUM_HEADS = 8
CLASS_NUM = 2
BATCH_SIZE = 64
MAX_LENGTH = 45
VOCAB_SIZE_DIAG = None   # computed at runtime
VOCAB_SIZE_LAB = None

# ── Training ─────────────────────────────────────────────────────────────
TRAIN_EPOCHS_DIAG = 5000
TRAIN_EPOCHS_LAB = 5000
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

# ── Organ Failure Mapping ────────────────────────────────────────────────
ORGAN_FAILURE_KEYWORDS = {
    "renal failure": [
        "Acute_kidney_failure_with_lesion_of_tubular_necrosis",
        "Acute_kidney_failure_with_tubular_necrosis",
        "Hypertensive_chronic_kidney_disease_unspecified_with_chronic_kidney_disease_stage_I_through_stage_IV_or_unspecified",
        "End_stage_renal_disease",
        "Hypertensive_chronic_kidney_disease_with_stage_5_chronic_kidney_disease_or_end_stage_renal_disease",
        "Acute_kidney_failure_with_tubular_necrosis",
        "Hypertensive_heart_and_chronic_kidney_disease_with_heart_failure_and_with_stage_5_chronic_kidney_disease_or_end_stage_renal_disease",
        "Chronic_kidney_disease_stage_5",
        "Anuria_and_oliguria",
        "Unspecified_kidney_failure",
        "Chronic_kidney_disease_Stage_V",
    ],
    "liver failure": [
        "Acute_and_subacute_necrosis_of_liver",
        "Acute_and_subacute_hepatic_failure_with_coma",
        "Acute_and_subacute_hepatic_failure_without_coma",
        "Alcoholic_hepatic_failure_without_coma",
        "Hepatic_failure_unspecified_without_coma",
        "Chronic_hepatic_failure_without_coma",
        "Acute_hepatitis_C_with_hepatic_coma",
    ],
    "Heart failure": [
        "Acute_on_chronic_combined_systolic_(congestive)_and_diastolic_(congestive)_heart_failure",
        "Acute_systolic_heart_failure",
        "Congestive_heart_failure_unspecified",
        "Acute_on_chronic_diastolic_(congestive)_heart_failure",
        "Hypertensive_heart_and_chronic_kidney_disease_with_heart_failure_and_with_stage_5_chronic_kidney_disease_or_end_stage_renal_disease",
        "Chronic_diastolic_heart_failure",
        "Acute_on_chronic_systolic_heart_failure",
        "Unspecified_diastolic_(congestive)_heart_failure",
        "Chronic_systolic_heart_failure",
        "Acute_diastolic_(congestive)_heart_failure",
        "Hypertensive_heart_disease_with_heart_failure",
        "Chronic_systolic_(congestive)_heart_failure",
        "Acute_on_chronic_systolic_(congestive)_heart_failure",
        "Chronic_diastolic_(congestive)_heart_failure",
        "Acute_on_chronic_diastolic_heart_failure",
        "Heart_failure_unspecified",
        "Diastolic_heart_failure_unspecified",
        "Biventricular_heart_failure",
        "Acute_systolic_(congestive)_heart_failure",
        "Unspecified_systolic_(congestive)_heart_failure",
        "Acute_diastolic_heart_failure",
        "Acute_on_chronic_combined_systolic_and_diastolic_heart_failure",
        "Acute_right_heart_failure",
        "Chronic_combined_systolic_and_diastolic_heart_failure",
        "Right_heart_failure_unspecified",
        "Systolic_heart_failure_unspecified",
        "End_stage_heart_failure",
        "Acute_on_chronic_right_heart_failure",
    ],
    "Respiratory failure": [
        "Acute_respiratory_failure",
        "Acute_respiratory_failure_with_hypoxia",
        "Acute_respiratory_failure_unspecified_whether_with_hypoxia_or_hypercapnia",
        "Acute_and_chronic_respiratory_failure_with_hypercapnia",
        "Acute_and_chronic_respiratory_failure_with_hypoxia",
        "Acute_respiratory_failure_with_hypercapnia",
        "Acute_respiratory_distress_syndrome",
        "Respiratory_failure_unspecified_unspecified_whether_with_hypoxia_or_hypercapnia",
        "Respiratory_failure_unspecified_with_hypoxia",
        "Acute_and_chronic_respiratory_failure",
        "Respiratory_failure_unspecified_with_hypercapnia",
        "Acute_respiratory_distress",
        "Acute_and_chronic_respiratory_failure_unspecified_whether_with_hypoxia_or_hypercapnia",
        "Chronic_respiratory_failure_unspecified_whether_with_hypoxia_or_hypercapnia",
        "Chronic_respiratory_failure_with_hypoxia",
        "Chronic_respiratory_failure_with_hypercapnia",
        "Chronic_respiratory_failure",
    ],
    "Neurologic dysfunction": [
        "Metabolic_encephalopathy",
        "Altered_mental_status",
        "Encephalopathy_unspecified",
        "Coronary_angioplasty_status",
        "Other_encephalopathy",
        "Toxic_encephalopathy",
        "Somnolence",
        "Cerebral_infarction_unspecified",
        "Epilepsy_unspecified_without_mention_of_intractable_epilepsy",
        "Other_alteration_of_consciousness",
        "Aphasia_following_cerebral_infarction",
        "Cerebral_embolism_with_cerebral_infarction",
        "Epilepsy_unspecified_not_intractable_without_status_epilepticus",
        "Other_toxic_encephalopathy",
        "Delirium_due_to_known_physiological_condition",
        "Central_pontine_myelinolysis",
        "Cerebral_edema",
        "Cerebral_palsy_unspecified",
        "Cerebral_artery_occlusion_unspecified_with_cerebral_infarction",
        "Epilepsy_unspecified_not_intractable_with_status_epilepticus",
        "Localization-related_(focal)_(partial)_epilepsy_and_epileptic_syndromes_with_complex_partial_seizures_without_mention_of_intractable_epilepsy",
        "Cerebral_infarction_due_to_unspecified_occlusion_or_stenosis_of_bilateral_middle_cerebral_arteries",
        "Localization-related_(focal)_(partial)_symptomatic_epilepsy_and_epileptic_syndromes_with_simple_partial_seizures_not_intractable_without_status_epilepticus",
        "Localization-related_(focal)_(partial)_epilepsy_and_epileptic_syndromes_with_complex_partial_seizures_with_intractable_epilepsy",
        "Other_generalized_epilepsy_and_epileptic_syndromes_not_intractable_without_status_epilepticus",
        "Cerebral_infarction_due_to_embolism_of_basilar_artery",
        "Lennox-Gastaut_syndrome_not_intractable_with_status_epilepticus",
        "Lennox-Gastaut_syndrome_intractable_without_status_epilepticus",
        "Iatrogenic_cerebrovascular_infarction_or_hemorrhage",
        "Cerebral_infarction_due_to_unspecified_occlusion_or_stenosis_of_right_posterior_cerebral_artery",
        "Generalized_nonconvulsive_epilepsy_without_mention_of_intractable_epilepsy",
        "Cerebral_infarction_due_to_embolism_of_left_middle_cerebral_artery",
        "Generalized_idiopathic_epilepsy_and_epileptic_syndromes_not_intractable_with_status_epilepticus",
        "Cerebral_infarction_due_to_embolism_of_other_cerebral_artery",
        "Cerebral_infarction_due_to_embolism_of_right_carotid_artery",
        "Other_epilepsy_not_intractable_with_status_epilepticus",
        "Other_seizures",
        "Generalized_convulsive_epilepsy_with_intractable_epilepsy",
        "Other_forms_of_epilepsy_and_recurrent_seizures_without_mention_of_intractable_epilepsy",
        "Other_paralytic_syndrome_following_cerebral_infarction_bilateral",
        "Epilepsy_unspecified_with_intractable_epilepsy",
        "Cerebral_infarction_due_to_embolism_of_left_cerebellar_artery",
        "Other_generalized_epilepsy_and_epileptic_syndromes_not_intractable_with_status_epilepticus",
        "Guillain-Barre_syndrome",
        "Other_forms_of_epilepsy_and_recurrent_seizures_with_intractable_epilepsy",
        "Cerebral_infarction_due_to_embolism_of_right_cerebellar_artery",
        "Cerebral_infarction_due_to_embolism_of_bilateral_cerebellar_arteries",
    ],
    "Electrolyte dysfunction": [
        "Hyposmolality_and/or_hyponatremia",
        "Hypo-osmolality_and_hyponatremia",
        "Electrolyte_and_fluid_disorders_not_elsewhere_classified",
        "Hypokalemia",
        "Hyperkalemia",
        "Other_disorders_of_phosphorus_metabolism",
        "Hypocalcemia",
        "Hyperpotassemia",
        "Hyperosmolality_and/or_hypernatremia",
        "Hypertension_unspecified",
        "Hypercalcemia",
        "Hyperosmolality_and_hypernatremia",
        "Hypomagnesemia",
        "Other_disorders_of_electrolyte_and_fluid_balance_not_elsewhere_classified",
        "Disorders_of_phosphorus_metabolism",
        "Hypopotassemia",
        "Magnesium_deficiency",
        "Hypermagnesemia",
        "Disorders_of_magnesium_metabolism",
    ],
    "Hematologic dysfunction": [
        "Anemia_unspecified",
        "Unspecified_acquired_hypothyroidism",
        "Pancytopenia",
        "Thrombocytopenia_unspecified",
        "Other_secondary_thrombocytopenia",
        "Hereditary_spherocytosis",
        "Other_abnormal_blood_chemistry",
        "Essential_thrombocythemia",
        "Hemophagocytic_lymphohistiocytosis",
        "Leukocytosis_unspecified",
        "Other_pancytopenia",
        "Immune_thrombocytopenic_purpura",
        "Other_specified_diseases_of_blood_and_blood-forming_organs",
        "Elevated_white_blood_cell_count_unspecified",
        "Leukocytopenia unspecified",
    ],
    "Metabolic dysfunction": [
        "Acidosis",
        "Mixed_acid-base_balance_disorder",
        "Mixed_disorder_of_acid-base_balance",
        "Alkalosis",
    ],
}

# Organ failure → letter mapping
ORGAN_FAILURE_LETTERS = {
    "renal failure": "K",
    "liver failure": "L",
    "heart failure": "C",
    "respiratory failure": "R",
    "neurologic dysfunction": "N",
    "hematologic dysfunction": "H",
    "metabolic dysfunction": "M",
    "electrolyte dysfunction": "E",
}

# Infection detection keywords
INFECTION_SHOCK_KEYWORDS = [
    "Severe_sepsis_with_septic_shock",
    "Septic_shock",
]

# ── Special tokens ───────────────────────────────────────────────────────
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
