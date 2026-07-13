"""
Data preparation pipeline for MIMIC-IV sepsis diagnosis extraction.
Replaces the data-fetching portion of mimic_sepsis_diag.py.
"""

import pandas as pd
from pathlib import Path

from config import (
    BASE_DIR, OUTPUT_DIR,
    MIMIC_SEPSIS_CSV,
    MIMIC_DIAGNOSES_ICD,
    MIMIC_D_ICD_DIAGNOSES,
    MIMIC_ICUSTAYS,
    MIMIC_ADMISSIONS,
    CLEANED_WITH_TRANSLATION,
    PT_DIAG_CLEAN,
)

# ICU units to keep
VALID_ICU_UNITS = [
    "Medical Intensive Care Unit (MICU)",
    "Medicine",
    "Medicine/Cardiology Intermediate",
]


def extract_diagnoses(sepsis_csv: str = None, output_dir: str = None):
    """
    # Extract sepsis patient diagnoses from MIMIC-IV, filter by ICU unit,
    # merge with translated diagnosis names, and produce:
    #   - mimi3_diag.csv (hadm_id + long_title + hospital_expire_flag)
    #   - pt_diag.csv (raw patient × diagnosis pairs)
    """
    if sepsis_csv is None:
        sepsis_csv = MIMIC_SEPSIS_CSV
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR)

    # Load sepsis patient IDs
    data = pd.read_csv(sepsis_csv)
    sepsis_ids = data["hadm_id"].drop_duplicates()
    print(f"Loaded {len(sepsis_ids)} sepsis patient admissions")

    # Load and filter diagnoses
    d_icd = pd.read_csv(MIMIC_D_ICD_DIAGNOSES)
    diagnoses_icd = pd.read_csv(MIMIC_DIAGNOSES_ICD)
    diagnoses_icd = diagnoses_icd[diagnoses_icd["hadm_id"].isin(sepsis_ids)]

    merged = pd.merge(diagnoses_icd, d_icd, on="icd_code", how="inner")

    # Filter by ICU unit
    icustays = pd.read_csv(MIMIC_ICUSTAYS)
    icustays = icustays[icustays["hadm_id"].isin(sepsis_ids)]
    icustays = icustays[icustays["first_careunit"].isin(VALID_ICU_UNITS)]

    masked_ids = icustays["hadm_id"].drop_duplicates()
    filtered = merged[merged["hadm_id"].isin(masked_ids)]
    data_s = merged[~merged["hadm_id"].isin(masked_ids)]

    # Load cleaned translation whitelist
    sicu_diag = pd.read_csv(CLEANED_WITH_TRANSLATION)
    sicu_diag_ids = sicu_diag.iloc[:, 1].drop_duplicates()

    # Keep only translations that appear in our dataset
    in_whitelist = filtered["long_title"].isin(sicu_diag_ids)
    s_data = filtered[in_whitelist]
    s_ids = s_data["hadm_id"].drop_duplicates()
    all_ids = filtered["hadm_id"].drop_duplicates()
    m_ids = all_ids[~all_ids.isin(s_ids)]

    # Use non-whitelist diagnoses (filtered through ICU)
    final = filtered[filtered["hadm_id"].isin(m_ids)].copy()
    data_t = filtered[~filtered["hadm_id"].isin(m_ids)].copy()

    # Merge admissions for ALL intermediate datasets
    admissions = pd.read_csv(MIMIC_ADMISSIONS)[["hadm_id", "hospital_expire_flag"]]

    # --- Helper: clean + merge admissions for a given dataframe ---
    def _clean_and_merge(df, name):
        """Apply name cleaning, group by patient, merge admissions, save CSV."""
        df = df.copy()
        df["long_title"] = df["long_title"].astype(str).str.replace(" ", "_")
        df["long_title"] = df["long_title"].str.replace(",", "", regex=False)
        # Group by patient
        grouped = df.groupby("hadm_id")["long_title"].apply(list).reset_index()
        grouped["long_title"] = grouped["long_title"].str.join(" ")
        # Merge admissions
        merged = pd.merge(grouped, admissions, on="hadm_id", how="inner")
        # Save
        out_path = output_dir / f"{name}_diag.csv"
        merged.to_csv(out_path, index=False)
        print(f"Saved {out_path.name} ({len(merged)} patients, {len(df)} pairs)")
        return merged

    # --- Clean and save final ---
    final_clean = _clean_and_merge(final, "mimi3")

    # --- Clean and save intermediate datasets ---
    data_s_merged = _clean_and_merge(data_s, "data_s")
    data_t_merged = _clean_and_merge(data_t, "data_t")

    # Save raw patient-diagnosis pairs (cleaned, no admissions merge)
    pt_diag_path = output_dir / "pt_diag.csv"
    final.to_csv(pt_diag_path, index=False)
    print(f"Saved pt_diag.csv ({len(final)} patient-diagnosis pairs)")

    print(f"\nTotal: mimi3={len(final_clean)}, data_s={len(data_s_merged)}, "
          f"data_t={len(data_t_merged)}")

    return final_clean, data_s_merged, data_t_merged


if __name__ == "__main__":
    extract_diagnoses()
