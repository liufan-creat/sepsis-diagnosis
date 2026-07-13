"""
eICU-specific label enrichment for classification_label column.

Reads sepsis_patients.csv diagnosisstring, extracts shock/sepsis status
and organ dysfunction indicators, then updates eicu_lab.csv's
classification_label and input columns accordingly.

Does NOT modify preparation.py. Standalone script.
"""

import pandas as pd
import re
from pathlib import Path

# ── eICU diagnosisstring → organ failure letter mapping ──
# These are natural-language patterns found in eICU diagnosisstrings,
# NOT MIMIC ICD codes. We match on the diagnosisstring directly.

ORGAN_KEYWORDS = {
    "R": [
        "respiratory failure",
        "acute respiratory failure",
        "ards",
        "acute respiratory distress",
        "respiratory distress",
        "ventilatory failure",
        "hypoxia",
        "hypercapnia",
    ],
    "K": [
        "acute renal failure",
        "acute kidney failure",
        "acute kidney injury",
        "acute renal insufficiency",
        "renal failure",
        "kidney failure",
        "renal insufficiency",
        "kidney insufficiency",
        "anuria",
        "oliguria",
    ],
    "C": [
        "congestive heart failure",
        "acute heart failure",
        "cardiac failure",
        "heart failure",
        "myocardial infarction",
        "cardiogenic shock",
        "circulatory system failure",
        "circulatory failure",
        "cardiac arrest",
    ],
    "L": [
        "hepatic failure",
        "acute hepatic failure",
        "liver failure",
        "liver insufficiency",
        "hepatic insufficiency",
        "fulminant hepatic",
        "jaundice",
    ],
    "N": [
        "metabolic encephalopathy",
        "encephalopathy",
        "altered mental status",
        "confusion",
        "coma",
        "seizure",
        "cerebrovascular accident",
        "stroke",
        "cerebral infarction",
        "intracranial hemorrhage",
    ],
    "H": [
        "dic syndrome",
        "disseminated intravascular coagulation",
        "thrombocytopenia",
        "platelet count low",
        "leukocytosis",
        "pancytopenia",
        "anemia",
        "bleeding disorder",
        "coagulopathy",
    ],
    "M": [
        "metabolic acidosis",
        "acidosis",
        "alkalosis",
        "electrolyte imbalance",
        "hyponatremia",
        "hypernatremia",
        "hypokalemia",
        "hyperkalemia",
        "hypocalcemia",
        "hypercalcemia",
        "hypomagnesemia",
        "severe",  # "sepsis|severe" in diagnosisstring → metabolic stress
    ],
}


def extract_organ_letters(diagnosisstring: str) -> str:
    """Extract organ failure letters from a single diagnosisstring."""
    if not isinstance(diagnosisstring, str):
        return ""

    ds_lower = diagnosisstring.lower()
    letters = []

    for letter, keywords in ORGAN_KEYWORDS.items():
        for kw in keywords:
            if kw in ds_lower:
                if letter not in letters:
                    letters.append(letter)
                break

    return "".join(sorted(set(letters)))


def determine_shock_status(diagnosisstring: str) -> str:
    """Determine if patient has shock or just sepsis."""
    if not isinstance(diagnosisstring, str):
        return "sepsis"

    ds_lower = diagnosisstring.lower()

    # Septic shock takes priority
    if "septic shock" in ds_lower:
        return "shock"

    # Also check for cardiovascular shock patterns
    if "shock" in ds_lower and ("cardiovascular" in ds_lower or "hypotension" in ds_lower):
        # Make sure it's not just "sepsis" without shock
        # "shock / hypotension|sepsis" → sepsis (not shock)
        # "shock / hypotension|septic shock" → shock
        if "septic shock" in ds_lower:
            return "shock"
        # If it says "sepsis" without "septic shock", it's sepsis
        if "sepsis" in ds_lower and "septic shock" not in ds_lower:
            return "sepsis"
        # "sirs" without sepsis/septic shock
        if "sirs" in ds_lower:
            return "sepsis"

    return "sepsis"


def enrich_eicu_labels(eicu_lab_path: str, sepsis_patients_path: str, output_path: str):
    """
    Load eicu_lab.csv and sepsis_patients.csv, enrich classification_label
    based on diagnosisstring, save updated file.

    Args:
        eicu_lab_path: path to outputs/eicu_lab.csv
        sepsis_patients_path: path to sepsis_patients.csv
        output_path: where to save the enriched CSV
    """
    # Load eicu_lab
    lab = pd.read_csv(eicu_lab_path)
    print(f"Loaded eicu_lab: {len(lab)} rows")

    # Load sepsis_patients
    sp = pd.read_csv(sepsis_patients_path)
    print(f"Loaded sepsis_patients: {len(sp)} rows")

    # Map patientunitstayid → diagnosisstring (key as string for matching)
    sp_map = sp.groupby("patientunitstayid")["diagnosisstring"].agg(lambda x: "|".join(x.unique())).to_dict()
    sp_map_str = {str(k): v for k, v in sp_map.items()}
    print(f"Unique patients in sepsis_patients: {len(sp_map_str)}")

    # Match hadm_id (eicu) with patientunitstayid (sp) — both as strings
    overlap = lab.hadm_id.astype(str).isin(sp_map_str.keys()).sum()
    print(f"Matching hadm_ids: {overlap} / {len(lab)} ({overlap/len(lab)*100:.1f}%)")

    # Enrich
    new_labels = []
    new_inputs = []
    unchanged = 0
    updated = 0

    for _, row in lab.iterrows():
        pid = row["hadm_id"]
        ds = sp_map_str.get(str(pid), "")

        if ds:
            shock_status = determine_shock_status(ds)
            organ_letters = extract_organ_letters(ds)
            new_label = f"{shock_status}_{organ_letters}"
            new_input = f"{new_label} {row['long_title']}"
        else:
            # No match in sepsis_patients, keep original
            new_label = row["classification_label"]
            new_input = row["input"]
            unchanged += 1
            new_inputs.append(new_input)
            new_labels.append(new_label)
            continue

        new_labels.append(new_label)
        new_inputs.append(new_input)
        if new_label != row["classification_label"]:
            updated += 1

    lab["classification_label"] = new_labels
    lab["input"] = new_inputs

    print(f"\nResults:")
    print(f"  Unchanged: {unchanged}")
    print(f"  Updated: {updated}")
    print(f"  Total: {len(lab)}")

    # Distribution
    print(f"\nNew label distribution (top 20):")
    print(lab.classification_label.value_counts().head(20))

    # Save
    lab.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")

    return lab


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich eICU classification labels from diagnosisstring")
    parser.add_argument("--eicu-lab", default="/home/liufan/AI_biology/hermes_coding/sepsis_diag/outputs/eicu_lab.csv")
    parser.add_argument("--sepsis-patients", default="/home/liufan/AI_biology/RNA/sepsis/eicu-2.0/sepsis_patients.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or args.eicu_lab.replace(".csv", "_enriched.csv")
    enrich_eicu_labels(args.eicu_lab, args.sepsis_patients, output)
