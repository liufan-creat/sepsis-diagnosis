"""
Data preparation pipeline for MIMIC-IV sepsis diagnosis prediction.

Single entry: run_pipeline(input_csv) → patient_summary_labeled.csv + lab_input.csv
"""

import argparse
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

from config import (
    BASE_DIR, OUTPUT_DIR,
    ORGAN_FAILURE_KEYWORDS,
    ORGAN_FAILURE_LETTERS,
    INFECTION_SHOCK_KEYWORDS,
    PT_DIAG,
)


def _normalize_diagnosis(diagnosis: str) -> str:
    return diagnosis.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')


def detect_organ_failure(diagnosis: str) -> str:
    if not isinstance(diagnosis, str):
        return "No organ failure"
    norm_diag = _normalize_diagnosis(diagnosis)
    for category, keywords in ORGAN_FAILURE_KEYWORDS.items():
        for kw in keywords:
            if kw in norm_diag or norm_diag in kw:
                return category
    return "No organ failure"


def get_infection_status(diagnosis: str) -> str:
    norm_diag = _normalize_diagnosis(diagnosis)
    for kw in INFECTION_SHOCK_KEYWORDS:
        if kw in norm_diag or norm_diag in kw:
            return "Sepsis Shock"
    return "Sepsis"


def _detect_input_format(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row_count = 0
        hadm_ids = set()
        sample_row = None
        for row in reader:
            row_count += 1
            hadm_ids.add(row["hadm_id"])
            if sample_row is None:
                sample_row = row

        if row_count == len(hadm_ids):
            return "mimi3"

        if sample_row:
            long_title = sample_row.get("long_title2", sample_row.get("long_title", ""))
            words = long_title.split()
            has_underscore_words = any("_" in w for w in words)
            if has_underscore_words and len(words) > 2:
                return "mimi3"

        return "pt_diag"


def _split_mimi3_long_title(long_title: str) -> list:
    words = long_title.split()
    merged = []
    buffer = []
    for w in words:
        if "_" in w:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append(w)
        else:
            buffer.append(w)
    if buffer:
        merged.append(" ".join(buffer))
    return merged


def _organ_failure_initials(organ_failure_categories: str) -> str:
    if not organ_failure_categories or organ_failure_categories == "None":
        return "_"

    initials = []
    for cat in organ_failure_categories.split(", "):
        cat = cat.strip().lower()
        if cat in ORGAN_FAILURE_LETTERS:
            initials.append(ORGAN_FAILURE_LETTERS[cat])

    seen = set()
    unique = []
    for c in initials:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return "".join(sorted(unique))


def run_pipeline(input_path: str = None, output_dir: str = None):
    """
    One command: read diagnosis CSV → patient_summary_labeled.csv + lab_input.csv.

    Parameters:
      input_path: mimi3_diag.csv or pt_diag.csv (any CSV with hadm_id + long_title)
      output_dir: output directory (defaults to OUTPUT_DIR)
    """
    if input_path is None:
        input_path = PT_DIAG
    if output_dir is None:
        output_dir = OUTPUT_DIR

    out_dir = Path(output_dir)

    # --- Step 1: detect format, aggregate per patient, label ---
    input_format = _detect_input_format(input_path)
    print(f"Input format: {input_format}")

    patient_diagnoses = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hadm_id = row["hadm_id"]
            raw_diag = row.get("long_title2", row.get("long_title", ""))

            if input_format == "mimi3":
                diags = _split_mimi3_long_title(raw_diag)
            else:
                diags = [raw_diag] if raw_diag else []

            patient_diagnoses[hadm_id].extend(diags)

    print(f"Patients: {len(patient_diagnoses)}")

    rows = []
    infection_stats = Counter()
    organ_stats = Counter()
    label_stats = Counter()

    for pid, diags in patient_diagnoses.items():
        infection_cats = set()
        organ_cats = set()
        for d in diags:
            inf = get_infection_status(d)
            infection_cats.add(inf)
            org = detect_organ_failure(d)
            if org != "No organ failure":
                organ_cats.add(org)

        part1 = "shock" if "Sepsis Shock" in infection_cats else "sepsis"
        part3 = _organ_failure_initials(
            ", ".join(sorted(organ_cats)) if organ_cats else ""
        )
        label = f"{part1}_{part3}"

        rows.append({
            "hadm_id": pid,
            "diagnosis_count": len(diags),
            "infection_status": ", ".join(sorted(infection_cats)),
            "organ_failure_categories": ", ".join(sorted(organ_cats)) if organ_cats else "None",
            "classification_label": label,
        })
        for ic in infection_cats:
            infection_stats[ic] += 1
        for oc in organ_cats:
            organ_stats[oc] += 1
        label_stats[label] += 1

    # Save patient_summary_labeled.csv
    labeled_path = out_dir / "patient_summary_labeled.csv"
    fieldnames = ["hadm_id", "diagnosis_count", "infection_status",
                  "organ_failure_categories", "classification_label"]
    with open(labeled_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {labeled_path.name} ({len(rows)} patients)")
    print(f"Label distribution: {dict(sorted(label_stats.items(), key=lambda x: -x[1]))}")

    # --- Step 2: merge back with raw data, create lab_input.csv ---
    summary = pd.read_csv(labeled_path)
    raw = pd.read_csv(input_path)

    lab = pd.merge(summary, raw, on="hadm_id", how="inner")
    lab["input"] = lab["classification_label"] + " " + lab["long_title"]

    # Keep only essential columns for training
    lab = lab[["hadm_id", "classification_label", "long_title", "hospital_expire_flag", "input"]]

    lab_output = out_dir / "lab_input.csv"
    lab.to_csv(lab_output, index=False)
    print(f"Saved {lab_output.name} ({len(lab)} rows, columns: {list(lab.columns)})")


def add_cls_prefix(input_path: str = None, output_path: str = None):
    """
    Add '[CLS] ' prefix to long_title → input column, overwrite in place.

    Parameters:
      input_path: CSV with hadm_id + long_title (e.g. mimi3_diag.csv)
      output_path: overwrite target (defaults to input_path)
    """
    if input_path is None:
        input_path = Path(OUTPUT_DIR) / "mimi3_diag.csv"
    if output_path is None:
        output_path = input_path

    data = pd.read_csv(input_path)
    data["input"] = "[CLS] " + data["long_title"].astype(str)
    data.to_csv(output_path, index=False)
    print(f"Saved {Path(output_path).name} ({len(data)} rows, columns: {list(data.columns)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sepsis data preparation pipeline")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to diagnosis CSV (mimi3_diag.csv or pt_diag.csv).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to OUTPUT_DIR.")
    args = parser.parse_args()

    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
    else:
        input_path = PT_DIAG

    run_pipeline(str(input_path), args.output_dir)

    # Default: add [CLS] prefix to input CSV, overwrite in place
    add_cls_prefix(str(input_path))
