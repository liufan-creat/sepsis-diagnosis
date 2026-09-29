"""
Data preparation utility for MIMIC-IV sepsis diagnosis prediction.

run_pipeline(input_csv) → adds '[CLS] ' prefix, writes <stem>_cls.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, PT_DIAG


def run_pipeline(input_path: str = None, output_dir: str = None):
    """
    Add '[CLS] ' prefix to long_title → input column, save as <stem>_cls.csv.

    Parameters:
      input_path: mimi3_diag.csv or pt_diag.csv (any CSV with hadm_id + long_title)
      output_dir: output directory (defaults to OUTPUT_DIR)
    """
    if input_path is None:
        input_path = PT_DIAG
    if output_dir is None:
        output_dir = OUTPUT_DIR

    in_path = Path(input_path)
    data = pd.read_csv(in_path)
    data["input"] = "[CLS] " + data["long_title"].astype(str)

    out_path = Path(output_dir) / f"{in_path.stem}_cls.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Saved {out_path.name} ({len(data)} rows, columns: {list(data.columns)})")


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

    input_path = args.file or PT_DIAG
    if not Path(input_path).exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(str(input_path), args.output_dir)
