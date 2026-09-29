"""
make_bounds_npz.py - build a minimal source BOUNDS npz from scalars.json.

prepare_tsm_bert.py --bounds_npz expects a source tokens npz, but it only
consumes THREE fields from it: channels, n_bins, tok_edges.  This helper
Rebuilds a minimal, data-free bounds npz from the aggregate
scalars.json shipped with this repo, so external-cohort tokenization is
possible WITHOUT any patient-level file.

Run (single line):
  python make_bounds_npz.py --scalars_json scalers.json --out_npz bounds.npz
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--scalars_json",
                default=os.path.join(HERE, "scalars.json"))
ap.add_argument("--out_npz", default=os.path.join(HERE, "bounds.npz"))
args = ap.parse_args()

s = json.load(open(args.scalars_json))
np.savez(args.out_npz,
         channels=np.array(s["channels"]),
         n_bins=np.int64(s["n_bins"]),
         tok_edges=np.asarray(s["tok_edges"], dtype=np.float64))
print(f"saved -> {args.out_npz}  "
      f"(channels={len(s['channels'])}  n_bins={s['n_bins']}  "
      f"edges={np.asarray(s['tok_edges']).shape})")
print("use with: python prepare_tsm_bert.py ... --bounds_npz "
      f"{args.out_npz} --miss_mode embed")
