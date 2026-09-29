"""
infer.py - minimal inference over the released BERT-TSM checkpoints.

Input: a tokenized cohort npz (prepare_tsm_bert.py output; see README for
the two-step pipeline from trajectory.h5 + scalars.json).
Output: per-admission probability of in-hospital death, per window.

Run (single line):
  python infer.py --npz <cohort_tokens.npz> --model_dir checkpoints --windows 30
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # repo root, so `import model` resolves
from model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True, help="cohort TOKENS npz")
ap.add_argument("--model_dir", default=os.path.join(HERE, "checkpoints"))
ap.add_argument("--windows", nargs="*", type=int, default=None)
ap.add_argument("--out_csv", default=None,
                help="where to write hadm_id + p per window (default: <npz>_infer.csv")
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out_csv or
                                           os.path.join(HERE,
                                                        os.path.basename(args.npz) +
                                                        "_infer.csv"))), exist_ok=True)

out_csv = args.out_csv or os.path.join(
    os.path.dirname(os.path.abspath(args.npz)),
    os.path.basename(args.npz).replace(".npz", "") + "_infer.csv")
device = args.device if args.device != "cuda" \
    else ("cuda" if torch.cuda.is_available() else "cpu")

d = np.load(args.npz, allow_pickle=False)
C = len(list(d["channels"]))
y = d["y"].astype(np.int64) if "y" in d.files else None
N = len(y) if y is not None else d["bert_tok_7"].shape[0]
windows = [int(w) for w in d["windows"]]
if args.windows:
    windows = [w for w in windows if w in args.windows]
windows.sort()

# miss_mode gate: npz and checkpoint must agree (tok=2 routing differs)
npz_mm = str(d["miss_mode"]) if "miss_mode" in d.files else "pad"

rows = {}
for w in windows:
    ck = torch.load(os.path.join(args.model_dir, f"bert_trm_w{w}.pt"),
                    map_location="cpu", weights_only=False)
    if ck["config"].get("miss_mode", "pad") != npz_mm:
        raise SystemExit(f"checkpoint w{w} miss_mode={ck['config'].get('miss_mode')} "
                         f"!= tokens npz miss_mode={npz_mm}")
    if int(ck["config"]["c_n"]) != C:
        raise SystemExit(f"checkpoint w{w} c_n={ck['config']['c_n']} != npz channels {C}")
    model = build_model(ck["config"], ck["state_dict"], device=device)
    model.eval()
    ps = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, args.batch):
            sl = np.arange(s, min(s + args.batch, N))
            tok = torch.from_numpy(d[f"bert_tok_{w}"][sl]).long()
            val = torch.from_numpy(d[f"bert_val_{w}"][sl]).float()
            km = torch.cat([torch.zeros(tok.shape[0], 1, dtype=torch.bool),
                            tok == 0], dim=1)
            ps[s:s + len(sl)] = torch.sigmoid(
                model(tok.to(device), val.to(device),
                      torch.from_numpy(d[f"bert_daypos_{w}"][sl]).long().to(device),
                      torch.from_numpy(d[f"bert_chpos_{w}"][sl]).long().to(device),
                      km.to(device))).cpu().numpy()
    rows[w] = ps
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    msg = f"window={w}d  n={N}"
    if y is not None:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) >= 2:
            msg += f"  AUC={roc_auc_score(y, ps):.4f}"
    print(msg)

df = pd.DataFrame({f"w{w}": v for w, v in rows.items()})
if "hadm_ids" in d.files:
    df.insert(0, "hadm_id", d["hadm_ids"])
df.to_csv(out_csv, index=False)
print(f"saved -> {out_csv}")
