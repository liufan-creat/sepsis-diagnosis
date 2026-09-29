"""
eval.py - CROSS-DATASET verification of the released BERT-TSM checkpoints.

Loads checkpoints/bert_trm_w{w}.pt and scores a TARGET tokens npz:
sigmoid(model(...)) -> AUC / Brier / ROC per window.
Model definition: model/model.py (single source of truth).

REQUIRED pipeline for a target cohort:
  1. prepare_tsm.py      --data_dir <cohort_dir> --scalars_json scalers.json
       -> target tsm_split.npz (SOURCE median/iqr only, no leakage)
  2. prepare_tsm_bert.py --npz <target tsm_split.npz> \
       --bounds_npz <SOURCE tsm_bert_tokens.npz> --out_npz <target tokens.npz>
       (SOURCE tok_edges only, no leakage)
  3. THIS script on the resulting tokens npz.

Run (single line):
  python eval.py --npz <target_tokens.npz> --model_dir checkpoints --out_dir <out>

Arguments:
  --npz         target TOKENS npz (prepare_tsm_bert.py output)
  --model_dir   folder with bert_trm_w*.pt (default: checkpoints/)
  --out_dir     metrics csv + ROC pdf output (default: bert_eval/)
  --windows     subset (default: all present)
  --batch       inference batch size (default 512)
  --device      cuda | cpu
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True,
                help="target TOKEN npz from prepare_tsm_bert.py")
ap.add_argument("--model_dir", default=os.path.join(HERE, "bert_trm"),
                help="folder containing bert_trm_w*.pt")
ap.add_argument("--out_dir", default=os.path.join(HERE, "bert_eval"),
                help="output folder")
ap.add_argument("--windows", nargs="*", type=int, default=None,
                help="subset of windows (default: all present)")
ap.add_argument("--batch", type=int, default=512,
                help="inference batch size (default 512); w=30 sequences "
                     "(L=510) - drop to 256/128 if CUDA OOM")
ap.add_argument("--imputed", type=str, default=None,
                help="override the imputed_trend.npz path (default: the "
                     "path stored in the checkpoint's 'imputed' field, i.e. "
                     "the TARGET cohort's imputed npz for eval)")
ap.add_argument("--device", default="cuda",
                help="cuda or cpu (cpu = OOM-proof fallback, slower)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

device = args.device if args.device != "cuda" \
    else ("cuda" if torch.cuda.is_available() else "cpu")

d = np.load(args.npz, allow_pickle=False)
C = len(list(d["channels"]))
N_BINS = int(d["n_bins"])
y = d["y"].astype(np.int64)
N = len(y)
avail = [int(w) for w in d["windows"]]
if args.windows:
    avail = [w for w in avail if w in args.windows]
avail.sort()
L_max = d[f"bert_tok_{avail[0]}"].shape[1]
print(f"TARGET = {args.npz}\n")
print(f"  N={N}  pos={int(y.sum())}  rate={y.mean():.3f}  "
      f"C={C}  n_bins={N_BINS}  L_max={L_max}  windows={avail}")
if len(np.unique(y)) < 2:
    raise SystemExit(f"target has a single class (pos={int(y.sum())}); "
                     f"AUC undefined")

# imputed-trend sidecar resolution (imp_mode checkpoints only, checked per
# window below): explicit --imputed wins; otherwise the sidecar is
# co-located with the TARGET tokens npz (<npz_dir>/imputed_trend.npz) -
# eval always consumes the target cohort's own imputed labels, never the
# source path recorded in the checkpoint.
IMP_T = None
_imputed_path = args.imputed
have = [w for w in avail
        if os.path.isfile(os.path.join(args.model_dir, f"bert_trm_w{w}.pt"))]
if not have:
    raise SystemExit(f"no matching windows; npz has {avail}, "
                     f"models in {args.model_dir}")


from model.model import BertTSM, RoPEMHA  # single source of truth
def build_tensors(w, idxs):
    # int8 in the npz -> .long() for nn.Embedding indices
    # (tok stored 3-D (B,w,C) by older prepare runs -> flatten day-major)
    k = lambda n: d[f"bert_{n}_{w}"][idxs]
    ta = k("tok")
    tok = torch.from_numpy(ta).long().reshape(ta.shape[0], -1)
    val = torch.from_numpy(k("val")).reshape(ta.shape[0], -1).float()
    daypos = torch.from_numpy(k("daypos")).long()
    chpos = torch.from_numpy(k("chpos")).long()
    if "miss_mode" in d.files and str(d["miss_mode"]) == "trend":
        trend_t = torch.from_numpy(
            k("trend").astype(np.int64).reshape(val.shape))
    else:
        trend_t = None
    if "miss_mode" in d.files and str(d["miss_mode"]) == "last":
        v1_t = torch.from_numpy(k("v1")).reshape(val.shape).float()
        age_t = torch.from_numpy(
            k("age").astype(np.int64).reshape(val.shape))
    else:
        v1_t, age_t = None, None
    # imputed sidecar: ONE (N, W30*C) array per cohort from gen_imputed.py;
    # windows are day-0-aligned prefixes, so slice the first w*C columns
    # (same day-major flat order as the tokens: index = t*C + c).
    imp_lbl_t, imp_val_t = None, None
    if IMP_T is not None:
        L_win = int(w) * C
        assert L_win == val.shape[1]
        imp_lbl_t = torch.from_numpy(
            IMP_T["imputed_trend"][idxs][:, :L_win].astype(np.int64))
        imp_val_t = torch.from_numpy(
            (IMP_T["imputed_valid"][idxs][:, :L_win] > 0).astype(np.uint8))
    km = tok == 0
    km = torch.cat([torch.zeros(km.shape[0], 1, dtype=torch.bool), km],
                   dim=1)
    return tok, val, daypos, chpos, km, trend_t, v1_t, age_t, \
        imp_lbl_t, imp_val_t


def safe_auc(yy, pp):
    return roc_auc_score(yy, pp) if len(np.unique(yy)) >= 2 else float("nan")


results = []
roc_store = {}
for w in have:
    ck = torch.load(os.path.join(args.model_dir, f"bert_trm_w{w}.pt"),
                    map_location="cpu", weights_only=False)  # local ckpt, trusted
    cfg = ck["config"]
    # miss_mode: checkpoint config (absent in old ckpts = pad) must agree
    # with the target tokens npz, else the tok=2 routing / key mask /
    # trend labels would silently mismatch -> fail fast.
    ck_mm = cfg.get("miss_mode", "pad")
    npz_mm = str(d["miss_mode"]) if "miss_mode" in d.files else "pad"
    if ck_mm != npz_mm:
        raise SystemExit(f"checkpoint miss_mode={ck_mm} != tokens npz "
                         f"miss_mode={npz_mm}")
    if ck_mm == "trend" and f"bert_trend_{w}" not in d.files:
        raise SystemExit(f"checkpoint is miss_mode=trend but the tokens "
                         f"npz has no bert_trend_{w} labels")
    if ck_mm == "last" and (f"bert_v1_{w}" not in d.files
                            or f"bert_age_{w}" not in d.files):
        raise SystemExit(f"checkpoint is miss_mode=last but the tokens "
                         f"npz has no bert_v1_{w}/bert_age_{w} sidecars")
    # imp_mode: the checkpoint trained with the imputed sidecar; resolve
    # the TARGET cohort's imputed_trend.npz (explicit --imputed wins, else
    # co-located with --npz). Load once (imp_mode is window-invariant).
    ck_imp = cfg.get("imp_mode", False)
    if ck_imp and IMP_T is None:
        if _imputed_path:
            imp_path = _imputed_path
        else:
            imp_path = os.path.join(os.path.dirname(os.path.abspath(
                args.npz)), "imputed_trend.npz")
        if not os.path.isfile(imp_path):
            raise SystemExit(f"checkpoint is imp_mode but no "
                             f"imputed_trend.npz found at {imp_path}")
        _imp = np.load(imp_path, allow_pickle=False)
        if "imputed_trend" not in _imp.files or \
                "imputed_valid" not in _imp.files:
            raise SystemExit(f"{imp_path}: missing "
                             f"imputed_trend/imputed_valid")
        IMP_T = _imp
        _n_pass = int(_imp["imputed_valid"].sum())
        print(f"imputed (target) = {imp_path} "
              f"(gate-passed MISS cells = {_n_pass})")
    if ck_imp and (ck_mm != "embed"):
        raise SystemExit(f"imp_mode checkpoint requires miss_mode=embed "
                         f"(got {ck_mm})")
    model = BertTSM(cfg.get("d_model", 64), cfg["w_max"], cfg["c_n"],
                    heads=cfg.get("n_heads", 4),
                    layers=cfg.get("n_layers", 6),
                    ffn=cfg.get("ffn", 256),
                    drop=cfg.get("dropout", 0.3),
                    pool=cfg.get("pool", "cls"),
                    value_mlp=cfg.get("value_mlp", False),
                    value_mode=cfg.get("value_mode", "linear"),
                    miss_mode=ck_mm,
                    imp_mode=cfg.get("imp_mode", False)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    idx = np.arange(N)
    ps = np.empty(N, dtype=np.float32)
    for s in range(0, N, args.batch):
        sl = slice(s, min(s + args.batch, N))
        tok, val, daypos, chpos, km, trend_t, v1_t, age_t, \
            imp_lbl_t, imp_val_t = build_tensors(w, idx[sl])
        with torch.no_grad():
            ps[sl] = torch.sigmoid(model(tok.to(device), val.to(device),
                                         daypos.to(device), chpos.to(device),
                                         km.to(device),
                                         trend_t.to(device)
                                         if trend_t is not None else None,
                                         v1_t.to(device)
                                         if v1_t is not None else None,
                                         age_t.to(device)
                                         if age_t is not None else None,
                                         imp_lbl_t.to(device)
                                         if imp_lbl_t is not None
                                         else None,
                                         imp_val_t.to(device)
                                         if imp_val_t is not None
                                         else None
                                         )).cpu().numpy()

    auc = safe_auc(y, ps)
    bri = brier_score_loss(y, ps)
    fpr, tpr, _ = roc_curve(y, ps)
    roc_store[w] = (fpr, tpr)
    results.append({"window": w, "auc": round(auc, 4),
                    "brier": round(bri, 4), "n": N,
                    "pos_rate": round(float(y.mean()), 4)})
    print(f"  window={w}d  AUC={auc:.4f}  Brier={bri:.4f}  (n={N})")
    if device == "cuda":
        torch.cuda.empty_cache()

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
           "tab:brown"]
fig, ax = plt.subplots(figsize=(6.5, 5.5))
auc_by_w = {r["window"]: r["auc"] for r in results}
for k, w in enumerate(have):
    fpr, tpr = roc_store[w]
    ax.plot(fpr, tpr, lw=2.2, color=_COLORS[k % len(_COLORS)],
            label=f"w={w}d  AUC={auc_by_w[w]:.4f}")
ax.plot([0, 1], [0, 1], ":", color="k", lw=0.7)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title(f"BERT (linear value + RoPE) - cross-dataset (target: {os.path.basename(args.npz)})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(args.out_dir, "roc_target_all_windows.pdf"))
plt.close(fig)

pd.DataFrame(results).to_csv(os.path.join(args.out_dir, "eval_metrics.csv"),
                             index=False)
print(f"\nsaved -> {args.out_dir}/eval_metrics.csv, roc_target_all_windows.pdf")
