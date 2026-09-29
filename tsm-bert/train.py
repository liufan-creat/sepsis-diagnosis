"""
train.py - train the BERT-TSM window models (w7 / w14 / w30) from the source
TOKENS npz (prepare_tsm_bert.py output).  One checkpoint per window.

Model definition: model/model.py (single source of truth; this file no longer
embeds a copy).

Run (single line):
  python train.py --npz <tsm_bert_tokens.npz> --out_dir <out_folder> --windows 7 14 30

Arguments (unchanged interface from the original pipeline):
  --npz         source tokens npz (must carry a matching miss_mode field)
  --out_dir     where bert_trm_w{w}.pt + bert_trm_metrics.csv + ROC pdfs go
  --windows     subset of windows (default: all in the npz)
  --seed / --miss_mode / --value_mode / --pool / --pos_w / --smooth /
    --warmup / --sched / --train_log / --init_from / --imputed
                see argparse below. The released checkpoints were trained
                with the released defaults (--miss_mode embed,
                --value_mode linear, --pool cls, seed 42) on the SOURCE
                cohort.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default=os.path.join(HERE, "data",
                                              "tsm_bert_tokens.npz"))
ap.add_argument("--out_dir", default=os.path.join(HERE, "bert_trm"))
ap.add_argument("--windows", nargs="*", type=int, default=None)
ap.add_argument("--seed", type=int, default=42,
                help="train seed (init + batch shuffle); the inner 80/20 "
                     "ES split is always SPLIT_SEED=42, independent, so that "
                     "multi-seed runs compare training variance only")
ap.add_argument("--pool", choices=["cls", "dual"], default="cls",
                help="cls = single CLS readout (v3.7); dual = CLS + masked "
                     "mean of OBSERVED tokens (missing/PAD excluded) -> "
                     "wider head (default cls for backward compat)")
ap.add_argument("--value_mode", choices=["linear", "raw", "squash"],
                default="linear",
                help="linear = v3.7 (W_v[ch]*val on standardized values); "
                     "raw = NEW ARCH: token value part = W_v[ch]*clamp("
                     "raw_physiological*scale_ch, +-CLAMP_LIM=20) with "
                     "LEARNED per-channel scale (init sc_init=1/p99|raw| "
                     "from the token npz, constant init = no extra RNG "
                     "draw); missing cells stay PAD (need "
                     "prepare_tsm_bert.py --raw npz); "
                     "squash = TabPFN SquashingScaler recipe applied to the "
                     "STANDARDIZED val (linear npz): token value part = "
                     "W_v[ch]*softclip(val), softclip(z)=z/sqrt(1+(z/B)^2), "
                     "B=3 - bounded, spikes compressed, no new params")
ap.add_argument("--value_mlp", action="store_true",
                help="add a shared nonlinear value path (1->32->hidden, "
                     "GELU) on top of the per-channel linear W_v*val path; "
                     "gates to 0 on PAD cells (default off = v3.7 linear)")
ap.add_argument("--smooth", type=float, default=0.0,
                help="label smoothing in the BCE loss (default 0 = hard, "
                     "v3.7 behavior)")
ap.add_argument("--pos_w", type=str, default="auto",
                help="'auto' = inverse class ratio of the train split "
                     "(v3.7); or a fixed float e.g. '4'")
ap.add_argument("--warmup", type=int, default=0,
                help="linear LR warmup epochs (0 = off, v3.7)")
ap.add_argument("--sched", choices=["none", "cosine"], default="none",
                help="LR schedule after warmup (none = constant, v3.7)")
ap.add_argument("--train_log", action="store_true",
                help="dump per-epoch ES AUC/Brier to train_log.csv")
ap.add_argument("--init_from", type=str, default=None,
                help="pretraining checkpoint (pretrain_bert.py output): "
                     "load BACKBONE keys only (w_v/e_day/e_ch/e_miss/cls/"
                     "encoder.*); head + pt_head.* stay randomized.  The "
                     "checkpoint must match w_max/c_n/layer geometry and "
                     "value_mode; e_miss is loaded when this run uses "
                     "miss_mode=embed (skipped with an INFO note in pad "
                     "mode) and is NOT written into the saved config")
ap.add_argument("--miss_mode", choices=["pad", "embed", "trend", "last"], default="pad",
          help="pad = v3.7 (missing cells are PAD, excluded from "
               "attention); embed = tokens built with "
               "prepare_tsm_bert.py --miss_mode embed: tok 2 = "
               "[MISS] cell before the censorship boundary, "
               "consumed as e_miss[daypos] + e_ch[chpos] and kept "
               "in attention; 0 = PAD (post-censorship or OOV) "
               "excluded via the key mask as before. trend = embed "
               "PLUS a per-MISS-cell direction label from the npz's "
               "bert_trend_<w> (0=NA/1=down/2=flat/3=up), consumed as "
               "e_miss + e_ch + e_trend[label]. last = embed PLUS the "
               "npz sidecars bert_v1_<w> (channel's last known reading, "
               "a real past value) and bert_age_<w> (age bucket "
               "0=NA/1-3d/4-7d/8-14d/>14d), consumed as "
               "e_miss + e_ch + W_last[ch]*v1 + e_age[age]. each mode "
               "needs a tokens npz built with the matching "
               "--miss_mode; checkpoint + eval must agree (checked at "
               "load)")
ap.add_argument("--imputed", type=str, default=None,
                help="imputed_trend.npz (gen_imputed.py output): per-MISS-cell "
                     "direction label (0=down/1=flat/2=up) from the frozen "
                     "pretrain backbone. Injected as e_imp_trend[label] on "
                     "the gated-passed MISS cells (imputed_valid==1); "
                     "gate-blocked MISS cells keep the plain [MISS] token. "
                     "Requires --miss_mode embed. Must be the SOURCE cohort's "
                     "imputed npz for training, the target's for eval.")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 6
FFN = 256
DROPOUT = 0.3
LR = 1e-3
WD = 1e-4
MAX_EPOCH = 200
PATIENCE = 30
BATCH = 64
# ---- v4 knobs (from args; defaults reproduce v3.7 behavior when omitted)
POOL = args.pool                 # cls | dual (CLS + masked mean of observed tokens)
VALUE_MLP = args.value_mlp       # shared nonlinear value path on top of W_v*val
# v5 value mode: 'linear' (v3.7: W_v[ch]*val, val standardized) or 'raw'
# (token value path = W_v[ch] * tanh(clamp(raw*sc, +-CLAMP_LIM)), sc a
# LEARNED per-channel scale initialized to sc_init = 1/iqr from the token
# npz; raw values are the actual physiological measurements, no fixed
# normalization - the scale is the model's job)
# clamp bound in init-scale (iqr-normalized) units; 99.9% of physiological
# values sit inside; safety valve for AST-level spikes (raw mode only)
# TabPFN SquashingScaler max_absolute_value for the squash value mode:
# soft-clip bound z/sqrt(1+(z/B)^2) -> +/-B, B=3 (TabPFN default)
VALUE_MODE = args.value_mode
SC_INIT = None
if args.value_mode == "raw":
    _d0 = np.load(args.npz, allow_pickle=False)
    if "sc_init" not in _d0.files:
        raise SystemExit("--value_mode raw requires a token npz built with "
                         "prepare_tsm_bert.py --raw (sc_init missing)")
    SC_INIT = torch.tensor(_d0["sc_init"].astype(np.float32))
SMOOTH = args.smooth             # label smoothing in BCEWithLogitsLoss
POS_W = args.pos_w               # 'auto' | float
WARMUP = args.warmup             # linear warmup epochs
SCHED = args.sched               # none | cosine
TRAIN_LOG = args.train_log       # dump per-epoch ES AUC/Brier csv
SEED = args.seed            # run seed: init + batch shuffle order
SPLIT_SEED = 42             # FIXED: inner 80/20 ES split, seed-independent
MISS_MODE = args.miss_mode  # pad = v3.7 | embed = [MISS] token path
INIT_FROM = args.init_from  # pretrain_bert.py checkpoint (backbone only)
IMPUTED = args.imputed      # imputed_trend.npz (None = off)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

d = np.load(args.npz, allow_pickle=False)
print(f"input  = {args.npz}")
print(f"device = {device}")
print(f"seed   = {SEED}   (inner ES split frozen at SPLIT_SEED={SPLIT_SEED})")
# cross-check model/npz miss_mode (fail fast before any training spend)
_npz_mm = str(d["miss_mode"]) if "miss_mode" in d.files else "pad"
if _npz_mm != MISS_MODE:
    raise SystemExit(f"--miss_mode {MISS_MODE} != tokens npz miss_mode "
                     f"{_npz_mm}; rebuild the npz or pass the matching "
                     f"flag (embed/trend require the matching npz)")
# trend/last mode: the npz must carry the matching sidecar tensors
if MISS_MODE in ("trend", "last"):
    _want_w = max(int(w) for w in d["windows"])
    _need = ("bert_trend", "bert_v1")[MISS_MODE == "last"]
    if f"{_need}_{_want_w}" not in d.files:
        raise SystemExit(f"miss_mode={MISS_MODE} but the tokens npz has no "
                         f"{_need}_<w> tensors (rebuild with "
                         f"prepare_tsm_bert.py --miss_mode {MISS_MODE})")
    print(f"miss_mode={MISS_MODE}: consuming "
          + ("bert_trend_<w> direction labels" if MISS_MODE == "trend"
             else "bert_v1_<w> + bert_age_<w> last-known sidecars")
          + " on MISS cells")
# imputed-trend sidecar (gen_imputed.py): needs miss_mode=embed (the MISS
# cells must exist), else the e_imp_trend path has no slot to attach to.
IMP_T = None
if IMPUTED:
    if MISS_MODE != "embed":
        raise SystemExit("--imputed requires --miss_mode embed (the "
                         "direction labels attach to the [MISS] slots)")
    _imp = np.load(IMPUTED, allow_pickle=False)
    if "imputed_trend" not in _imp.files or \
            "imputed_valid" not in _imp.files:
        raise SystemExit(f"{IMPUTED}: missing imputed_trend/imputed_valid")
    IMP_T = _imp
    _n_pass = int(_imp["imputed_valid"].sum())
    _vc = np.bincount(_imp["imputed_trend"].ravel()[
        _imp["imputed_valid"].ravel() > 0], minlength=3)
    print(f"imputed = {IMPUTED}")
    print(f"  gate-passed MISS cells = {_n_pass} "
          f"({_imp['imputed_valid'].mean():.3f} of all cells) | "
          f"down={_vc[0]} flat={_vc[1]} up={_vc[2]}")
C = len(list(d["channels"]))
N_BINS = int(d["n_bins"])
y = d["y"]
tr_idx, va_idx = d["tr_idx"], d["va_idx"]
W_MAX = max(int(w) for w in d["windows"])
avail = [int(w) for w in d["windows"]]
if args.windows:
    avail = [w for w in avail if w in args.windows]
avail.sort()
N = len(y)
print(f"N={N}  train={len(tr_idx)}  val={len(va_idx)}  "
      f"C={C}  n_bins={N_BINS}  max_seq={W_MAX * C}")
print(f"windows={avail}  d_model={D_MODEL} layers={N_LAYERS} "
      f"heads={N_HEADS} lr={LR} wd={WD}\n")


from model.model import BertTSM, RoPEMHA  # single source of truth
def build_input_tensors(w, idxs):
    # int8 in the npz -> .long() for nn.Embedding indices
    # (tok stored 3-D (B,w,C) by older prepare runs -> flatten day-major)
    k = lambda n: d[f"bert_{n}_{w}"][idxs]
    ta = k("tok")
    tok = torch.from_numpy(ta).long().reshape(ta.shape[0], -1)
    val = torch.from_numpy(k("val")).reshape(ta.shape[0], -1).float()
    daypos = torch.from_numpy(k("daypos")).long()
    chpos = torch.from_numpy(k("chpos")).long()
    if MISS_MODE == "trend":
        trend_t = torch.from_numpy(
            k("trend").astype(np.int64).reshape(val.shape))
    else:
        trend_t = None
    if MISS_MODE == "last":
        v1_t = torch.from_numpy(k("v1")).reshape(val.shape).float()
        age_t = torch.from_numpy(
            k("age").astype(np.int64).reshape(val.shape))
    else:
        v1_t, age_t = None, None
    # imputed sidecar: gen_imputed.py stores ONE array per cohort at the
    # w30 (generation) window. Windows are day-0-aligned prefixes of w30,
    # so the (B, W30*C) array sliced to the first w*C columns is exactly
    # the window's own cells (gen_imputed uses the same day-major flat
    # order as the tokens: index = t*C + c).
    imp_lbl_t, imp_val_t = None, None
    if IMP_T is not None:
        L_full = IMP_T["imputed_trend"].shape[1]
        L_win = int(w) * C
        sl = IMP_T["imputed_trend"][idxs][:, :L_win]
        sv = IMP_T["imputed_valid"][idxs][:, :L_win]
        assert L_win <= L_full and L_win == val.shape[1]
        imp_lbl_t = torch.from_numpy(sl.astype(np.int64))
        imp_val_t = torch.from_numpy((sv > 0).astype(np.uint8))
    key_mask = tok == 0                       # (B, L), True=PAD only
                                         # (embed/trend/last: MISS=2 attends)
    key_mask = torch.cat([torch.zeros(key_mask.shape[0], 1,
                                       dtype=torch.bool), key_mask], dim=1)
    return tok, val, daypos, chpos, key_mask, trend_t, v1_t, age_t, \
        imp_lbl_t, imp_val_t, \
        torch.as_tensor(y[idxs].astype(np.int64))


def inner_split():
    """Identical inner 80/20 ES split to the other models
    (same seed + same per-window shuffle order).  Uses the FIXED
    SPLIT_SEED, so changing --seed never changes the ES split."""
    rng = np.random.default_rng(SPLIT_SEED)
    sets = {}
    for w in avail:
        tr_pos = [i for i in tr_idx if y[i] == 1]
        tr_neg = [i for i in tr_idx if y[i] == 0]
        rng.shuffle(tr_pos)
        rng.shuffle(tr_neg)
        es_set = set(tr_pos[: int(round(0.2 * len(tr_pos)))]) | \
            set(tr_neg[: int(round(0.2 * len(tr_neg)))])
        sets[w] = ([i for i in tr_idx if i not in es_set],
                   [i for i in tr_idx if i in es_set])
    return sets


es_split = inner_split()

# ---- configurable loss (v4): pos_weight in the loss; label smoothing
# applied to the TARGETS (NOT via the loss arg) because BCEWithLogitsLoss
# rejects pos_weight + label_smoothing together in torch>=2.2.
# y_t = y*(1-s) + 0.5*s  (standard symmetric smoothing; s=0 -> hard y)
PW = torch.tensor((1 - y[tr_idx].mean()) / y[tr_idx].mean()
                  if POS_W == "auto" else float(POS_W))
criterion = nn.BCEWithLogitsLoss(pos_weight=PW)


def smooth_targets(ys):
    """Symmetric label smoothing, a no-op when SMOOTH == 0."""
    if SMOOTH <= 0.0:
        return ys
    return ys * (1.0 - SMOOTH) + 0.5 * SMOOTH


def auc_of(model, w, idxs, chunk=512):
    """Chunked eval forward: the merged-source val set (n=3,864, seq 510)
    OOMs on a single (B, H, L, L) attention pass (~15 GiB); batch the
    scores, AUC is order-invariant."""
    idxs = np.asarray(idxs)
    model.eval()
    ps = np.empty(len(idxs), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idxs), chunk):
            b = idxs[s:s + chunk]
            tok, val, daypos, chpos, km, trend_t, v1_t, age_t, \
                imp_lbl_t, imp_val_t, ys = build_input_tensors(w, b)
            ps[s:s + chunk] = torch.sigmoid(
                model(tok.to(device), val.to(device),
                      daypos.to(device), chpos.to(device),
                      km.to(device),
                      trend_t.to(device) if trend_t is not None else None,
                      v1_t.to(device) if v1_t is not None else None,
                      age_t.to(device) if age_t is not None else None,
                      imp_lbl_t.to(device) if imp_lbl_t is not None
                      else None,
                      imp_val_t.to(device) if imp_val_t is not None
                      else None,
                      )).cpu().numpy()
    yb = y[idxs]
    return roc_auc_score(yb, ps), ps


results = []
roc_store = {}
for w in avail:
    inner_tr, inner_es = es_split[w]
    model = BertTSM(D_MODEL, W_MAX, C, pool=POOL,
                    value_mlp=VALUE_MLP, value_mode=VALUE_MODE,
                    sc_init=SC_INIT, miss_mode=MISS_MODE,
                    imp_mode=(IMPUTED is not None)).to(device)
    if INIT_FROM:
        _pt = torch.load(INIT_FROM, map_location=device)
        _pc = _pt["config"]
        if int(_pc["w_max"]) != W_MAX:
            raise SystemExit(f"--init_from w_max {_pc['w_max']} != {W_MAX}")
        if int(_pc["c_n"]) != C:
            raise SystemExit(f"--init_from c_n {_pc['c_n']} != {C}")
        if (int(_pc["n_layers"]) != N_LAYERS or int(_pc["n_heads"]) != N_HEADS
                or int(_pc["ffn"]) != FFN or int(_pc["d_model"]) != D_MODEL):
            raise SystemExit("--init_from geometry mismatch "
                             f"({_pc['n_layers']}L/{_pc['n_heads']}h/"
                             f"d{_pc['d_model']}/ffn{_pc['ffn']} vs "
                             f"{N_LAYERS}L/{N_HEADS}h/d{D_MODEL}/ffn{FFN})")
        if str(_pc.get("value_mode", "linear")) != VALUE_MODE:
            raise SystemExit(f"--init_from value_mode {_pc.get('value_mode')} "
                             f"!= this run {VALUE_MODE}")
        # blacklist: the task heads are re-initialized fresh; everything
        # else IS the pretrained backbone.  NB w_v/cls are leaf Parameters
        # (keys "w_v"/"cls", NO dot) so a prefix-style whitelist would drop
        # them - a blacklist over the head keys is the robust form.
        _skip = ("head.", "pt_head.")
        _loaded = {k: v for k, v in _pt["state_dict"].items()
                   if not k.startswith(_skip)}
        model.load_state_dict(_loaded, strict=False)
        if "e_miss.weight" in _pt["state_dict"] and \
                model.miss_mode not in ("embed", "trend"):
            print(f"init_from: e_miss pretrained but this run is miss_mode="
                  f"{MISS_MODE} (no e_miss path) - backbone loaded without "
                  f"it (pad-mode finetune keeps v3.7 missing handling)")
        # e_trend has no pretrained counterpart (pretrain is embed mode);
        # it stays randomly initialized in trend finetune
        _pre = _pt["config"].get("pretrain", {})
        print(f"init_from = {INIT_FROM}")
        print(f"  copied {len(_loaded)} backbone keys | pretrain "
              f"ep={_pt.get('best_epoch')} es_mse={_pre.get('es_mse')} "
              f"w={_pre.get('window')} trend_w={_pre.get('trend_w')}")
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # optional LR schedule: linear warmup -> cosine decay -> 1e-5
    use_sched = (WARMUP > 0 or SCHED == "cosine")
    if use_sched:
        import math as _m
        def _sched(epoch):
            if epoch <= WARMUP:
                return max(1.0, epoch) / max(1.0, WARMUP)
            if SCHED == "cosine":
                p = (epoch - WARMUP) / max(1, MAX_EPOCH - WARMUP)
                return 1e-5 + 0.5 * (1 + _m.cos(p * _m.pi)) * LR
            return LR
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_sched)

    def step(idxs):
        optim.zero_grad()
        tok, val, daypos, chpos, km, trend_t, v1_t, age_t, \
            imp_lbl_t, imp_val_t, ys = build_input_tensors(w, idxs)
        model.train()
        loss = criterion(model(tok.to(device), val.to(device),
                               daypos.to(device), chpos.to(device),
                               km.to(device),
                               trend_t.to(device)
                               if trend_t is not None else None,
                               v1_t.to(device) if v1_t is not None
                               else None,
                               age_t.to(device) if age_t is not None
                               else None,
                               imp_lbl_t.to(device)
                               if imp_lbl_t is not None else None,
                               imp_val_t.to(device)
                               if imp_val_t is not None else None),
                         smooth_targets(ys.to(device).float()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    best_auc, best_state, epoch_stop, bad = -1.0, None, 0, 0
    epoch = 0
    for epoch in range(1, MAX_EPOCH + 1):
        model.train()
        perm = np.random.permutation(inner_tr)
        for s in range(0, len(perm), BATCH):
            step(perm[s:s + BATCH])
        es_auc, _ = auc_of(model, w, inner_es)
        if use_sched:
            sched.step()
        if es_auc > best_auc:
            best_auc = es_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epoch_stop, bad = epoch, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.to(device)

    va_auc, p_va = auc_of(model, w, list(va_idx))
    tri = list(inner_tr) + list(inner_es)
    tr_auc, p_tr = auc_of(model, w, tri)
    bri_va = brier_score_loss(y[va_idx], p_va)
    print(f"window={w}d  trained={epoch}ep  best={epoch_stop}ep  "
          f"AUC_train={tr_auc:.4f}  AUC_val={va_auc:.4f}  "
          f"Brier_val={bri_va:.4f}  seed={SEED}\n")
    results.append({"window": w, "epoch": epoch_stop, "seed": SEED,
                    "auc_train": round(tr_auc, 4),
                    "auc_val": round(va_auc, 4),
                    "brier_val": round(bri_va, 4)})
    fpr_v, tpr_v, _ = roc_curve(y[va_idx], p_va)
    fpr_t, tpr_t, _ = roc_curve(y[np.array(tri)], p_tr)
    roc_store[w] = (fpr_v, tpr_v, fpr_t, tpr_t)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"w_max": W_MAX, "c_n": C, "d_model": D_MODEL,
                   "n_heads": N_HEADS, "n_layers": N_LAYERS, "ffn": FFN,
                   "dropout": DROPOUT, "window": w,
                   "token_mode": "linear_value", "pos_enc": "rope",
                   "pool": POOL, "value_mlp": VALUE_MLP,
                   "value_mode": VALUE_MODE, "miss_mode": MISS_MODE,
                   "imp_mode": (IMPUTED is not None),
                   "seed": SEED},
        "auc_val": va_auc, "auc_train": tr_auc,
        "pos_weight": float(PW),
        "npz": args.npz,
        "imputed": IMPUTED,
    }, os.path.join(args.out_dir, f"bert_trm_w{w}.pt"))


def plot_combined(split, title, ls, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = sorted(roc_store)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, w in enumerate(ws):
        fpr, tpr = roc_store[w][ls]
        ax.plot(fpr, tpr, lw=2.2,
                color=cmap(k / max(len(ws) - 1, 1)),
                label=f"w={w}d  AUC={split[w]:.4f}")
    ax.plot([0, 1], [0, 1], ":", color="k", lw=0.7)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, fname))
    plt.close(fig)


plot_combined({r["window"]: r["auc_val"] for r in results},
              "BERT model (linear value + RoPE) - VAL ROC (all windows)", slice(0, 2),
              "roc_val_all_windows.pdf")
plot_combined({r["window"]: r["auc_train"] for r in results},
              "BERT model (linear value + RoPE) - TRAIN ROC (all windows)", slice(2, 4),
              "roc_train_all_windows.pdf")
pd.DataFrame(results).to_csv(os.path.join(args.out_dir,
                                          "bert_trm_metrics.csv"),
                             index=False)

print("== summary ==")
for r in results:
    print(f"  window {r['window']}d: AUC_val={r['auc_val']:.4f} "
          f"(train={r['auc_train']:.4f}) Brier={r['brier_val']:.4f}")
print(f"\nsaved -> {args.out_dir}/  (bert_trm_w*.pt, "
      f"roc_*_all_windows.pdf, bert_trm_metrics.csv)")
