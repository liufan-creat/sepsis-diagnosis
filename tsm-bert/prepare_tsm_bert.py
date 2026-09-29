"""
prepare_tsm_bert.py - tokenization of the tsm windows (continuous values,
no discretization).

One job: turn each std_X_w (standardized value + per-channel obs mask)
into a token vocabulary.  No value amplitude, no interpolation,
no UP/DOWN/NORMAL trend tokens, no discretization:

  token(day t, channel c), index = t*C + c:
    observed, value inside that channel's source range -> OBS (1)
    observed, value outside the source range          -> PAD (0)
    missing cell                                      -> PAD (0)
  no bins: the model maps the standardized value to a vector with a
  per-channel linear layer (W_v[ch] * val, see model_bert.py).
ONCE on the SOURCE training set (observed cells of std_X_{w_max}; w_max
is a prefix window, so it contains every cell of the shorter windows).
Edges are stored in the token npz (tok_edges).  For a TARGET cohort pass
--bounds_npz <source tokens.npz> and the source edges are reused
verbatim - same no-leakage discipline as prepare_tsm.py --scale_npz
(target data never touches the token fit).

Output npz, per window w:
  bert_tok_{w}     int8  (N, w*C)   0 = PAD, 1 = observed (indicator)
  bert_val_{w}     f32   (N, w*C)   standardized value at observed,
                                   in-range cells; 0 otherwise (PAD cells
                                   carry no value -> zero injection)
  bert_daypos_{w}  int8  (N, w*C)   day id 0..w-1
  bert_chpos_{w}  int8  (N, w*C)   channel id 0..C-1

MISS-TOKEN MODE (--miss_mode embed): split missingness at the per-hadm
right-censorship boundary = the LAST day with any observed channel (for
death patients the data stops at death, for discharged patients at
discharge - both are right-censoring, identical in information structure,
never a label).  After that day EVERY cell becomes PAD (censorship,
excluded from attention); missing cells BEFORE it become MISS (2).
Token vocabulary with --miss_mode embed:
    1 = OBS (observed, in source range)
    2 = MISS (missing before the censorship boundary -> [MISS] token,
              participates in attention, carries no value)
    0 = PAD (post-censorship OR observed-but-out-of-source-range;
             excluded from attention, zero injection)
Default --miss_mode pad reproduces v3.7 exactly (MISS folded into PAD).
The model (model_bert.py / eval_bert.py, miss_mode embedded in the
checkpoint config) decides how tok=2 is consumed.
shared: hadm_ids, tr_idx, va_idx, channels, windows, y,
        n_bins (int), tok_edges (C, N_BINS+1) float64

Run (single line):
  python prepare_tsm_bert.py --npz <tsm_split.npz> --bounds_npz <SOURCE tsm_bert_tokens.npz> --out_npz <out.npz> --miss_mode embed

Arguments:
  --npz         source/target tsm_split.npz (default: <here>/data/tsm_split.npz)
  --bounds_npz  source TOKENS npz whose tok_edges bin THIS data (cross-
                dataset verification).  If omitted, edges are FIT on this
                npz's own train rows
  --out_npz     output token npz (default: <here>/data/tsm_bert_tokens.npz)
  --bins        quantile grid size (default 16). NOT a token bucket count:
                tokens are continuous (no discretization). The grid is used
                only for its MIN/MAX edges -> the OOV->PAD cutoff, and for
                --bounds_npz consistency checks between source and target.
  --raw         RAW-VALUE mode (new architecture): val carries the RAW
                physiological value recovered by inverse standardization
                (raw = std*iqr + median with the median/iqr stored in the
                SAME tsm_split.npz); PAD = missing ONLY (no OOV->PAD).
                The model applies its own learnable per-channel scale
                (init 1/p99|raw|, a data-derived constant stored as
                sc_init) so no fixed normalization is imposed. tok_edges
                are then unused (--bounds_npz is ignored in this mode).
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--npz", default=os.path.join(HERE, "data", "tsm_split.npz"))
ap.add_argument("--bounds_npz", default=None,
                help="source TOKENS npz whose tok_edges bin THIS data "
                     "(target data NOT used for the fit); if omitted, "
                     "edges are fit on this npz's train rows")
ap.add_argument("--out_npz", default=os.path.join(HERE, "data",
                                                  "tsm_bert_tokens.npz"))
ap.add_argument("--bins", type=int, default=16,
                help="quantile grid size (default 16) - NOT a token bucket "
                     "count (tokens are continuous); grid min/max only set "
                     "the OOV->PAD cutoff and source/target consistency check")
ap.add_argument("--raw", action="store_true",
                help="raw-value mode: val stores RAW physiological values "
                     "(inverse-standardized with the npz's own median/iqr), "
                     "PAD = missing only; stores sc_init = 1/p99|raw| for "
                     "the model's learnable per-channel scale")
ap.add_argument("--miss_mode", choices=["pad", "embed", "trend", "last"],
            default="pad",
            help="pad = v3.7 default (all missing/OOV cells -> PAD 0); "
                 "embed = split missingness at the per-hadm right-"
                 "censorship boundary (last day with any observation): "
                 "missing cells before it -> MISS (tok 2), everything "
                 "after it -> PAD (tok 0); model consumes tok 2 as "
                 "e_miss[day] + e_ch. trend = SAME tokenization as "
                 "embed, PLUS a bert_trend_<w> label tensor on the tok 2 "
                 "cells: relative change of the LAST VISIBLE SEGMENT "
                 "before the gap vs the prior OBS pair within reach "
                 "(0=NA no usable pair / 1=down / 2=flat / 3=up). last = "
                 "SAME tokenization as embed, PLUS bert_v1_<w> (standardized "
                 "value of the channel's most recent prior OBS inside the "
                 "window, no reach cutoff - the real last-known reading, "
                 "0 when the channel was never observed pre-gap) and "
                 "bert_age_<w> (age bucket of that prior reading: "
                 "0=NA / 1=1-3d / 2=4-7d / 3=8-14d / 4=>14d). model "
                 "consumes tok 2 as e_miss + e_ch + W_last[ch]*v1 + "
                 "e_age[age].")
args = ap.parse_args()
RAW = args.raw
MISS_MODE = args.miss_mode if not args.raw else "pad"
if RAW and args.miss_mode != "pad":
    print("NOTE: --raw forces miss_mode=pad (v3.7 parity)")
TREND_REACH = 7      # days: prior-OBS reach window for trend labels
TREND_EPS = 0.5      # |delta| < 0.5 std units -> flat band (global)
# event/continuous channels keep NA (lab-interval semantics do not apply)
TREND_EXCLUDE = {"map", "urine"}

d = np.load(args.npz, allow_pickle=False)
ch = list(d["channels"])
C = len(ch)
y = d["y"]
tr_idx = d["tr_idx"]
w_max = max(int(w) for w in d["windows"])
X_all = {int(w): d[f"std_X_{w}"] for w in d["windows"]}
N_BINS = args.bins

# ------------------------------------- per-channel quantile grid
# (OOV->PAD cutoff via min/max edges only; no discretization of tokens)
# raw-value mode: no bucketing needed (the model learns its own scale);
# EDGES stay unset and the in_range logic below is bypassed.
MED = d["median"].astype(np.float64)
IQR = d["iqr"].astype(np.float64)
if RAW:
    print("RAW mode: val = raw physiological values (inverse-standardized "
          "with the npz's own median/iqr); scale learned by the model "
          "(init 1/p99|raw|); PAD = missing only")
elif args.bounds_npz:
    src = np.load(args.bounds_npz, allow_pickle=False)
    EDGES = src["tok_edges"].astype(np.float64).copy()
    if int(src["n_bins"]) != N_BINS:
        raise ValueError(f"--bins {N_BINS} != source n_bins "
                         f"{int(src['n_bins'])}")
    if list(src["channels"]) != ch:
        raise ValueError("channel mismatch with the source tokens npz")
    print(f"OOV edges (quantile grid min/max) = SOURCE fit from {args.bounds_npz} "
          f"(target data NOT used for the fit)")
else:
    # source mode: fit on THIS train set, observed cells of the w_max window
    V0 = X_all[w_max][:, :, :C]
    M0 = X_all[w_max][:, :, C:2 * C] > 0
    trV, trM = V0[tr_idx], M0[tr_idx]
    pooled = trV[trM]
    pooled_edges = (np.quantile(pooled, np.linspace(0, 1, N_BINS + 1))
                    if pooled.size else np.linspace(-5, 5, N_BINS + 1))
    print(f"OOV quantile grid FIT on this npz train rows "
          f"({int(tr_idx.size)} hadms x {w_max}d, observed cells only)")
    EDGES = np.zeros((C, N_BINS + 1), dtype=np.float64)
    no_obs = []
    for c in range(C):
        v = trV[:, :, c][trM[:, :, c]]
        if v.size:
            e = np.quantile(v, np.linspace(0, 1, N_BINS + 1))
            if e[-1] <= e[0]:        # constant channel -> tiny spread
                e = np.linspace(e[0] - 0.5, e[0] + 0.5, N_BINS + 1)
            EDGES[c] = e
        else:
            EDGES[c] = pooled_edges
            no_obs.append(ch[c])
    if no_obs:
        print(f"no observed train cell (pooled fallback): {no_obs}")

N = X_all[w_max].shape[0]

# ---- right-censorship boundary (miss_mode=embed only)
# per-hadm last day with ANY observed channel in the w_max window.  The
# data stops there (death or discharge); days after it carry no possible
# information.  Uses the OBSERVATION MASK ONLY - never the label - so no
# leakage.  hadms with no observation at all get last_day = -1 (entirely
# post-censorship -> all PAD, CLS survives via the never-masked CLS key).
if MISS_MODE in ("embed", "trend", "last"):
    Mmax = X_all[w_max][:, :, C:2 * C] > 0
    anyday = Mmax.any(axis=2)                         # (N, w_max)
    last_day = np.where(
        anyday.any(axis=1),
        w_max - 1 - np.argmax(anyday[:, ::-1], axis=1),
        -1).astype(np.int64)                          # (N,)
    print(f"miss_mode={MISS_MODE}: censorship boundary = last day with any "
          f"observation (mask-derived, label-free);\n"
          f"  last_day distribution: min={last_day.min()} "
          f"max={last_day.max()}  no-obs hadms={int((last_day < 0).sum())}")
else:
    last_day = None


def trend_labels(tok3, V3):
    """4-class trend label on the tok-2 (MISS) cells of ONE window slice.

    tok3/V3: (N, w, C) final tokens + standardized values of window w.
    Label of a MISS cell at day d = the direction of the LAST VISIBLE
    segment of its channel before the gap (a MISS cell has no current
    value - this is the only well-defined reading of "vs prior value"):
      v1 = most recent OBS of the channel at some day < d
      v0 = the OBS before v1
      delta = v1 - v0  (standardized units, source-scaled)
      delta >= +TREND_EPS      -> 3 (up)
      |delta| <  TREND_EPS     -> 2 (flat)
      delta <= -TREND_EPS      -> 1 (down)
      reference pair unusable  -> 0 (NA): no v1 within TREND_REACH days,
        no v0 within TREND_REACH of v1, single-observation channels,
        or channel in TREND_EXCLUDE (MAP/URINE: event/continuous, not
        interval labs).  Deterministic, window-internal only (the model
        never sees pre-window measurements), never touches the label.
    """
    N_, W_, C_ = tok3.shape
    tr = np.zeros((N_, W_, C_), dtype=np.int8)
    for c in range(C_):
        if ch[c] in TREND_EXCLUDE:
            continue
        v1 = np.full(N_, np.nan)          # most recent prior OBS value
        d1 = np.full(N_, -1)              # ... day of v1
        v0 = np.full(N_, np.nan)          # second most recent OBS value
        d0 = np.full(N_, -1)              # ... day of v0
        obsc = (tok3[:, :, c] == 1)
        for dd in range(W_):
            missrow = (tok3[:, dd, c] == 2)
            if missrow.any():
                ok = (missrow & np.isfinite(v1) & np.isfinite(v0)
                      & (dd - d1 <= TREND_REACH) & (d1 - d0 <= TREND_REACH))
                delta = v1 - v0
                lab = np.where(delta >= TREND_EPS, 3,
                               np.where(delta <= -TREND_EPS, 1, 2))
                tr[ok, dd, c] = lab[ok].astype(np.int8)
            u = obsc[:, dd]
            if u.any():
                v0 = np.where(u, v1, v0)
                d0 = np.where(u, d1, d0)
                v1 = np.where(u, V3[:, dd, c], v1)
                d1 = np.where(u, dd, d1)
    return tr


def last_v1_age(tok3, V3):
    """(v1, age) of the channel's MOST RECENT prior OBS for every cell of
    ONE window slice (N, w, C), computed on the final tokens.

    v1[i,d,c]  = standardized value of the nearest OBS of channel c at
                  some day < d, window-internal only; 0.0 when the channel
                  was never observed before day d (the model must not
                  mistake 0 for a real zero reading - the age bucket says NA).
    age[i,d,c] = bucket of the gap (d - day_of_v1):
                  0 = NA (no prior)  1 = 1-3d  2 = 4-7d  3 = 8-14d  4 = >14d
    Deterministic, mask-derived, never touches the label.  Windows are
    PREFIXES of w_max (day 0 aligned), so recomputing per window slice is
    exact.
    """
    N_, W_, C_ = tok3.shape
    v1 = np.zeros((N_, W_, C_))
    age = np.zeros((N_, W_, C_), dtype=np.int8)
    for c in range(C_):
        v = np.full(N_, np.nan)          # running most-recent OBS value
        dd0 = np.full(N_, -1)            # ... its day
        obsc = (tok3[:, :, c] == 1)
        for dd in range(W_):
            gap = dd - dd0
            has = np.isfinite(v)
            age[:, dd, c] = np.where(
                has, np.where(gap <= 3, 1,
                              np.where(gap <= 7, 2,
                                       np.where(gap <= 14, 3, 4))), 0)
            v1[:, dd, c] = np.where(has, v, 0.0)
            u = obsc[:, dd]
            if u.any():
                v = np.where(u, V3[:, dd, c], v)
                dd0 = np.where(u, dd, dd0)
    return v1, age

blocks = {}
print(f"\n{'w':>3}  {'cells':>8}  {'obs':>8}  {'OOV->PAD':>12}  "
      f"{'PAD':>8}\n---")
for w in sorted(X_all):
    Xw = X_all[w]                       # (N, w, 2C)
    V = Xw[:, :, :C].astype(np.float64)            # value lanes
    M = Xw[:, :, C:2 * C] > 0           # (N, w, C) observation mask
    if RAW:
        # raw mode: inverse-standardize to physiological values,
        # missing cells stay 0 (PAD carries no value)
        V = V * IQR + MED
        V = np.where(np.isfinite(V), V, 0.0)
        in_range = np.ones_like(M, dtype=bool)     # bypass OOV clipping
        n_oov = 0
    else:
        # observed & inside the source range -> OBS 1, else PAD 0 (the
        # model's per-channel linear value path carries the magnitude)
        in_range = (V >= EDGES[:, 0]) & (V <= EDGES[:, -1])
        n_oov = int((M & ~in_range & np.isfinite(V)).sum())
    tok = np.where(M & in_range, 1, 0).astype(np.int8)
    tok = np.where(np.isfinite(V), tok, 0).astype(np.int8)
    base_day = np.repeat(np.arange(w, dtype=np.int8), C)
    base_ch = np.tile(np.arange(C, dtype=np.int8), w)
    daypos = np.tile(base_day, (N, 1))
    chpos = np.tile(base_ch, (N, 1))
    # embed mode: missing cells at/before the censorship boundary become
    # MISS (2); after the boundary they stay PAD (0) - the post-boundary
    # days are missing by construction (last observed day), so only the
    # pre-boundary assignment changes anything.  OOV observed cells keep
    # v3.7 behavior: PAD (0).
    n_miss = 0
    if MISS_MODE in ("embed", "trend", "last"):
        day_axis = (np.arange(w) <= last_day[:, None])        # (N, w)
        miss_pre = (~M) & day_axis[:, :, None]                # (N, w, C)
        tok = np.where(miss_pre, 2, tok).astype(np.int8)
        n_miss = int(miss_pre.sum())
    if MISS_MODE == "trend":
        tr = trend_labels(tok, V)
        missflat = (tok == 2).ravel()
        lab = np.bincount(tr.ravel()[missflat], minlength=4)
        print(f"         trend labels on {int(missflat.sum())} MISS cells: "
              f"NA={lab[0]} down={lab[1]} flat={lab[2]} up={lab[3]} "
              f"(reach={TREND_REACH}d, eps={TREND_EPS})")
        blocks[f"bert_trend_{w}"] = tr.reshape(N, w * C).astype(np.int8)
    if MISS_MODE == "last":
        v1f, agef = last_v1_age(tok, V)
        missflat = (tok == 2).ravel()
        lab = np.bincount(agef.ravel()[missflat], minlength=5)
        print(f"         v1/age on {int(missflat.sum())} MISS cells "
              f"(age buckets NA/1-3d/4-7d/8-14d/>14d): "
              f"{lab[0]}/{lab[1]}/{lab[2]}/{lab[3]}/{lab[4]}")
        # keep the sidecars at MISS cells only - OBS/PAD slots must not
        # carry a stale "prior value" into the model (zero elsewhere)
        blocks[f"bert_v1_{w}"] = np.where(
            missflat.reshape(N, w * C), v1f.reshape(N, w * C),
            0.0).astype(np.float32)
        blocks[f"bert_age_{w}"] = agef.reshape(N, w * C).astype(np.int8)
    n_obs = int(M.sum())
    n_pad = int((tok == 0).sum())
    print(f"{w:3d}  {V.size:8d}  {n_obs:8d}  "
          f"{n_oov:6d} (oov{' ' if RAW else ': '})  {n_pad:8d}")
    print(f"         OBS->token {int((tok == 1).sum())} | "
          f"MISS (pre-censor) {n_miss} | PAD {int((tok == 0).sum())}")
    val = np.where((M & in_range) & (tok == 1), V, 0.0)  # value-injection input
    val = val.reshape(N, w * C).astype(np.float32)
    blocks[f"bert_tok_{w}"] = tok.reshape(N, w * C).astype(np.int8)
    blocks[f"bert_val_{w}"] = val
    blocks[f"bert_daypos_{w}"] = daypos
    blocks[f"bert_chpos_{w}"] = chpos

# raw mode: sc_init = 1/p99(|raw|) over the TRAIN observed cells of the
# w_max window. 1/iqr is rejected: narrow-distribution channels (ph iqr
# 0.105 -> 1/iqr 9.5) map the median to ~70 and saturate the clamp=20
# bound on init, destroying the channel's contrast. 1/p99 puts the 99th
# percentile at ~1.0 for every channel, so no clamp fires at init and the
# learnable scale starts in a sensible, unit-magnitude regime.
# (constant init, no RNG draw; tok_edges unused, kept as zeros for schema)
if RAW:
    edges_store = np.zeros((C, 1), dtype=np.float64)
    Xmax_ = X_all[w_max]
    Vmax = Xmax_[:, :, :C].astype(np.float64)
    Mmax = Xmax_[:, :, C:2 * C] > 0
    rawmax = Vmax * IQR + MED
    rawmax = np.where(np.isfinite(rawmax), rawmax, 0.0)
    p99 = np.zeros(C, dtype=np.float64)
    for c in range(C):
        vc = rawmax[tr_idx, :, c][Mmax[tr_idx, :, c]]
        vc = np.abs(vc)
        p99[c] = np.percentile(vc, 99) if vc.size else 1.0
    sc_init = 1.0 / np.maximum(p99, 1e-9)
    extra = {"sc_init": sc_init.astype(np.float64)}
else:
    edges_store = EDGES
    extra = {}
np.savez(args.out_npz,
         miss_mode=np.array(MISS_MODE),
         hadm_ids=d["hadm_ids"],
         tr_idx=tr_idx,
         va_idx=d["va_idx"],
         channels=np.array(ch),
         windows=d["windows"],
         y=y,
         n_bins=np.int64(N_BINS),
         tok_edges=edges_store,
         **extra,
         **blocks)
print(f"\nsaved -> {args.out_npz}")
print("token code: tok 0=PAD 1=OBS 2=MISS(miss_mode=embed) , val = "
      + ("RAW physiological value (model: W_v[ch]*clamp(raw*scale_ch), "
         "scale init 1/p99|raw|) (miss_mode forced to pad)" if RAW else
         "standardized value (model: W_v[ch]*val)"))
if RAW:
    print(f"sc_init (1/p99|raw|): {np.round(sc_init, 4)}")
