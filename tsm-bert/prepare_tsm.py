"""
prepare_tsm.py - one-shot data preparation for the temporal sepsis model.

Does one job then stops:
  1. load trajectories + observation mask from <data_dir>/trajectory.h5
  2. join outcome label (hospital_expire_flag) from <data_dir>/mimi3_diag.csv
  3. split 70/30 at hadm_id level (one hadm = one sample), stratified by
     label, fixed seed
  4. per-channel robust standardization params (median / IQR) from TRAIN
     only
  5. save data + all params to <out_dir>/tsm_split.npz (single source of
     truth for all downstream scripts)

Run (single line):
  python prepare_tsm.py --data_dir <cohort_dir> --scalars_json scalers.json

Arguments:
  --data_dir   input folder containing trajectory.h5 and mimi3_diag.csv
  --out_dir    output folder for tsm_split.npz and prep_report.txt
  --scalars_json  SOURCE scalars.json (aggregate median/iqr + channel order)
               to standardize THIS data with (cross-dataset verification).
               If omitted, median/iqr are fit on THIS dataset's own TRAIN
               rows. Target runs must pass the SOURCE file so that
               cross-cohort standardization never leaks.
  --drop_channels  channel names to drop after the h5 load (release trains
               on 15 channels: crp and fibrinogen are dropped from the
               17-channel h5; keep the same list as the source run so the
               channel order matches scalars.json).
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

SEED = 42
# prediction windows: keep only the first N days of the 163-day trajectory
# (7 = early, 14 = mid, 30 = full-observation window)
WINDOWS = (7, 14, 30)
HERE = os.path.dirname(os.path.abspath(__file__))      # this repo root
ROOT = os.path.dirname(HERE)


def resolve_input_paths(data_dir):
    """Locate the two mandatory input files inside data_dir and validate.

    Returns (traj_h5, diag_csv, channels) after a read-only probe of the h5
    so that structural problems surface HERE, not mid-run.
    """
    traj_h5 = os.path.join(data_dir, "trajectory.h5")
    diag_csv = os.path.join(data_dir, "mimi3_diag.csv")
    for p in (traj_h5, diag_csv):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"required input not found: {p}")

    with h5py.File(traj_h5) as f:
        need = {"trajectories", "observation_mask", "hadm_ids",
                "channel_names"}
        missing = need - set(f.keys())
        if missing:
            raise KeyError(f"trajectory.h5 missing keys: {sorted(missing)}")
        tr_shape = f["trajectories"].shape
        ob_shape = f["observation_mask"].shape
        if tr_shape[:2] != ob_shape[:2] or tr_shape[2] != ob_shape[2]:
            raise ValueError(
                f"shape mismatch: trajectories {tr_shape} vs "
                f"observation_mask {ob_shape}")
        channels = [c.decode() for c in f["channel_names"][:]]
    print(f"input probe OK: {traj_h5}\n"
          f"  trajectories {tr_shape} | mask {ob_shape} | "
          f"channels={channels}")
    return traj_h5, diag_csv, channels


def load_data(traj_h5, diag_csv):
    """Load raw arrays + joined labels. One hadm = one row."""
    with h5py.File(traj_h5) as f:
        traj = f["trajectories"][:]                        # (N, T, C)
        obs = f["observation_mask"][:]                     # (N, T, C)
        hadms = f["hadm_ids"][:]                           # (N,)
        channels = [c.decode() for c in f["channel_names"][:]]

    diag = pd.read_csv(diag_csv, usecols=["hadm_id",
                                          "hospital_expire_flag"])
    if len(diag) != diag.hadm_id.nunique():
        raise ValueError("duplicate hadm_id in diag csv")
    y = (diag.set_index("hadm_id")
              .reindex(hadms)
              .hospital_expire_flag)
    if y.isna().any():
        raise ValueError(f"missing labels for "
                         f"{int(y.isna().sum())} hadms (not in diag csv)")
    y = y.values.astype(np.int8)
    return traj, obs, hadms, channels, y


def make_hadm_split(y, seed=SEED, train_ratio=0.7):
    """Stratified 70/30 index split at hadm level, fixed seed.

    Returns (tr_idx, va_idx). Deterministic: same y + seed -> same split.
    """
    rng = np.random.default_rng(seed)
    tr_idx_list, va_idx_list = [], []
    for pos in (1, 0):
        idx = np.where(y == pos)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * train_ratio))
        tr_idx_list.append(idx[:n_tr])
        va_idx_list.append(idx[n_tr:])
    tr_idx = np.sort(np.concatenate(tr_idx_list))
    va_idx = np.sort(np.concatenate(va_idx_list))
    if len(np.intersect1d(tr_idx, va_idx)) > 0:
        raise AssertionError("LEAK: an index is in both train and val")
    return tr_idx, va_idx


def fit_robust_scaler(trajectories, mask, channels, tr_mask, T):
    """Per-channel median / IQR from TRAIN, observed cells only.

    Robust to heavy tails (lactate 20+, AST 28000). Constant or all-missing
    channels get IQR=1 and are reported; their info travels via the mask.

    Returns (median, iqr, zero_iqr_channels).
    """
    C = trajectories.shape[2]
    med = np.zeros(C, dtype=np.float64)
    iqr = np.ones(C, dtype=np.float64)
    zero_iqr = []
    for c in range(C):
        xc = trajectories[:, :, c].reshape(-1)             # (N*T,)
        mc = (mask[:, :, c].reshape(-1).astype(bool)
              & np.repeat(tr_mask, T)
              & np.isfinite(xc))
        v = xc[mc]
        if len(v) == 0:
            zero_iqr.append(channels[c])
            continue
        med[c] = np.median(v)
        q75, q25 = np.percentile(v, [75, 25])
        if q75 - q25 <= 0:
            zero_iqr.append(channels[c])
        else:
            iqr[c] = q75 - q25
    return med, iqr, zero_iqr


def build_window_inputs(trajectories, mask, med, iqr, windows):
    """Standardize + append per-channel mask.

    Layout of std_X_w (N, w, 2C): cols [0:C] standardized values,
    cols [C:2C] the same channels' observation mask. Per-channel mask
    (not a single scalar) because missingness is per channel.
    Non-finite values -> 0.
    """
    out = {}
    for w in windows:
        xv = (trajectories[:, :w, :] - med) / iqr
        xv = np.where(np.isfinite(xv), xv, 0.0).astype(np.float32)
        xv = np.concatenate(
            [xv, mask[:, :w, :].astype(np.float32)], axis=2)
        out[f"std_X_{w}"] = xv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "outputs"),
                    help="input folder: trajectory.h5 + mimi3_diag.csv")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "data"),
                    help="output folder for tsm_split.npz + prep_report.txt")
    ap.add_argument("--scalars_json", default=None,
                    help="SOURCE scalars.json (aggregate median/iqr/channel "
                         "order) to standardize THIS data with (cross-dataset "
                         "verification). If set, the target split is IGNORED "
                         "for scaling (no leakage). Requires matching channel "
                         "count/order.")
    ap.add_argument("--drop_channels", nargs="*", default=None,
                    metavar="CHAN",
                    help="channel names to drop AFTER the h5 load (the h5 "
                         "keeps all channels; the npz is written without "
                         "these).  source and every target run must pass "
                         "the SAME list so --scale_npz channel check holds")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_npz = os.path.join(args.out_dir, "tsm_split.npz")

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    resolve_input_paths(args.data_dir)                     # probe + fail fast

    traj, obs, hadms, channels, y = load_data(
        os.path.join(args.data_dir, "trajectory.h5"),
        os.path.join(args.data_dir, "mimi3_diag.csv"))
    # --drop_channels: slice OUT the named channels right after the h5 load
    # (h5 file untouched on disk; everything from here on - split, scaling,
    # windows, saved npz - operates on the reduced set). Names must exist;
    # order of the remaining channels is preserved so source/target runs
    # that pass the same list are --scale_npz-compatible.
    if args.drop_channels:
        drop_idx = []
        for name in args.drop_channels:
            if name not in channels:
                raise SystemExit(f"--drop_channels: {name!r} not in "
                                 f"h5 channels {channels}")
            drop_idx.append(channels.index(name))
        keep_idx = [i for i in range(len(channels)) if i not in drop_idx]
        traj = traj[:, :, keep_idx]
        obs = obs[:, :, keep_idx]
        channels = [channels[i] for i in keep_idx]
        log_note = (f"dropped channels {args.drop_channels} -> "
                    f"{len(channels)} channels: {channels}")
        print(log_note)
    N, T, C = traj.shape
    log(f"loaded: {N} hadms x {T} days x {C} channels")
    log(f"label: pos={int(y.sum())} rate={y.mean():.3f}")

    tr_idx, va_idx = make_hadm_split(y)
    tr_m, va_m = np.zeros(N, bool), np.zeros(N, bool)
    tr_m[tr_idx], va_m[va_idx] = True, True
    log(f"\n== 70/30 stratified split (hadm level, seed {SEED}) ==")
    for name, m in (("train", tr_m), ("val", va_m)):
        i = np.where(m)[0]
        log(f"{name:5s}: {len(i):4d} hadms | pos={int(y[i].sum()):4d} "
            f"rate={y[i].mean():.3f}")

    if args.scalars_json:
        # cross-dataset: reuse SOURCE scaling, never fit on the target set
        import json
        src = json.load(open(args.scalars_json))
        src_ch = list(src["channels"])
        if src_ch != channels:
            raise ValueError(
                f"channel mismatch: source {src_ch} vs target {channels}")
        med = np.asarray(src["median"], dtype=np.float64)
        iqr = np.asarray(src["iqr"], dtype=np.float64)
        iqr[iqr <= 0] = 1.0
        log(f"\nscaling  = SOURCE params from {args.scalars_json} "
            f"(target set NOT used for scaling)")
    else:
        med, iqr, zero_iqr = fit_robust_scaler(traj, obs, channels, tr_m, T)
        if zero_iqr:
            log(f"\nconstant / all-NaN channels (pinned to 0, mask carries "
                f"info): {zero_iqr}")

    blocks = build_window_inputs(traj, obs, med, iqr, WINDOWS)

    np.savez(out_npz,
             hadm_ids=hadms,
             tr_idx=tr_idx,
             va_idx=va_idx,
             channels=np.array(channels),
             median=med,
             iqr=iqr,
             windows=np.array(WINDOWS, np.int8),
             y=y,
             **blocks)
    log(f"\nsaved -> {out_npz}")
    log(f"std_X_w shapes: "
        f"{ {k: v.shape for k, v in blocks.items()} }")

    report = os.path.join(args.out_dir, "prep_report.txt")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
