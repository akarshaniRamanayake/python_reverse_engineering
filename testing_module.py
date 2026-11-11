"""Testing module: how close is a simulation to spike_data.npz?

Reverse Engineering Project, Part 2 (follow-up to Assignments 2A and 2B).
Author: Abdulhamit Gokce.

What it does
    1. Loading (T0): the candidate file is read without pickled objects.
       Files from other teams often differ from spike_data.npz (missing
       ``t``, other key names, float32, shape (1, 10000), .npy/.csv
       instead of .npz, a few NaNs). The loader repairs what can be
       repaired without guessing the signal itself and records every
       repair. Only files with no usable 10000-sample signal get 0.
    2. Structure check: the strict comparison with spike_data.npz (keys
       exactly t and raw_data, float64, shape (10000,), no NaN/Inf, t equal
       to the original) is still reported. A file that needed repairs is
       scored, but its composite is reduced by SCHEMA_PENALTY (10 %).
       With --strict the old rule applies: any failed check -> 0.
    3. Measures 10 properties in the candidate and in the original.
    4. Turns every difference (error e, 0 = identical) into a score with
       one rule:   score = min(1, tolerance / e)
       tolerance = the error that two runs of a perfect generator show by
       chance (95th percentile, calibrated in validate_and_plot.py).
       Chance-level error -> 1, twice the chance level -> 0.5.
    5. Tier score = mean of its metric scores;
       composite  = 100 * mean of the four tier scores.

Usage
    python testing_module.py candidate.npz [more.npz ...]
    python testing_module.py teams_folder/            # every file inside
    python testing_module.py candidate.npz --reference path/spike_data.npz
    python testing_module.py teams_folder/ --csv ranking.csv
    python testing_module.py candidate.npz --strict   # no repairs allowed

    from testing_module import evaluate
    print(evaluate("candidate.npz")["composite"])

Requirements: numpy, scipy. No random numbers: same file -> same score.
One unreadable file never stops the ranking of the others.
"""
import argparse
import csv
import glob
import os

import numpy as np
from scipy import signal, stats
from scipy.ndimage import median_filter

FS = 1000.0              # sampling rate (Hz): 1 sample = 1 ms
N = 10_000               # samples in spike_data.npz
W, K = 500, 4.5          # Assignment 2B detector: window, threshold factor
MASK = (2, 7)            # noise: drop samples [-2, +7] ms around each spike
WAVE = np.arange(8)      # mean spike waveform on lags [0, +7] ms
MEDIAN = 501             # Assignment 2A moving-median baseline (samples)
BANDS = (0, 5, 10, 20, 40, 80, 160, 320, 501)   # Hz; [0, 5) = drift band
MATCH = 2                # two spikes match if at most 2 ms apart
COPY_LIMIT = 0.3         # noise correlation above this = copied noise
SCHEMA_PENALTY = 0.10    # composite x 0.9 if the file needed repairs
MAX_NAN_FRACTION = 0.01  # up to 1 % NaN/Inf samples are interpolated
EXTENSIONS = (".npz", ".npy", ".csv", ".txt", ".mat")

# accepted names (lower case, without spaces, dashes or dots)
SIGNAL_KEYS = ("raw_data", "rawdata", "raw", "data", "signal", "sig",
               "x", "y", "v", "voltage", "trace", "recording", "lfp",
               "sim", "simulation", "simulated", "simulated_data",
               "sim_data", "simulated_raw_data", "raw_data_sim",
               "x_sim", "output")
TIME_KEYS = ("t", "time", "times", "timestamp", "timestamps", "time_s",
             "t_s", "tt", "ts", "time_ms", "t_ms")

# metric: (tier, tolerance, description)
METRICS = {
    "rmse": ("T1", 0.43, "RMSE of the two signals (a.u.)"),
    "drift_rmse": ("T1", 0.035, "RMSE of the moving medians (a.u.)"),
    "noise_sd": ("T2", 0.027, "relative error of the noise SD"),
    "noise_ks": ("T2", 0.021, "KS distance of the noise values"),
    "drift_band": ("T3", 0.21, "|log2| power ratio, 0-5 Hz"),
    "band_profile": ("T3", 0.19, "mean |log2| power ratio, 5-500 Hz"),
    "firing_rate": ("T4", 0.1, "firing-rate difference (Hz)"),
    "waveform_rmse": ("T4", 0.12, "RMSE of the mean waveforms (a.u.)"),
    "spike_times": ("T4", 0.023, "1 - F1 of spike times (+-2 ms)"),
    "isi_ks": ("T4", 0.22, "KS distance of the intervals"),
}
TIERS = {"T1": "Signal and drift", "T2": "Noise", "T3": "Spectrum",
         "T4": "Spikes"}
# compared at the same moments -> need the original spike times;
# left out of the "statistics-only" score
TIME_LOCKED = ("rmse", "spike_times")


class UnusableFile(Exception):
    """The file holds no signal that can be compared with the original."""


# ----------------------------------------------------------------------------
# Assignment 2B detector
# What: threshold = sliding mean + k * sliding SD (W = 500 samples); a spike
#       is the first sample of every run above the threshold.
# Why:  every team knows it, so all files are measured the same way. The
#       spikes are ~17 noise SDs high and >= 11 ms apart, so a more
#       elaborate detector would find the same 42 spikes.
# ----------------------------------------------------------------------------
def local_mean_sd(x, w=W):
    """Sliding mean and SD over w samples (cumulative sums, edge padding)."""
    padded = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    c1 = np.concatenate(([0.0], np.cumsum(padded)))
    c2 = np.concatenate(([0.0], np.cumsum(padded ** 2)))
    mean = (c1[w:] - c1[:-w]) / w
    sd = np.sqrt(np.maximum(0.0, (c2[w:] - c2[:-w]) / w - mean ** 2))
    return mean, sd


def detect_spikes(x, k=K):
    """Sample index of the first sample of every run above mean + k * SD."""
    mean, sd = local_mean_sd(x)
    above = x > mean + k * sd
    return np.flatnonzero(above & ~np.r_[False, above[:-1]])


# ----------------------------------------------------------------------------
# T0: reading and repairing other teams' files
# What: read the file (never unpickle), find the signal and the time
#       array, and repair the shape, dtype and missing values.
# Why:  the module ranks every team. A team with a good simulation but
#       a missing ``t`` array or a key called "data" must not crash the
#       ranking or be reported as a 0 without explanation. Repairs never
#       change the signal values, except interpolating a few NaNs.
# ----------------------------------------------------------------------------
def _norm(name):
    return name.lower().replace(" ", "").replace("-", "_").replace(".", "_")


def read_arrays(path):
    """All numeric arrays in a file, as {name: ndarray}."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return {"arr_0": np.load(path, allow_pickle=False)}
    if ext in (".csv", ".txt"):
        table = np.genfromtxt(path, delimiter="," if ext == ".csv" else None,
                              names=None, comments="#")
        if table.ndim == 2 and np.all(np.isnan(table[0])):
            table = table[1:]                    # header row
        return {"arr_0": table}
    if ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(path)
        return {k: v for k, v in mat.items() if not k.startswith("__")}
    with np.load(path, allow_pickle=False) as data:   # .npz (or no ext)
        return {key: data[key] for key in data.files}


def _columns(name, a):
    """Split one array into named 1D candidates."""
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        return {}
    a = np.squeeze(a)
    if a.ndim == 1:
        return {name: a}
    if a.ndim == 2 and min(a.shape) <= 4:        # (2, N), (N, 2), ...
        cols = a if a.shape[0] <= 4 else a.T
        return {f"{name}[{j}]": c for j, c in enumerate(cols)}
    return {}


def _is_time(a):
    """Monotonic array with a constant step (looks like timestamps)."""
    if a.size < 3 or not np.all(np.isfinite(a)):
        return False
    d = np.diff(a.astype(np.float64))
    return bool(d.min() > 0 and np.allclose(d, d[0], rtol=1e-3))


def _pick(cands, keys):
    lookup = {_norm(n.split("[")[0]): n for n in cands}
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def load_candidate(path):
    """Return (t, x, repairs, raw) for any reasonable candidate file.

    t, x: float64 arrays of shape (N,); repairs: list of strings;
    raw: the arrays as stored in the file (for the strict check).
    Raises UnusableFile if no comparable signal is found.
    """
    try:
        raw = read_arrays(path)
    except Exception as err:            # noqa: BLE001 (other teams' files)
        text = str(err)
        if "pickle" in text.lower():
            text = ("not a valid .npz/.npy file, or it stores Python "
                    "objects instead of numeric arrays (save with "
                    "np.savez(path, t=t, raw_data=x))")
        raise UnusableFile(f"cannot read the file: {text}") from err
    if not raw:
        raise UnusableFile("the file contains no arrays")

    cands = {}
    for name, a in raw.items():
        cands.update(_columns(name, a))
    if not cands:
        raise UnusableFile("no numeric 1D array in the file "
                           f"(keys: {sorted(raw)})")
    repairs = []

    # signal: a known name first, else the longest non-time array
    sig_name = _pick(cands, SIGNAL_KEYS)
    if sig_name is not None and _is_time(cands[sig_name]) \
            and _norm(sig_name.split("[")[0]) not in ("raw_data", "signal"):
        sig_name = None
    if sig_name is None:
        others = [n for n in cands if not _is_time(cands[n])]
        if not others:
            raise UnusableFile("only time-like arrays found "
                               f"(keys: {sorted(raw)})")
        sig_name = max(others, key=lambda n: (cands[n].size == N,
                                              cands[n].size))
    if sig_name != "raw_data":
        repairs.append(f"signal read from '{sig_name}' (not 'raw_data')")
    x = cands.pop(sig_name)
    stored = np.shape(raw.get(sig_name, x))
    if "[" not in sig_name and stored != x.shape:
        repairs.append(f"signal shape {stored} flattened to {x.shape}")

    # time: a known name first, else any timestamp-like array
    t_name = _pick(cands, TIME_KEYS)
    if t_name is None:
        times = [n for n in cands if _is_time(cands[n])]
        t_name = times[0] if times else None
    t = None if t_name is None else cands[t_name]
    if t is None:
        repairs.append("'t' missing: rebuilt as arange(N) / 1000 s")
    elif t_name != "t":
        repairs.append(f"time read from '{t_name}' (not 't')")

    # dtype
    if x.dtype != np.float64:
        repairs.append(f"raw_data cast from {x.dtype} to float64")
    x = x.astype(np.float64)

    # length
    if x.size < N:
        raise UnusableFile(f"signal has {x.size} samples, {N} are needed")
    if x.size > N:
        repairs.append(f"signal truncated from {x.size} to {N} samples")
        x = x[:N]

    # NaN / Inf
    bad = ~np.isfinite(x)
    if bad.any():
        if bad.mean() > MAX_NAN_FRACTION:
            raise UnusableFile(f"{bad.sum()} NaN/Inf samples "
                               f"(> {MAX_NAN_FRACTION:.0%})")
        idx = np.arange(N)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
        repairs.append(f"{bad.sum()} NaN/Inf samples interpolated")

    # time array: units and length (scoring is by sample index anyway)
    if t is not None:
        t = t.astype(np.float64)
        if t.size >= N and _is_time(t[:N]):
            step = float(np.median(np.diff(t[:N])))
            if np.isclose(step, 1.0, rtol=1e-3):
                repairs.append("t was in ms: converted to s")
                t = t / 1000.0
            elif not np.isclose(step, 1.0 / FS, rtol=1e-3):
                repairs.append(f"t has step {step:g} (not 1 ms): "
                               "rebuilt as arange(N) / 1000 s")
                t = None
        else:
            repairs.append("t unusable (length or not uniform): rebuilt")
            t = None
    t = np.arange(N) / FS if t is None else t[:N]
    return t, x, repairs, raw


def check_structure(data, t_ref):
    """Strict comparison with spike_data.npz: list of (check, passed)."""
    checks = [("keys are exactly t and raw_data",
               sorted(data) == ["raw_data", "t"])]
    for key in ("t", "raw_data"):
        if key not in data:
            checks.append((f"{key}: present", False))
            continue
        a = np.asarray(data[key])
        numeric = a.dtype.kind in "biuf"
        checks += [(f"{key}: shape (10000,)", a.shape == (N,)),
                   (f"{key}: dtype float64", a.dtype == np.float64),
                   (f"{key}: no NaN or Inf",
                    numeric and bool(np.all(np.isfinite(a))))]
    t = np.asarray(data.get("t", np.empty(0)))
    checks.append(("t equal to the original (dt = 1 ms)",
                   t.shape == t_ref.shape and t.dtype == np.float64
                   and np.allclose(t, t_ref, rtol=0, atol=1e-9)))
    return checks


def load(path):
    """Arrays of an .npz file (kept for backward compatibility)."""
    return read_arrays(path)


# ----------------------------------------------------------------------------
# Features and errors
# What: the same properties are measured in both signals: spikes (2B),
#       residual noise (signal - local mean, spike windows removed), the 2A
#       moving-median drift, band powers of the Welch spectrum and the mean
#       spike waveform. Each metric is one simple formula on these.
# Why:  each tier answers one question. T1: is the signal (and its drift)
#       in the right place? T2: is the noise the right size and shape?
#       T3: is the power at slow and fast frequencies right? T4: are the
#       spikes right (how many, which shape, when, how spaced)?
# ----------------------------------------------------------------------------
def features(x):
    """All properties of one signal."""
    n = x.size
    mean, _ = local_mean_sd(x)
    spikes = detect_spikes(x)
    keep = np.ones(n, dtype=bool)
    for i in spikes:
        keep[max(0, i - MASK[0]):i + MASK[1] + 1] = False
    resid = x - mean                                  # 2B residual
    freqs, psd = signal.welch(x - x.mean(), fs=FS, nperseg=1024,
                              detrend=False)
    bands = np.array([psd[(freqs >= lo) & (freqs < hi)].sum()
                      for lo, hi in zip(BANDS[:-1], BANDS[1:])])
    starts = spikes[spikes + WAVE[-1] < n]
    return {"x": x, "spikes": spikes, "keep": keep, "resid": resid,
            "noise": resid[keep], "bands": bands,
            "drift": median_filter(x, size=MEDIAN, mode="nearest"),
            "waves": resid[starts[:, None] + WAVE]}


def rms(d):
    return float(np.sqrt(np.mean(d ** 2)))


def ks(a, b):
    """Largest gap between two empirical CDFs (nan if a sample is empty)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(stats.ks_2samp(a, b, method="asymp").statistic)


def f1(a, b):
    """F1 of two spike trains; a pair matches if at most MATCH ms apart."""
    if len(a) + len(b) == 0:
        return 1.0
    free = np.ones(len(b), dtype=bool)
    hits = 0
    for time in a:
        dist = np.where(free, np.abs(b - time), MATCH + 1)
        if len(dist) and dist.min() <= MATCH:
            free[dist.argmin()] = False
            hits += 1
    return 2 * hits / (len(a) + len(b))


def mean_wave(f):
    return f["waves"].mean(axis=0) if len(f["waves"]) else np.nan


def errors(ref, cand):
    """Error of every metric: 0 = identical, larger = worse."""
    inner = slice(MEDIAN // 2, -(MEDIAN // 2))   # median is biased at ends
    duration = ref["x"].size / FS                # s
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(np.log2(cand["bands"] / ref["bands"]))
        sd_err = abs(cand["noise"].std() / ref["noise"].std() - 1) \
            if cand["noise"].size else float("nan")
    return {
        "rmse": rms(cand["x"] - ref["x"]),
        "drift_rmse": rms(cand["drift"][inner] - ref["drift"][inner]),
        "noise_sd": float(sd_err),
        "noise_ks": ks(cand["noise"], ref["noise"]),
        "drift_band": float(ratio[0]),
        "band_profile": float(ratio[1:].mean()),
        "firing_rate": abs(len(cand["spikes"]) - len(ref["spikes"]))
        / duration,
        "waveform_rmse": rms(mean_wave(cand) - mean_wave(ref)),
        "spike_times": 1 - f1(cand["spikes"], ref["spikes"]),
        "isi_ks": ks(np.diff(cand["spikes"]), np.diff(ref["spikes"])),
    }


# ----------------------------------------------------------------------------
# Scores
# What: error -> score with min(1, tolerance / error), then means.
# Why:  one rule for every metric, no hand-tuned weights, and an error that
#       a perfect simulation could also show is never punished.
# ----------------------------------------------------------------------------
def score(error, tolerance):
    if not np.isfinite(error):          # e.g. no spikes detected
        return 0.0
    return 1.0 if error <= tolerance else tolerance / error


def composite(scores, skip=()):
    """100 * mean over tiers of the mean metric score in each tier."""
    tiers = {tier: float(np.mean([s for m, s in scores.items()
                                  if METRICS[m][0] == tier
                                  and m not in skip]))
             for tier in TIERS}
    return 100 * float(np.mean(list(tiers.values()))), tiers


def compare(ref, x):
    """Errors, scores and composite of signal x against features ref."""
    cand = features(x)
    err = errors(ref, cand)
    scores = {m: score(err[m], METRICS[m][1]) for m in METRICS}
    total, tiers = composite(scores)
    stats_only, _ = composite(scores, skip=TIME_LOCKED)
    # copy check: correlation of the two noise traces outside all spikes
    both = ref["keep"] & cand["keep"]
    with np.errstate(divide="ignore", invalid="ignore"):
        noise_r = float(np.corrcoef(ref["resid"][both],
                                    cand["resid"][both])[0, 1])
    return {"errors": err, "scores": scores, "tiers": tiers,
            "composite": total, "statistics_only": stats_only,
            "noise_r": noise_r, "n_spikes": len(cand["spikes"])}


def evaluate(path, reference="spike_data.npz", ref=None, strict=False):
    """Full evaluation of one candidate file (dict; composite 0-100).

    Never raises for a bad candidate: problems end up in report["notes"].
    """
    original = load_candidate(reference)
    t_ref = original[0]
    if ref is None:
        ref = features(original[1])
    report = {"checks": [], "repairs": [], "composite": 0.0,
              "statistics_only": 0.0, "unpenalised": 0.0, "notes": []}
    try:
        _, x, repairs, raw = load_candidate(path)
    except UnusableFile as err:
        report["notes"].append(f"unusable file: {err}; score 0")
        return report
    report["checks"] = check_structure(raw, t_ref)
    report["repairs"] = repairs
    compliant = all(ok for _, ok in report["checks"])
    if strict and not compliant:
        report["notes"].append("structure check failed (--strict): score 0")
        return report
    try:
        report.update(compare(ref, x))
    except Exception as err:            # noqa: BLE001 (never stop a ranking)
        report["notes"].append(f"evaluation failed ({err}); score 0")
        return report
    report["unpenalised"] = report["composite"]
    if not compliant:
        factor = 1.0 - SCHEMA_PENALTY
        report["composite"] *= factor
        report["statistics_only"] *= factor
        report["notes"].append(f"structure differs from spike_data.npz: "
                               f"score x {factor:.2f} after repairs")
    if report["noise_r"] > COPY_LIMIT:
        report["notes"].append(f"noise correlates with the original noise "
                               f"(r = {report['noise_r']:.2f}): copied?")
    return report


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def show(name, rep):
    print(f"=== {name} ===")
    for check, ok in rep["checks"]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    for repair in rep["repairs"]:
        print(f"  [FIX ] {repair}")
    for tier, title in TIERS.items():
        if "tiers" not in rep:
            break
        print(f"{tier} {title}: {100 * rep['tiers'][tier]:.1f}")
        for m, (metric_tier, tol, text) in METRICS.items():
            if metric_tier == tier:
                print(f"    {m:<14} {text:<36} error "
                      f"{rep['errors'][m]:8.4f}  tol {tol:<6} "
                      f"score {rep['scores'][m]:.2f}")
    if "tiers" in rep:
        print(f"Spikes found: {rep['n_spikes']}; noise correlation with "
              f"the original: {rep['noise_r']:.3f}")
    print(f"Statistics-only score: {rep['statistics_only']:.1f}")
    print(f"COMPOSITE SCORE: {rep['composite']:.1f} / 100")
    for note in rep["notes"]:
        print(f"  NOTE: {note}")
    print()


def expand(paths, reference):
    """Files to score: folders are expanded, the reference is skipped."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(f for f in glob.glob(os.path.join(p, "*"))
                           if f.lower().endswith(EXTENSIONS))
            files += found
        else:
            files.append(p)
    ref_abs = os.path.abspath(reference)
    return [f for f in files if os.path.abspath(f) != ref_abs]


def main():
    parser = argparse.ArgumentParser(description="Score simulations of "
                                     "spike_data.npz from 0 to 100.")
    parser.add_argument("candidates", nargs="+",
                        help="files (.npz/.npy/.csv/.txt/.mat) or folders")
    parser.add_argument("--reference", default="spike_data.npz")
    parser.add_argument("--strict", action="store_true",
                        help="score 0 for any file that differs in "
                        "structure from spike_data.npz (no repairs)")
    parser.add_argument("--csv", help="also write the ranking to this file")
    args = parser.parse_args()
    if not os.path.exists(args.reference):
        parser.error(f"{args.reference} not found: put spike_data.npz in "
                     f"this folder or use --reference")
    ref = features(load_candidate(args.reference)[1])
    files = expand(args.candidates, args.reference)
    if not files:
        parser.error("no candidate files found")
    reports = []
    for path in files:
        if not os.path.exists(path):
            rep = {"checks": [], "repairs": [], "composite": 0.0,
                   "statistics_only": 0.0, "unpenalised": 0.0,
                   "notes": ["file not found; score 0"]}
        else:
            rep = evaluate(path, args.reference, ref, strict=args.strict)
        reports.append(rep)
        show(path, rep)
    ranked = sorted(zip(files, reports),
                    key=lambda item: -item[1]["composite"])
    if len(reports) > 1:
        print("=== Ranking ===")
        for rank, (name, rep) in enumerate(ranked, start=1):
            flag = f"  [{'; '.join(rep['notes'])}]" if rep["notes"] else ""
            print(f"{rank:>2}. {rep['composite']:5.1f}  (statistics-only "
                  f"{rep['statistics_only']:5.1f})  {name}{flag}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["rank", "file", "composite", "statistics_only",
                          "before_penalty", "n_repairs", "notes"])
            for rank, (name, rep) in enumerate(ranked, start=1):
                out.writerow([rank, name, f"{rep['composite']:.2f}",
                              f"{rep['statistics_only']:.2f}",
                              f"{rep['unpenalised']:.2f}",
                              len(rep["repairs"]), "; ".join(rep["notes"])])
        print(f"Ranking written to {args.csv}")


if __name__ == "__main__":
    main()
