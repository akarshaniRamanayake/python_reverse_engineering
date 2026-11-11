# Reverse Engineering a Noisy Neural Recording

**Author:** Abdulhamit Gökçe (U38891823)
**Course project:** follow-up to Assignments 2A and 2B, *"Finding Spikes in Noisy Neural Data"*

This project works out how the recording `spike_data.npz` was generated. It then builds a new recording with the same structure and writes a testing module that measures how close any simulation is to the original. The testing module will be used in class to rank the teams' simulations.

---

## Contents

- [Project goals](#project-goals)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [The data](#the-data)
- [Methodology](#methodology)
- [Part 1 — Reverse engineering](#part-1--reverse-engineering)
- [Part 2 — Testing module](#part-2--testing-module)
- [Part 3 — Validation](#part-3--validation)
- [Main results](#main-results)
- [Assumptions and limitations](#assumptions-and-limitations)

---

## Project goals

The assignment has two parts:

1. **Reverse engineering.** Find out how the recording was made and produce a simulated recording, a *results file* with **exactly the same structure** as the original.
2. **Testing module.** Write a module that measures **how close a simulation is to the original**. The class will use it to decide which team got the closest result.

---

## Project structure

| File | What it does |
|:--|:--|
| `Reverse_Engineering_Spikes.ipynb` | The whole project in one Google Colab notebook: analysis, simulation, testing module, validation and figures. Opens with all outputs already shown. |
| `spike_data.npz` | The original recording (input). |
| `simulated_spike_data.npz` | **Results file** (deterministic mode, the submission). Same keys, data type and shape as the original. |
| `simulated_spike_data_stochastic.npz` | The same model, but with 42 new random spike times. |
| `model_parameters.json` | All fitted parameters and the results of every model check. |
| `testing_module.py` | **Testing module.** Scores any candidate file from 0 to 100. Can be used from the command line or imported. |

When the notebook runs, it also re-creates the four output files (the two simulated files, `model_parameters.json` and `testing_module.py`). In addition it writes:

- a `controls/` folder, which holds the test files used in Part 3;
- `reverse_engineering_results.zip`, which contains the four output files.

---

## How to run

### Requirements

- Python 3
- `numpy`, `scipy`, `matplotlib` (all three are already installed in Google Colab)

Outside Colab, install them with:

```bash
pip install numpy scipy matplotlib
```

### Option A — Google Colab (recommended)

1. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → choose `Reverse_Engineering_Spikes.ipynb`.
2. Click **Runtime → Run all**.
3. When the upload box appears under *Step 1*, choose `spike_data.npz`.
   You can instead drag `spike_data.npz` into Colab's **Files** panel before pressing *Run all*; then no upload box appears.
4. The last cell downloads `reverse_engineering_results.zip` with all result files.

The whole notebook runs in well under a minute. All random numbers use fixed seeds, so every run gives exactly the same numbers.

### Option B — Local Jupyter

Put `spike_data.npz` in the same folder as the notebook and run all cells.

### Using the testing module alone

From the command line (with `spike_data.npz` in the same folder):

```bash
python testing_module.py simulated_spike_data.npz
python testing_module.py team_A.npz team_B.npz --reference path/to/spike_data.npz
python testing_module.py teams_folder/ --csv ranking.csv     # every file in a folder
python testing_module.py team_A.npz --strict                 # no repairs: wrong structure = 0
```

When several files are given, the module also prints a ranking. Accepted formats are `.npz`, `.npy`, `.csv`, `.txt` and `.mat`. A missing or unreadable file gets 0 with the reason printed; it never stops the ranking of the other files.

From Python:

```python
from testing_module import evaluate, show

report = evaluate("simulated_spike_data.npz", reference="spike_data.npz")
print(report["composite"])        # score from 0 to 100
show("my file", report)           # full table of checks and metrics
```

In the notebook, other teams' files can be scored in **Part 4** by listing them in `OTHER_TEAM_FILES`.

---

## The data

`spike_data.npz` contains two 1-D `float64` arrays, each of length 10,000:

- `t`: time in seconds, from 0.000 to 9.999 s (step 1 ms, sampling rate 1000 Hz, duration 10 s)
- `raw_data`: the signal in arbitrary units (a.u.)

Global statistics: min −1.691, max 6.202, mean 0.077, median 0.035, SD 0.734 a.u.

---

## Methodology

I describe every sample of the recording as the sum of three parts:

```
x(t) = b(t)  +  s(t)  +  η(t)
       drift    spikes   noise
```

- **Drift `b(t)`:** a slow wave of the baseline.
- **Spikes `s(t)`:** every spike jumps up by the same height `A` and then decays exponentially with time constant `τ`.
- **Noise `η(t)`:** white Gaussian noise with standard deviation `σ`.

The main steps are:

1. Find the spike times with the Assignment 2B detector.
2. Fit the drift and the spike height by least squares, trying many decay times `τ`.
3. Take the noise as whatever the model leaves unexplained.
4. Check every assumption against a simple alternative, or against what noise alone would produce.
5. Simulate a new recording and save it in the original format.
6. Build a testing module and check that it ranks good and bad simulations correctly.

---

## Part 1 — Reverse engineering

### 1. Spike detection (Assignment 2B detector)

- For each sample I compute a local mean and a local SD over a sliding window of **W = 500 samples**.
- Threshold = local mean + **k = 4.5** × local SD.
- A spike is the **first sample** of each run above the threshold.

This finds **42 spikes** (firing rate **4.20 Hz**), at exactly the same indices as in Assignment 2B. The shortest interval between two spikes is 11 ms.

### 2. Model fit

The drift is modelled as one sine cycle over the recording plus a straight line:

```
b(t) = a · sin(2πt / 10 s)  +  c · (t − 5 s)
```

For a fixed decay time `τ`, the model is a simple linear regression in `a`, `c` and `A`, so `numpy.linalg.lstsq` gives the best values directly. I repeat this for `τ` = 1.0, 1.1, …, 8.0 ms and keep the `τ` with the smallest error.

| Parameter | Estimate |
|:--|:--|
| Drift amplitude `a` | 0.535 ± 0.007 a.u. |
| Drift trend `c` | −0.0418 ± 0.0017 a.u./s |
| Spike height `A` | 5.012 ± 0.032 a.u. |
| Decay time `τ` | 3.0 ms |
| Noise SD `σ` | 0.2985 a.u. |

The spikes are about 16.8 times larger than the noise SD. The fitted drift peaks at +0.644 a.u. (2.30 s) and has its minimum at −0.644 a.u. (7.70 s).

### 3. Checking the assumptions

| Question | Test | Result |
|:--|:--|:--|
| Is sine + trend the right drift? | BIC of four drift models (lower = better) | Sine + trend has the lowest BIC (−24,146.5). Without the trend, the BIC is about 611 higher. |
| Do all spikes have the same height? | Fit one height per spike and compare the spread with the noise-only spread | SD 0.205 vs. 0.208 expected from noise, so one height is enough. |
| Is the decay exponential? | Average spike shape vs. `exp(−l/3)` | Largest difference 0.018. |
| Are the small "rebounds" in the tail real? | Count rises at +2, +3 and +4 ms vs. the number expected by chance | 7 observed vs. 6.6 expected in total, so they are noise. |
| Is the noise white and Gaussian? | Skewness, excess kurtosis, lag-1 autocorrelation | 0.021, −0.043, −0.020, all close to 0. |
| Is sample 1014 an outlier? | Chance that the largest of 10,000 Gaussian samples is this big | p = 0.15, so it is normal noise. |
| Is the spike train Poisson-like? | Count dispersion and ISI coefficient of variation | 1.26 (χ² test p = 0.25) and 1.07, both close to 1. |

**Note on the noise level:** the 2B method (signal minus local mean, with the spike windows removed) measures 0.303 a.u. on the original. The generator's noise SD is 0.2985 a.u. The difference comes from the estimator, not from the data.

### 4. Simulation

- **Deterministic mode (submission, seed 2026):** the same 42 spike times, the fitted drift and spike shape, and **new** Gaussian noise. Saved as `simulated_spike_data.npz`.
- **Stochastic mode (seed 2027):** the same model, but with 42 **new random** spike times. They are drawn uniformly and redrawn until no two spikes are closer than a 6 ms dead time. Saved as `simulated_spike_data_stochastic.npz`.

Both files are checked after saving:

- the keys are exactly `t` and `raw_data`;
- both arrays are `float64` with shape `(10000,)`;
- there are no NaN or Inf values;
- the time axis is identical to the original.

| File | Min | Max | Mean | Median | SD | Spikes |
|:--|--:|--:|--:|--:|--:|--:|
| Original | −1.691 | 6.202 | 0.077 | 0.035 | 0.734 | 42 |
| Deterministic | −1.788 | 6.082 | 0.074 | 0.036 | 0.731 | 42 |
| Stochastic | −1.637 | 6.081 | 0.077 | 0.047 | 0.718 | 42 |

---

## Part 2 — Testing module

### Why not just RMSE?

The simulation has its own random noise, so even a **perfect** simulation has an RMSE of about 0.42 against the original. A smooth curve with **no noise at all** would get a *better* RMSE (about 0.30) even though it looks nothing like a real recording. So the module compares several properties of the signals, not only the sample-by-sample difference.

### How a file is scored

**Step 1 — Loading and structure check.** The original has:

- exactly the keys `t` and `raw_data`;
- `float64` arrays of shape `(10000,)`;
- no NaN or Inf values;
- the same time axis as the original.

Every check is reported as PASS/FAIL. Other teams' files often differ from this structure, so the loader repairs what it can without changing the signal values, and lists every repair as `[FIX]`:

- `t` missing → rebuilt as `arange(10000) / 1000` s; `t` in ms → converted to s;
- other key names (`data`, `signal`, `time`, `x`/`y`, `arr_0`/`arr_1`, a stacked `(2, 10000)` array) → the signal and time arrays are found by name or, failing that, by shape (the time array is the one with a constant step);
- `float32` → `float64`; shape `(1, 10000)` → `(10000,)`; more than 10,000 samples → truncated;
- up to 1 % NaN/Inf samples → linear interpolation.

A file that needed repairs is scored, but its composite is multiplied by **0.9**, so the required structure still matters. With `--strict`, any failed check gives **0**. Only a file without a usable 10,000-sample signal (too short, only a time array, Python objects, not a data file) gets **0**, with the reason printed. Files are opened with `allow_pickle=False`, so no code inside a file can run.

**Step 2 — Ten metrics in four groups (tiers).** The same properties are measured in both signals:

| Tier | Metric | What it compares | Tolerance |
|:--|:--|:--|:--:|
| T1 Signal & drift | `rmse` | Sample-by-sample difference (RMSE) | 0.43 |
| | `drift_rmse` | 501-sample moving-median baselines | 0.035 |
| T2 Noise | `noise_sd` | Relative difference of the noise SD | 0.027 |
| | `noise_ks` | Noise value distributions (KS distance) | 0.021 |
| T3 Spectrum | `drift_band` | Welch power below 5 Hz (log2 ratio) | 0.21 |
| | `band_profile` | Welch power in 7 bands from 5 to 500 Hz (mean log2 ratio) | 0.19 |
| T4 Spikes | `firing_rate` | Difference in firing rate (Hz) | 0.1 |
| | `waveform_rmse` | Mean spike waveform, 0 to +7 ms | 0.12 |
| | `spike_times` | 1 − F1 of spike times (match within ±2 ms) | 0.023 |
| | `isi_ks` | Inter-spike interval distributions (KS distance) | 0.22 |

The noise is measured as in Assignment 2B: the signal minus the local mean, with the samples from −2 to +7 ms around each spike removed. Spikes are found with the same 2B detector (W = 500, k = 4.5).

**Step 3 — From errors to a score.** Each metric gets:

```
score = min(1, tolerance / error)
```

The **tolerance** is the error that two runs of a perfect generator show by chance (95th percentile over 200 pairs). An error at chance level gets full marks, twice the chance level gets 0.5, and so on.

- Tier score = mean of its metric scores.
- **Composite score = 100 × mean of the four tier scores.**

**Extra outputs:**

- **Statistics-only score:** the same score without the two metrics that need the exact original spike times (`rmse`, `spike_times`). It shows whether a model with its own random spike times gets the statistics right.
- **Copy check:** if the candidate's noise correlates with the original's noise above 0.3, the file is flagged as "copied?".

The module uses no random numbers, so the same file always gets the same score.

---

## Part 3 — Validation

- **Tolerance check:** the notebook repeats the calibration with new seeds. The recomputed values are close to the fixed values in the module (for example 0.427 vs. 0.43 for `rmse`). The small differences come only from the random seeds.
- **Robustness:** 100 noise seeds per mode.
- **Figures** (shown inside the notebook):
  1. Full traces with the moving median and the fitted drift, plus a close-up of 2250–2500 ms.
  2. Average spike waveform: normalised waveforms with `exp(−l/3)`, and the mean ± SD for all three files.
  3. Residual noise histogram, normal Q–Q plot and Welch spectra.
  4. Number of detected spikes vs. threshold factor `k` (3.0–6.5), with 5–95% bands over 100 seeds.
  5. Scores of the control files.
- **Control files:** 16 files with one deliberate change each, plus one file saved as `float32` (wrong structure). They test whether the module ranks correctly.

---

## Main results

### Scores of the two simulation files

| File | Composite | Statistics-only |
|:--|:--:|:--:|
| `simulated_spike_data.npz` (deterministic, submission) | **100.0** | 100.0 |
| `simulated_spike_data_stochastic.npz` | **88.0** | 99.7 |

- Every error of the deterministic file is below its chance level. Its noise is new: the correlation with the original noise is only 0.046.
- The stochastic file loses points only on the two metrics that need the original spike times.
- Over 100 seeds: deterministic **100.0 ± 0.0**; stochastic **88.0 ± 1.5** (statistics-only 99.6 ± 1.7).
- The range of `k` that gives exactly 42 spikes: original 4.3–5.4, deterministic file 4.0–5.4, stochastic file 3.5–5.2.

### Control files (best first)

| Candidate | Score | Statistics-only | Flag |
|:--|--:|--:|:--|
| Copy of the original | 100.0 | 100.0 | copied |
| Original + new noise N(0, 0.05²) | 100.0 | 100.0 | copied |
| **Deterministic file (submission)** | **100.0** | **100.0** | – |
| Spike times shifted by +1 ms | 97.3 | 100.0 | – |
| Original + new noise N(0, 0.1²) | 93.7 | 93.7 | copied |
| Deterministic file saved as `float32` | 90.0 | 90.0 | repaired, × 0.9 |
| **Stochastic file** | **88.0** | **99.7** | – |
| Drift without linear trend | 82.5 | 73.8 | – |
| Spike amplitude × 0.7 | 78.9 | 77.6 | – |
| Spike decay τ = 1.5 ms | 77.9 | 76.8 | – |
| No drift | 72.5 | 65.0 | – |
| Noise SD × 1.5 | 69.6 | 72.1 | – |
| Noise-free model (drift + spikes) | 68.3 | 68.3 | – |
| Double firing rate (84 spikes) | 59.3 | 67.7 | – |
| No spikes | 49.9 | 53.8 | – |
| Shuffled original | 17.1 | 14.8 | – |
| White noise (mean, SD of original) | 10.5 | 6.1 | – |

**What this shows:**

- Both correct models rank above every file with a wrong drift, spike shape, noise level or firing rate.
- A smooth, noise-free model is **not** rewarded (68.3), because the noise tier catches it.
- Copies of the original score high but are **flagged**.
- A correct simulation saved with the wrong data type is repaired and scored, but loses 10 % (90.0).

### Summary of findings

The recording is the sum of:

- a slow drift: one 0.1 Hz sine cycle of 0.535 a.u. plus a trend of −0.042 a.u./s;
- 42 identical spikes of 5.01 a.u. with a 3.0 ms exponential decay;
- white Gaussian noise with SD 0.2985 a.u.

The tail "rebounds", the spread in spike heights and the large value at sample 1014 are all explained by noise.

---

## Assumptions and limitations

- **One recording.** All parameters come from a single 10-second recording. The drift model fits well inside these 10 s but should not be extrapolated beyond them.
- **Simplified spikes.** The spikes are positive, rise within one sample (1 ms) and decay exponentially. Real extracellular spikes are biphasic and much faster. The model reproduces this file, not real biophysics.
- **Spike train.** The stochastic mode keeps exactly 42 spikes and uses a Poisson process with a 6 ms dead time. Bursts or adaptation are not modelled.
- **Tolerances.** The tolerances come from my own fitted model, so they reflect its noise level and spike count.
- **Ties.** All correct deterministic reconstructions score 100, so they tie. Differences below about 4 points between stochastic submissions are within chance.
- **Detector.** The spike metrics only see the spikes that the 2B detector finds. A missed spike counts as noise.
- **Copies.** A copy of the original with a little extra noise still scores near 100. Only the copy flag exposes it, so flagged files should be checked by hand.
- **Environment.** The outputs shown in the notebook were produced with Python 3.13, NumPy 2.5 and Matplotlib 3.11. The code uses only standard functions of these libraries.
