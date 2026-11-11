# Upload instructions: python_reverse_engineering

Repository: https://github.com/akarshaniRamanayake/python_reverse_engineering

## What was wrong

The repository contains only `README.md`, `Reverse_Engineering_Spikes.ipynb` and `.DS_Store`.
The output files that the README lists were never uploaded, which causes two errors:

- **"No reconstruction file in repo"**: the results file `simulated_spike_data.npz` is missing.
- **"File not found"**: `testing_module.py` and the reference file `spike_data.npz` are missing, so other teams cannot run the testing module.

Nothing in the repository is wrong. Files are only missing.

## Final file list (8 files)

| File | Action |
|:--|:--|
| `simulated_spike_data.npz` | **Add**: results file (the submission) |
| `testing_module.py` | **Add**: testing module (fixed version that handles missing or renamed arrays) |
| `spike_data.npz` | **Add**: original recording, used as the reference |
| `simulated_spike_data_stochastic.npz` | **Add**: stochastic simulation |
| `model_parameters.json` | **Add**: fitted parameters and model checks |
| `.gitignore` | **Add**: keeps `.DS_Store` and temporary files out of the repository |
| `README.md` | **Replace**: updated description of the testing module |
| `Reverse_Engineering_Spikes.ipynb` | **Replace**: notebook now writes the fixed testing module |
| `.DS_Store` | **Delete**: macOS system file, not part of the project |

## Steps

1. Open the repository on GitHub.
2. Click **Add file → Upload files**.
3. Drag in the 8 files listed above.
   - Use only the files from this delivery, not older copies with the same names.
   - Check that no file name has changed while downloading. For example, `testing_module (1).py` must be renamed to `testing_module.py`.
   - Do **not** upload the `.zip` file. It is a backup only.
4. Write a commit message, for example: `Add results file, testing module and reference data`. Then click **Commit changes**.
   GitHub replaces `README.md` and the notebook automatically because they have the same names.
5. Open `.DS_Store` in the repository, click the **trash icon** (Delete file), and commit.

## Check after uploading

The repository should contain exactly these 8 files:

```
.gitignore
README.md
Reverse_Engineering_Spikes.ipynb
model_parameters.json
simulated_spike_data.npz
simulated_spike_data_stochastic.npz
spike_data.npz
testing_module.py
```

Test command (after cloning or downloading the repository):

```bash
python testing_module.py simulated_spike_data.npz simulated_spike_data_stochastic.npz
```

Expected result:

```
 1. 100.0  (statistics-only 100.0)  simulated_spike_data.npz
 2.  88.0  (statistics-only  99.7)  simulated_spike_data_stochastic.npz
```

To score other teams' files:

```bash
python testing_module.py team_A.npz team_B.npz
python testing_module.py teams_folder/ --csv ranking.csv
```
