# Frozen research reproduction

The manuscript uses `P9_full_20261011_v1` and `P10_statistics_20261011_v1`,
with protocol `PM_P6_20261010_v1` and seed 20260921. Older exploratory runs and
quick checkpoints are not the manuscript's main evidence.

The surrounding submission-material directory contains:

```text
02_Code/physiological_tool_selection/     algorithms, tests, allocations, full runner
02_Code/P10_statistical_analysis/        person-level statistics and sensitivity code
03_Data/qualified_indices/              frozen qualification indices, not raw signals
04_Results/P9_full_20261011_v1/          eight full runs, models and predictions
04_Results/P10_statistics_20261011_v1/   statistics, refitted sensitivity models and figures
```

Each P9 result folder is named `<task>_<model>_full`, for tasks `af_afl`,
`hr_reliability`, `stress`, `continuous_monitoring`, and models `classical`, `cnn`.
The two heavy monitoring context models are fitted separately at 10 and 30 s.
Source records and arms belonging to one participant inherit the same role.

Use Python 3.12.4 with the package versions supplied by the scientific package's
`requirements.txt` and actual saved environment/profiling files. The original
CPU build was PyTorch 2.14.0+cpu. The experiment runner caps PyTorch/BLAS threads
at four. Do not upgrade dependencies while trying to reproduce recorded timing.

From `02_Code/physiological_tool_selection`, a full-run example is:

```powershell
python -B scripts/run_full.py --data-root "<official-local-raw-data-root>" --task af_afl --model classical --preset full --output "<new-result-root>/af_afl_classical_full"
```

Repeat with the other three task names and `--model cnn`, in series, with NEW
output directories. Do not rerun into preserved results. Use the frozen
person/record/configuration files; do not substitute quick models or silently
reduce participants, windows, event duration, budgets or fit epochs. The runner
has no full-run participant/epoch override. The old package `README.md` is the
initial implementation snapshot; `README_P9.md` documents the full runner.

For statistics, follow `02_Code/P10_statistical_analysis/README.md` in order:
`run_statistics.py`, `run_diagnostics.py`, `run_sensitivity.py`,
`run_interpretation.py`, `run_monitor_constraints.py`, then `make_manifest.py`.
Use the newly reproduced P9 root and a NEW analysis root. The sensitivity stage
requires the source data and a separate scratch directory for temporary feature
caches. It includes six newly fitted HR target/alignment models, not re-labeled
old predictions. Unchanged models reuse stored probabilities and original
measured timings. Do not run other workloads while collecting new latency.

`result_manifest.json` and `reproducibility.json` in the results identify
file sizes/SHA-256 and exact code/input versions. The manuscript-source package
supplies the 26 final P10 CSVs and two additional aggregate demographic CSVs.
All third-party raw recordings and participant-level source clinical metadata
must be obtained separately; aggregate demographic summaries do not add new
model covariates or alter the scientific allocations.

The HR reference-correction path is an offline information upper bound, not a
validated wrist ECG device. The policy's 5% negative-positive-decision cap is
not an evaluation FAR bound. Report fractions and undefined-risk support must
be retained along with any loss improvement. Cluster intervals condition on
the fitted pipeline and do not establish clinical efficacy or deployment safety.
Public repository and research-code release licence await author confirmation.
