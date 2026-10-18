# Real-data quick pipeline

This reproducible runner is a smoke experiment, not the formal independent-cohort analysis. It uses only the P6-frozen people and P2–P5-qualified windows; it does not load historical models/results. The raw data root is supplied explicitly, and every output directory must be new. Full execution is not enabled by this runner.

## Commands

Run from this scientific-code directory with the existing Python 3.12 CPU environment:

```powershell
python -B scripts/run_quick.py --data-root "D:\大学学习\论文\论文项目\URL" --task af_afl --model classical --preset quick --output "D:\大学学习\论文\投稿论文\04_Results\my_new_quick\af_afl_classical_quick"
```

Supported tasks are `af_afl`, `hr_reliability`, `stress`, and `continuous_monitoring`; supported model families are `classical` and `cnn`. Run each task/family in a different new directory. In PyCharm select `scripts/run_quick.py`, the above arguments, the project-code directory as working directory, and `论文项目/.venv/Scripts/python.exe` as interpreter. Use `PYTHONDONTWRITEBYTECODE=1` to keep compiled caches out of submission materials. Console and inspection logs belong outside the submission folder.

## Scientific definitions

Classification smoke windows are hash-ranked by person/record/time before model predictions; 48 person-time windows are retained per selected person, keeping both Sport wrists together. VitalDB training uses one person per frozen class/CV-fold cell, and policy, conformal and evaluation use two per class. DaLiA/WESAD retain their nine training people and three policy/conformal people. External sources retain three ranked evaluation IDs, or two for each supplementary stress protocol. These reduced subsets do not satisfy or replace the full planned validation sizes. Roles are never reallocated.

Monitoring clips use the longest already-qualified episode per person, with onset ±180 seconds rounded to complete 10-second blocks. All ten training people and three people per other role are retained. Annotation-based selection is used only to exercise the smoke pipeline, not to estimate monitoring efficacy. It occurs before predictions. Event censoring at clip boundaries and observed follow-up are saved; reference labels do not enter the runtime controller or trigger its resets.

CNN smoke runs cap each of the three training-person CV folds at two epochs and refit the median selected epoch. This does not change the full 25-epoch frozen setting. Secondary EDA scaling is fitted inside the respective training-person folds. All models are saved and reloaded before inference. Policies are fitted only on policy people, then strategy/budget-specific conformal thresholds on separate conformal people. Evaluation labels cannot change thresholds, budgets or people.

Primary probability/quality and budget alone determine adaptive secondary calls. The secondary computation is executed lazily and recorded. HR reference correction remains an explicitly offline oracle, and error is computed using the final reported HR. Do not interpret its normalized cost as sensor energy or its CPU time as measured ECG collection latency.

## Outputs

Each scientific result directory contains `run_config.json` with code/protocol/source-index hashes, environment and exact command; `selection_before_training.json`; model checkpoints and fit-window provenance; policy/profile metadata; executed calibration scores and finite/infinite quantiles; `predictions.csv`; person-level `metrics.json`/CSV; figures and exact plotted source tables; and `result_manifest.json` with artifact hashes. Monitoring additionally saves every tick, scoring intervals, event matches, gaps, alarm spans and infeasible-target indicators. Undefined selective risk, single-class AUROC and absent detections remain null, never a successful zero.

`CPU_milliseconds` measures required processing and actually called tools inside the completed window. Disk IO is outside this timing; real sensor collection latency is not measured. Formal person-cluster inference, sensitivity analyses and final manuscript figures are not produced by quick. Third-party raw signals are not included in this result package.
