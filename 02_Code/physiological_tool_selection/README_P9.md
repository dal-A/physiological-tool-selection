# Full independent-cohort experiments (P6 protocol)

This version adds full runners without modifying the sealed P7/P8 implementation
or smoke outputs. Python 3.12, CPU, and the pinned package requirements are used.
Raw third-party signals are acquired separately from their official sources.
Qualified indices and frozen allocations are under `03_Data` and `config`.

From this package directory:

```powershell
python -B scripts/run_full.py --data-root "D:/大学学习/论文/论文项目/URL" --task af_afl --model classical --preset full --output "D:/大学学习/论文/投稿论文/04_Results/NEW_RUN/af_afl_classical_full"
```

Repeat with `--model cnn` and tasks `hr_reliability`, `stress`, and
`continuous_monitoring`. Each output must be a NEW directory. There is no epoch,
participant, window or checkpoint override. CNN uses training-person-only
three-fold epoch selection (cap 25, patience 5), then fresh full-training refit.
Run experiments serially so policy latency profiles are not measured under
competition from another fit. Set the working directory to this package in
PyCharm and use the same `.venv` interpreter as the original project.

All frozen eligible people/windows are retained, including technical no-report
cases. Sport's two wrists form paired person-time observations, not independent
participants. HR sources and backup stress protocols are reported separately.
Monitoring streams every complete ten-second block in all frozen records;
unknown rhythms do not reset the runtime. Native records are scored separately
and then aggregated within each person. Both the ten- and thirty-second heavy
models are trained independently.

Outputs include frozen input/code SHA, exact contributions and training rows,
models/scalers/CV epochs/sampled draws, policy states/bins/profiles, separate
executed-tool conformal scores by strategy/budget/duration, every prediction and
actual call/CPU cost, person metrics, event matches and exposure/censoring, paired
primary contrasts with 10,000 fixed-pipeline cluster bootstrap replicates, and
descriptive full figures with plotted source tables. Null risk or delay remains
null when its denominator is zero. Files are not independently accepted simply
because a command exits zero; completed matrices require independent review.

The later statistical stage adds the joint five-contrast classical inference,
locked sensitivity analyses and final publication figures. CNN contrasts are
exploratory. A split conformal set under temporal dependence/domain shift is
empirical evidence, not a patient-level or clinical safety guarantee. HR reference
correction is an OFFLINE upper bound: reference acquisition latency is unknown.
The AF/AFL task is not pure AF, and VitalDB finger signals are not wrist-device
external validation. Zero alarms or zero reported errors alone are not success.
