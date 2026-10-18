# Person-level statistical analysis and sensitivity experiments

This package reproduces the P10 analyses of the frozen `P9_full_20261011_v1`
experiments. It does not replace, overwrite or tune the P9 experiment outputs.
Use the environment and dependency versions recorded in P9. No cloud API or new
raw data download is required. Derived tables and plots are research materials;
internal inspection reports and temporary feature caches are not submission data.

## Reproduction

From this directory, with the existing scientific Python environment:

```powershell
python -B run_statistics.py --p9-root ../../04_Results/P9_full_20261011_v1 --output ../../04_Results/P10_statistics_new
python -B run_diagnostics.py --p9-root ../../04_Results/P9_full_20261011_v1 --output ../../04_Results/P10_statistics_new --data-root <official-local-data-root>
python -B run_sensitivity.py --p9-root ../../04_Results/P9_full_20261011_v1 --output ../../04_Results/P10_statistics_new --data-root <official-local-data-root> --scratch <separate-internal-scratch-directory>
python -B run_interpretation.py --output ../../04_Results/P10_statistics_new
python -B run_monitor_constraints.py --output ../../04_Results/P10_statistics_new
python -B make_manifest.py --p9-root ../../04_Results/P9_full_20261011_v1 --output ../../04_Results/P10_statistics_new
```

Run these stages sequentially. Existing experiment directories are not silently
overwritten. Tests are `test_statistical_core.py` and
`test_sensitivity_contract.py`, in addition to the original scientific tests.

## Statistical estimands

- Independent source participant IDs, not windows, wrists, records, activities
  or stress states, are the sampling units. Cross-source absence of overlapping
  identities cannot be independently established and remains an assumption.
- PPG-Sport simultaneous wrists are first averaged within participant/time for
  decision loss, reporting, risk, cost, calls and coverage. Its 29 participants
  are not 58 independent observations. GalaxyPPG has 23 participants. Their
  source-specific evaluation is separate, not a single >=30-person cohort.
- Main confidence intervals use 10,000 participant-cluster percentile resamples
  (seed 20260921), conditional on the fitted models and fixed development split.
  They do not include model-training/split uncertainty. AF/AFL draws are stratified
  by the two independent-person reference classes, with equal class weighting.
- Selective risk and retained MAE are undefined for a person with no reports.
  Undefined values are not zeros: contributor counts and valid replicate counts
  are given. A CI is withheld when fewer than 95% of replicates are defined.
- The five prespecified classical paired contrasts share one Holm adjustment:
  AF/AFL adaptive-minus-quality loss, Stress-Predict adaptive-minus-quality loss,
  HR quality-minus-attempt loss separately in PPG-Sport and GalaxyPPG, and SHDB-AF
  adaptive-minus-fixed miss-penalized delay. The exact paired sign test tests the
  probability of a negative **nonzero** difference, not the mean-difference null.
  CNN comparisons, supplementary protocols and all sensitivity comparisons are
  exploratory; no winning setting is selected and the primary family is not
  expanded after examining results.
- Event-wise proportions and rates within each monitoring person are formed
  before person-macro averaging. All 30 evaluation people are retained; modified
  event-duration analyses disclose the actual contributing people and events.
  Missed onsets receive the specified horizon penalty. Detected-only delay is a
  different, conditional endpoint. False-alarm rates use known negative **valid
  full-clock** exposure, including abstained negative exposure.
- Person-level event indicators have Wilson, exact two-sided and exact one-sided
  bounds. Zero observed errors do not establish zero population risk or clinical
  safety. Empirical prediction-set coverage under temporal dependence and domain
  shift is not a distribution-free conditional/patient safety guarantee.

## Support and secondary diagnostics

`cohort_distribution.csv`, `individual_contribution_flow.csv` and
`frozen_person_roles.csv` contain disjoint role support. The complete label table
is **`label_distribution_all_roles.csv`**. `label_distribution.csv` is the initial
saved-decision/native-monitor support table and does not include all classification
training/policy labels. `training_fit_support.csv` explicitly distinguishes
candidate windows from windows usable for each model fit; unavailable windows
remain in evaluation support.

`endpoint_cluster_CI.csv` retains audited P9 person endpoint definitions. For
PPG-Sport its `primary_classifier_*`, `AUROC`, `all_estimable_PPG_MAE` and
`PPG_estimability` fields are supplementary **within-person wrist-window**
diagnostics, not the wrist-collapsed primary endpoint. Use the explicit
`HR_diagnostic_cluster_CI.csv` and `person_HR_diagnostics.csv` for newly computed
person-time-normalized HR MAE/estimability. AF/AFL `AUROC` in the CI table uses
patient mean final probabilities; a within-patient AUROC is not estimable for a
person whose reference class is constant. Source-pooled window AUROC is diagnostic
only and is not substituted for patient discrimination.

`latency_diagnostics.csv` distinguishes descriptive pooled-window p95 from the
mean of participant p95 values and the latter's cluster CI. All timings come from
the original isolated P9 measurements. Sensitivities replay fixed recorded
latencies and policy profiling ratios; they are not newly timed end-to-end runs.
Physical reference acquisition delay and mobile energy consumption were not
measured and are not inferred from desktop CPU time.

## Sensitivity contract

Prespecified one-factor changes cover alpha 0.05/0.10/0.20; train-person weighted
quality quantiles 0.10/0.20/0.30; planner penalties 0.005/0.02/0.05; HR unreliability
thresholds strictly >5/>10/>15 bpm; monitoring policy negative-decision caps
1%/5%/10%; onset-matching horizons 60/120/300 s; and eligible event duration
30/60 s. PPG-DaLiA wholly-contained reference-center alignment is an additional
prespecified alignment sensitivity.

Unchanged classification models are replayed from saved full probabilities;
each strategy is recalibrated only on its disjoint conformal people. Changed HR
targets and alignment **train new complete models**, with participant-grouped
training CV and fresh initialization, and refit policy/conformal stages only on
their fixed development roles. Temporary full feature/tensor caches remain outside
the submission data. Monitoring replay preserves the light-driven acquisition
schedule, reset times and recorded completion times. Alpha/quality/cap changes
refit the applicable calibration pools; horizon/event-duration changes are
offline scoring changes, not feedback into policy calibration.

## Interpretation and figures

AF/AFL uses perioperative finger PPG, not pure AF or wrist-device validation.
HR reference correction is an **offline independent-reference upper bound**, not
a deployed wrist ECG function. Stress protocols are reported separately.
Lower loss obtained primarily by abstention must be interpreted with report
fraction: the 20% minimum-report guard is not relaxed. The policy 5% negative
positive-decision cap is not a guarantee on evaluation false alarms/hour.
`reporting_guard_flags.csv` and `claim_limits.json` explicitly retain every
failed reporting guard. They are interpretation boundaries, not repaired outcomes.
`monitor_policy_and_evaluation_constraints.csv` distinguishes feasibility of the
policy-stage preconformal 90% onset-detection target from final held-out detection;
neither that target nor the policy negative-decision cap guarantees test performance.

Every figure has a source table and English explanatory caption in
`figure_captions.csv`. PNGs are 300 dpi; corresponding SVGs retain editable text.
Plots keep adverse results, no-report undefined values and overlapping strategies.
Lines between budget/activity operating points guide the eye, not a post-hoc
optimized frontier. No clinical efficacy claim is made solely from these figures.

`result_manifest.json` seals output SHA-256 and byte counts and records the exact
immutable P9 inputs. `reproducibility.json` gives the final analysis-code and
original P9 code hashes. `analysis_config.json` is the initial-generation snapshot,
not an assertion that later sensitivity modules were already present at that time.
Generation snapshots are not internal task acceptance reports.
