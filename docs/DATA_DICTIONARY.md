# Derived data dictionary

The 26 statistical CSVs are exact copies of the frozen P10 tables. The two demographic tables are aggregate source summaries newly used in the manuscript. No original signals are included.

Independent source participant IDs are the sampling units. Wrist pairs and repeated records are not independent people. Probabilities, coverage, report fractions, selective risk and decision loss are proportions unless their field explicitly states percent. HR errors are bpm; delays are seconds; CPU latency units are stated in the corresponding metric/column. FAR is false alarms per known-negative valid hour. Normalized tool costs are dimensionless proxies, not money or energy. `budget_hundredths` is an integer representation of normalized budget (420 means 4.20).

Blank/NaN endpoints or intervals remain undefined, never zero. Defined participant and bootstrap-replicate counts must accompany risk or conditional-delay claims. Primary CIs use 10,000 participant-cluster percentile resamples, conditional on fixed fitted models and development roles. The exact sign test concerns the sign among nonzero paired differences, not a null hypothesis about the mean effect. CNN, sensitivity and supplementary-protocol inference are exploratory.

## Tables and column names

### HR_diagnostic_cluster_CI.csv

Explicit person-time-normalized HR estimability and MAE diagnostics; use these for paired-wrist MAE. Rows: 192.

Columns: `model`, `source`, `strategy`, `budget_hundredths`, `metric`, `exploratory`, `aggregation`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### activity_protocol_strata.csv

Source-specific activity/protocol strata and person-level support; exploratory, not independent groups. Rows: 1680.

Columns: `task`, `model`, `source`, `activity`, `strategy`, `metric`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### additional_cohort_demographics.csv

Aggregate raw-source age and source-recorded sex/gender summaries for monitoring and supplementary stress cohorts; missing ages excluded, not imputed. Rows: 3.

Columns: `source`, `n`, `M`, `F`, `unknown`, `median_age`, `q1`, `q3`, `age_available`, `field`.

### all_person_endpoints.csv

Every participant endpoint by source/model/strategy/budget; the independent sampling unit for primary statistics. Rows: 5784.

Columns: `eligible_windows`, `reported_windows`, `report_fraction`, `selective_risk`, `dangerous_acceptance_all_windows`, `no_reports`, `mean_decision_loss`, `selective_risk_undefined_reason`, `subject`, `label`, `mean_cost`, `secondary_call_fraction`, `set_coverage`, `primary_mean_probability`, `primary_classifier_AUROC`, `primary_classifier_accuracy`, `final_mean_probability`, `AUROC`, `source`, `strategy`, `budget_hundredths`, `task`, `model`, `retained_MAE`, `all_estimable_PPG_MAE`, `PPG_estimability`, `records`, `events`, `detected`, `missed`, `detection_rate`, `miss_penalized_delay`, `detected_only_delay`, `detected_only_delay_undefined_reason`, `negative_hours`, `false_alarms`, `FAR_per_non_target_hour`, `FAR_undefined_reason`, `negative_alarm_occupancy`, `non_target_positive_decision_fraction`, `known_negative_decision_denominator`, `eligible_pure_heavy_decisions`, `reported_heavy_decisions`, `valid_signal_hours`, `normalized_cost`, `normalized_cost_per_valid_hour`, `heavy_calls`, `heavy_calls_per_valid_hour`, `p95_CPU_milliseconds`, `complete_10s_ticks`, `no_accepted_heavy_decision`, `no_decision_seconds`, `unknown_exposure_seconds`, `technical_missing_seconds`.

### classification_sensitivity_CI.csv

Prespecified one-factor classification sensitivity endpoints with fixed-pipeline person-cluster intervals. Rows: 2688.

Columns: `task`, `model`, `source`, `strategy`, `variant`, `factor`, `factor_value`, `metric`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### classification_sensitivity_contrasts.csv

Exploratory paired sensitivity contrasts; not additional primary-family tests. Rows: 68.

Columns: `task`, `model`, `source`, `variant`, `metric`, `contrast`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`, `improved_people`, `worsened_people`, `ties`, `sign_effective_people`, `sign_p_raw`, `sign_test_null`.

### cohort_distribution.csv

Independent source IDs, records and person-time support by development/evaluation role. Rows: 19.

Columns: `task`, `source`, `role`, `independent_source_IDs`, `records`, `wrist_windows`, `unique_person_time_windows`, `technical_unavailable_windows`, `valid_signal_hours`, `record_exposure_hours`, `negative_valid_hours`, `unknown_reference_hours`, `qualifying_onsets`.

### endpoint_cluster_CI.csv

Primary and supplementary endpoint estimates, 95% person-cluster intervals and defined contributor/replicate support. Rows: 2176.

Columns: `task`, `model`, `source`, `strategy`, `budget_hundredths`, `metric`, `aggregation`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### frozen_person_roles.csv

Frozen source participant role membership; multiple records/wrists inherit one person role. Rows: 362.

Columns: `subject`, `task`, `source`, `role`, `label`, `training_CV_fold`, `source_subject`.

### individual_contribution_flow.csv

Participant, candidate window, usable window and task contribution flow by frozen role. Rows: 366.

Columns: `task`, `source`, `role`, `subject`, `records`, `wrist_windows`, `unique_person_time_windows`, `technical_unavailable_windows`, `record`, `complete_10s_blocks`, `valid_signal_seconds`, `record_exposure_seconds`, `unknown_reference_seconds`, `known_negative_valid_seconds`, `qualifying_onset_events`.

### label_distribution.csv

Saved-decision/native-monitor label support only; not a substitute for the all-role label table. Rows: 24.

Columns: `task`, `source`, `role`, `label`, `label_definition`, `contributing_people`, `windows`.

### label_distribution_all_roles.csv

Complete reference-label distribution across every frozen classification/monitoring role. Rows: 42.

Columns: `task`, `source`, `role`, `label`, `label_definition`, `contributing_people`, `wrist_windows`, `unique_person_time_windows`, `technical_unavailable_wrist_windows`.

### latency_diagnostics.csv

Original CPU latency diagnostics, separating pooled-window p95 from participant-p95 macro averages and intervals. Rows: 196.

Columns: `task`, `model`, `source`, `strategy`, `budget_hundredths`, `pooled_window_p95_CPU_milliseconds`, `mean_of_person_p95_CPU_milliseconds`, `aggregation`, `exploratory`, `physical_sensor_acquisition_latency`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### monitor_policy_and_evaluation_constraints.csv

Policy-selection constraints versus evaluation outcomes; feasibility and 5% policy cap are not clinical FAR guarantees. Rows: 40.

Columns: `model`, `variant`, `strategy`, `policy_negative_decision_cap`, `selected_alarm_probability_threshold`, `policy_maximum_negative_positive_fraction`, `policy_cap_pass`, `policy_preconformal_90percent_detection_target_feasible`, `policy_preconformal_minimum_detection`, `final_evaluation_person_macro_detection`, `final_evaluation_person_macro_FAR_per_negative_hour`, `final_evaluation_person_macro_report_fraction`, `final_evaluation_person_macro_miss_penalized_delay`, `evaluation_people`, `event_contributing_people`, `horizon_seconds`, `minimum_event_duration_seconds`, `exploratory`, `clinical_validity_established`, `policy_preconformal_target_not_final_test_detection_guarantee`, `policy_decision_cap_not_test_FAR_bound`.

### monitor_sensitivity_CI.csv

Exploratory continuous-monitoring sensitivity endpoints, clustering records within people. Rows: 360.

Columns: `task`, `model`, `source`, `strategy`, `variant`, `factor`, `factor_value`, `metric`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`.

### monitor_sensitivity_contrasts.csv

Exploratory adaptive-minus-fixed monitoring contrasts, retaining uncertainty and misses. Rows: 20.

Columns: `task`, `model`, `source`, `variant`, `contrast`, `metric`, `all_evaluation_people`, `event_contributing_people`, `exploratory`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`, `improved_people`, `worsened_people`, `ties`, `sign_effective_people`, `sign_p_raw`, `sign_test_null`.

### monitor_sensitivity_support.csv

Participants, eligible onsets and exposure contributing to each monitoring sensitivity setting. Rows: 40.

Columns: `model`, `variant`, `strategy`, `evaluation_people`, `event_contributing_people`, `qualifying_events`, `detected_events`, `negative_valid_hours`, `horizon_seconds`, `minimum_event_duration_seconds`.

### patient_ROC_points.csv

Patient-level AF/AFL ROC coordinate points from mean primary-PPG probabilities in the attempt strategy; not pooled-window ROC. Strategy-specific final-probability AUROCs in the endpoint table are distinct. Rows: 28.

Columns: `model`, `FPR`, `TPR`, `threshold`.

### person_HR_diagnostics.csv

Within-person paired-time HR MAE and estimability diagnostics used to produce the HR CI table. Rows: 1664.

Columns: `task`, `model`, `source`, `strategy`, `budget_hundredths`, `subject`, `PPG_estimability_person_time`, `all_estimable_PPG_MAE_person_time`, `retained_MAE_person_time`, `wrists_averaged_within_person_time`.

### person_error_event_bounds.csv

Participant-level error-event indicators with Wilson and exact one/two-sided bounds; zero observed events do not mean zero risk. Rows: 56.

Columns: `task`, `model`, `source`, `strategy`, `endpoint`, `independent_people`, `people_with_at_least_one_event`, `proportion`, `Wilson95_low`, `Wilson95_high`, `exact95_low`, `exact95_high`, `exact_one_sided95_upper`, `zero_observed_events_not_zero_risk_proof`.

### person_latency.csv

Original measured participant CPU latency summary data; no reference acquisition time or mobile energy estimate. Rows: 5784.

Columns: `task`, `model`, `source`, `strategy`, `budget_hundredths`, `subject`, `p95_CPU_milliseconds`, `observed_calls`.

### person_paired_differences.csv

Same-participant differences for five primary contrasts and exploratory CNN parallels. Rows: 344.

Columns: `id`, `model`, `subject`, `left`, `right`, `difference`, `label`.

### primary_paired_contrasts.csv

Five classical primary contrasts with one Holm family, paired cluster CIs and exact sign-test support; CNN parallels exploratory. Rows: 10.

Columns: `id`, `model`, `source`, `metric`, `contrast`, `budget_hundredths`, `exploratory`, `negative_difference_is_better`, `estimate`, `CI95_low`, `CI95_high`, `independent_people`, `contributing_people`, `valid_replicates`, `requested_replicates`, `seed`, `undefined_reason`, `improved_people`, `worsened_people`, `ties`, `sign_effective_people`, `sign_p_raw`, `sign_test_null`, `sign_p_Holm_family_of_five`, `inference_family`.

### qualification_flow.csv

Source qualification, exclusion and allocation flow; no pooling of undersized external HR sources. Rows: 7.

Columns: `source`, `qualified_people`, `main_evaluation_people`, `details`.

### reporting_guard_flags.csv

Prespecified minimum-report support flags, including settings that fail the 20% reporting guard. Rows: 192.

Columns: `task`, `model`, `source`, `strategy`, `budget_hundredths`, `independent_people`, `report_fraction`, `minimum_report_fraction`, `minimum_report_guard_pass`, `empirical_set_coverage`, `clinical_validity_established`, `secondary_HR_reference_is_offline_oracle`, `failure_reason`.

### training_fit_support.csv

Candidate versus model-usable training support; unusable evaluation windows are not silently removed. Rows: 338.

Columns: `task`, `model`, `modality`, `subject`, `candidate_windows`, `included_in_fit`.

### vitaldb_evaluation_demographics.csv

Aggregate evaluation demographics separately for 30 annotated sinus and 30 AF/AFL patients; not new model covariates. Rows: 2.

Columns: `group`, `n`, `M`, `F`, `unknown`, `age_median`, `age_q1`, `age_q3`, `age_available`.

### window_reference_labels.csv

Reconstructed reference-label support used for all-role distributions; retained source/person/role context. Rows: 67622.

Columns: `task`, `subject`, `source`, `role`, `record`, `start_seconds`, `end_seconds`, `wrist`, `technical_valid`, `label`, `label_definition`.

## Publication figure mapping

`figure_mapping.csv` maps the 23 vector PDF assets to their source tables and displayed numbers: 15 figures in each main manuscript (including a language-specific workflow) and seven supplementary figures. Asset filenames are stable and do not necessarily match printed figure numbers. The original 20 statistical displays retain frozen P10 values; workflow diagrams are explanatory. `report_loss_points.csv` contains 48 unchanged selected endpoint rows for the reporting--loss figure. No raw waveforms are included.
