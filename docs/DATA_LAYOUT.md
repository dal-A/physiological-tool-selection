# Local source-data layout required by the adapters

Pass the directory containing the following source folders through `--data-root`.
Only acquire sources under their own access and use conditions. The ZIP of frozen
eligibility indices contains no raw signal samples and cannot replace this data.

```text
<data-root>/
  vitaldb_af_afl_p3_v1/prepared_native_windows/<caseid>.npz
  ppg+dalia/data/PPG_FieldStudy/S*/S*.pkl
  WESAD/WESAD/S*/S*.pkl
  ppg_sport_p4_candidate/source_repository/<index-relative-path>
  galaxyppg_p4_v1/Dataset/<source-subject>/E4/{BVP,ACC}.csv
  additional_cohorts/stress_predict/extracted/
    Stress-Predict-Dataset-main/Raw_data/<record>/{BVP,ACC,EDA}.csv
  additional_cohorts/wearable_device_stress_exercise_1_0_1/extracted/
    wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/
    Wearable_Dataset/STRESS/<record>/{BVP,ACC,EDA}.csv
  shdb_af_p5_v1_0_1/<record>.hea
  shdb_af_p5_v1_0_1/<record>.dat
```

The VitalDB NPZ files are prepared native-rate paired windows, not a provider's
off-the-shelf download. The frozen adapter checks fields `subjectid`, `label`,
`starts`, `ecg`, `ppg`, `ecg_dt` and `ppg_dt` against the qualification index.
The exact original acquisition/qualification outputs are necessary for an exact
full-run replay. This code archive does not supply a new downloader or recreate
those intermediate files automatically.

PPG-Sport and GalaxyPPG retain the frozen reference HR and timestamp correction
recorded by the eligibility indices. Stress source directories retain native
Empatica timing headers. SHDB-AF uses the fixed 200 Hz ECG1 schema. Substituting a
different release or regenerating eligibility by model performance changes the
scientific design and does not reproduce the article's experiment.
