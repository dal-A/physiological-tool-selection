# Reproducible publication figures

This extension produces a schematic of the frozen tool-selection execution
and a reporting-versus-loss display of existing classical operating points.
It does not train models, fit policies, recompute confidence intervals or change
the P9/P10 scientific outputs.

Use Python 3.12 and ReportLab 4.4.9. The source directory must contain the frozen
`endpoint_cluster_CI.csv` and the open-licensed `NotoSerifSC_Regular.ttf` supplied
in the bilingual LaTeX source bundle. Font licensing is supplied there. Run:

```text
python build_publication_figures.py --source-dir <extracted-bilingual-source> --output-dir <new-figure-directory>
```

The workflow depicts disjoint development roles and gated, available-primary
window classification. The attempt control, unavailable/under-budget primary,
offline HR-reference override and continuous-monitor controller have separate
rules in the scientific package and manuscript algorithm.

The endpoint projection selects the four strategies at budgets 4.20 for VitalDB,
PPG-Sport and GalaxyPPG and 2.70 for Stress-Predict. It preserves all 48 selected
metric records in `report_loss_points.csv` with the source-file SHA-256.
Horizontal/vertical bars are separate marginal person-cluster 95% intervals,
not joint regions. The dashed loss=0.2 line is analytical all-abstention loss,
not an extra experimental arm. The HR reference points coincide by construction.

`new_figure_provenance.json` records source/output hashes. PDF output is vector,
150 mm wide with 8.5 pt regular figure text and redundant shape/color encoding.
The research-software public release URL and licence remain author decisions;
this extension does not assert a public release or an institution-approved licence.
