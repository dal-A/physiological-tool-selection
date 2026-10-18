# Physiological Tool Selection

Research software accompanying **Quality-aware tool selection and selective physiological reporting: a subject-disjoint multi-cohort evaluation**.

Authors: Yuna Shan, Jun Guo, Hao Zhao and Yongming Chen. Correspondence: Jun Guo, junguo0407@cuit.edu.cn. Funding: university-level undergraduate innovation and entrepreneurship training project X202610621352, Chengdu University of Information Technology.

The implementation combines physiological signal processing, quality proxies, logistic regression and lightweight convolutional models, conditional secondary-tool use, split conformal prediction and causal change-point monitoring. Statistical analysis uses participant-level aggregation and paired cluster bootstrap intervals. The scientific protocol is `PM_P6_20261010_v1`; the corresponding full experiments and statistics are `P9_full_20261011_v1` and `P10_statistics_20261011_v1`.

## Repository contents

| Location | Content |
|---|---|
| `02_Code/physiological_tool_selection/` | Adapters, algorithms, full/quick runners, frozen configurations and scientific tests |
| `02_Code/P10_statistical_analysis/` | Participant-level statistics, diagnostics and prespecified sensitivity analysis |
| `02_Code/P12_publication_figures_20261012_v1/` | Reproducible workflow and reporting-versus-loss figures |
| `03_Data/qualified_indices.zip` | Compressed frozen participant/window eligibility indices; no physiological waveforms |
| `figure_inputs/` | Frozen endpoint table and licensed font needed by the publication-figure extension |
| `docs/DATA_SOURCES.md` | Official data releases, acquisition links and source-specific conditions |
| `docs/REPRODUCTION.md` | Full experiment and analysis sequence |
| `CITATION.cff` | Software citation metadata; no publication DOI is claimed |
| `PUBLICATION_MANIFEST.json` | File sizes and SHA-256 hashes for this release copy |

The package does not include third-party raw signals, trained models or the complete large full-result directories. The article's accompanying research package retains those derived results separately. This repository is sufficient to inspect the scientific implementation and run its synthetic/contract tests after unpacking the indices. Full experiments also require the legally acquired and correctly prepared source recordings.

## Environment and index setup

The recorded environment is **Windows 11, Python 3.12.4, CPU PyTorch 2.14.0+cpu**. Exact scientific dependency pins are in the original package requirements. Configure the matching CPU PyTorch build for your platform before installing the remaining requirements. Other operating systems and dependency upgrades have not been validated against the frozen results.

From the repository root in PowerShell, using the intended Python environment:

```powershell
python -m pip install -r requirements.txt
python -m zipfile -e 03_Data/qualified_indices.zip 03_Data/qualified_indices
```

Unpack the indices once into a new checkout. Keep the `02_Code` / `03_Data` relative layout. Do not edit the frozen JSONs or reformat their bytes: their SHA-256 hashes are enforced by the implementation. Absolute paths in historical environment/allocation metadata identify the original machine; actual source access uses `--data-root` and source-relative paths.

## Verify without running full experiments

Run each command from the specified directory. Tests use synthetic signals and the packaged indices; they do not download third-party recordings.

```powershell
cd 02_Code/physiological_tool_selection
python -B -m pytest -q -p no:cacheprovider
cd ../P10_statistical_analysis
python -B -m pytest -q -p no:cacheprovider
cd ../..
```

## Full experiments

Acquire and prepare the exact releases described in `docs/DATA_SOURCES.md` and `docs/DATA_LAYOUT.md`. This release does not include a turnkey source downloader or the original dataset-acquisition/qualification workflow. In particular, the VitalDB adapter expects the already qualified, native-rate paired-window NPZ files in `vitaldb_af_afl_p3_v1/prepared_native_windows`; downloading only the provider's original files does not create that intermediate directory.

From `02_Code/physiological_tool_selection`:

```powershell
python -B scripts/run_full.py --data-root "D:/path/to/official-local-data-root" --task af_afl --model classical --preset full --output "D:/path/to/new-results/af_afl_classical_full"
```

Use tasks `af_afl`, `hr_reliability`, `stress` and `continuous_monitoring`, each with `classical` and `cnn`. Run the eight commands sequentially with new output directories. Quick runs are smoke checks, not the article's full evidence. Follow `docs/REPRODUCTION.md` and the P10 scientific README for statistics; analysis requires the newly reproduced full results or the separately retained complete P9 outputs.

## Reproduce the additional publication figures

From the repository root, with the ReportLab requirement installed:

```powershell
python 02_Code/P12_publication_figures_20261012_v1/build_publication_figures.py --source-dir figure_inputs --output-dir "D:/path/to/new-publication-figures"
```

The endpoint table is the verified existing statistical table. The reference line is analytical; figure generation does not create new experimental outcomes.

## Interpretation and reuse

- The rhythm task is AF/AFL versus annotated sinus rhythm; VitalDB uses perioperative finger signals.
- HR reference replacement is an offline information upper bound, not a validated wrist ECG feature.
- Source-specific HR and supplementary stress cohorts remain separate; participant counts are not inflated by windows, wrists or protocols.
- Report fraction, undefined conditional risk and unfavorable outcomes must be retained. Empirical conformal coverage is not a clinical safety guarantee.

See `LICENSE_STATUS.md` before reuse. No open-source software licence has yet been selected. Third-party datasets keep their own terms; the bundled Noto font has its accompanying SIL Open Font License.

