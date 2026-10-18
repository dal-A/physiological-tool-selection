"""Full-cohort provenance, bounded CSV writing and unrounded scientific outputs."""
from pathlib import Path
from collections import Counter
import csv
import json
import sys
import numpy as np
from .protocol import PACKAGE, PROTOCOL_SHA, ALLOCATION_SHA, sha256
from .experiment_io import clean, write_json, write_csv, environment


def create_full_run(output, task, family):
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError('Never overwrite an existing experiment: ' + str(path))
    path.mkdir(parents=True)
    write_json(path/'run_config.json', {
        'version': 'P9_full_protocol_p6_v1', 'task': task, 'model': family,
        'preset': 'full', 'CNN_max_epochs': 25, 'CNN_patience': 5,
        'all_frozen_people_and_qualified_windows': True,
        'warm_start': False, 'evaluation_used_for_fit_or_tuning': False,
        'clinical_validation_claim': False,
        'inputs': {'protocol_sha256': PROTOCOL_SHA, 'allocation_sha256': ALLOCATION_SHA,
                   'qualified_indices_manifest_sha256': sha256(PACKAGE.parents[1]/'03_Data/qualified_indices/manifest.json')},
        'code_files': {str(p.relative_to(PACKAGE)): sha256(p) for p in sorted(PACKAGE.rglob('*.py'))},
        'command': sys.argv, 'environment': environment()})
    return path


def seal_full_run(path):
    path = Path(path)
    # Prevent a long-running matrix from mixing code revisions.
    config = json.loads((path/'run_config.json').read_text(encoding='utf-8'))
    for rel, expected in config['code_files'].items():
        if sha256(PACKAGE/rel) != expected:
            raise AssertionError('Code changed during full run: '+rel)
    write_json(path/'effective_environment.json', environment())
    rows = [{'file': str(p.relative_to(path)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
            for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'result_manifest.json']
    write_json(path/'result_manifest.json', {'status': 'FULL_ARTIFACTS_SAVED_AWAITING_INDEPENDENT_ACCEPTANCE',
               'files': rows, 'not_a_clinical_safety_proof': True})


class CsvSink:
    """Write complete rows incrementally; no whole-stream list or silent truncation."""
    def __init__(self, path):
        self.stream = Path(path).open('w', encoding='utf-8', newline='')
        self.writer = None
        self.count = 0
    def add(self, rows):
        for row in rows:
            row = clean(row)
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            if set(row) != set(self.writer.fieldnames):
                raise ValueError('CSV row schema changed')
            self.writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
            self.count += 1
        self.stream.flush()
    def close(self):
        if self.writer is None:
            self.stream.write('no_observations\n')
        self.stream.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def selection_row(w):
    return {'subject': w.subject, 'source': w.source, 'role': w.role, 'record': w.record,
            'start_seconds': w.start, 'end_seconds': w.end, 'wrist': w.wrist,
            'technical_valid': w.technical_valid}


def flow_rows(windows):
    groups = {}
    for w in windows:
        key = (w.source, w.role, w.subject)
        g = groups.setdefault(key, {'records': set(), 'times': set(), 'windows': 0, 'invalid': 0})
        g['records'].add(w.record); g['times'].add((w.record, w.start, w.end))
        g['windows'] += 1; g['invalid'] += int(not w.technical_valid)
    return [{'source': s, 'role': r, 'subject': p, 'records': len(g['records']),
             'wrist_windows': g['windows'], 'unique_person_time_windows': len(g['times']),
             'technical_unavailable_windows': g['invalid']} for (s, r, p), g in sorted(groups.items())]


def full_figures(path, summaries, monitor=False):
    """Descriptive full outputs. P10 adds frozen inference/sensitivity figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = Path(path)/'figures'; figdir.mkdir()
    write_csv(figdir/'plotted_source_data.csv', summaries)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'attempt':'#666666', 'quality':'#0072B2', 'fixed_fusion':'#D55E00',
              'adaptive':'#009E73', 'fixed':'#0072B2'}
    def save(fig, name):
        fig.tight_layout(); fig.savefig(figdir/(name+'.png'), dpi=300)
        fig.savefig(figdir/(name+'.svg')); plt.close(fig)
    if not monitor:
        sources = sorted({r['source'] for r in summaries})
        for source in sources:
            part = [r for r in summaries if r['source']==source]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            for strategy in ('attempt','quality','fixed_fusion','adaptive'):
                rows = sorted((r for r in part if r['strategy']==strategy), key=lambda r:r['budget'])
                for ax, key, label in zip(axes, ['mean_decision_loss','report_fraction','selective_risk'],
                                          ['Decision loss','Reported fraction','Selective risk']):
                    ax.plot([r['mean_cost'] for r in rows],
                            [np.nan if r[key] is None else r[key] for r in rows],
                            'o-', color=colors[strategy], label=strategy)
                    ax.set(xlabel='Normalized mean tool cost', ylabel=label)
                    if key!='mean_decision_loss': ax.set_ylim(-.03,1.03)
            axes[0].legend(fontsize=8); fig.suptitle(source+' — full independent evaluation')
            save(fig, source.replace('-','_')+'_budget_risk_report')
        main = [r for r in summaries if r['budget']==max(t['budget'] for t in summaries)]
        fig, ax = plt.subplots(figsize=(max(7, len(main)*.65),4))
        ax.bar(range(len(main)), [r['secondary_call_fraction'] for r in main],
               color=[colors[r['strategy']] for r in main])
        ax.set_xticks(range(len(main)), [r['source']+'\n'+r['strategy'] for r in main], rotation=55, ha='right', fontsize=8)
        ax.set(ylabel='Secondary call fraction', ylim=(0,1)); save(fig,'secondary_calls')
    else:
        fig, axes = plt.subplots(1,3,figsize=(12,4))
        for ax, key, label in zip(axes,['miss_penalized_delay','FAR_per_non_target_hour','detection_rate'],
                                 ['Miss-penalized delay (s)','False alarm onsets / non-target hour','Onset detection fraction']):
            for strategy in ('fixed','adaptive'):
                rows = [r for r in summaries if r['strategy']==strategy]
                ax.plot(range(len(rows)), [np.nan if r.get(key) is None else r[key] for r in rows],
                        'o', color=colors[strategy], label=strategy, markersize=4)
            ax.set(xlabel='Frozen evaluation participant', ylabel=label)
        axes[0].legend(); save(fig,'monitor_person_endpoints')
