"""Scientific run provenance and plain, machine-readable result artifacts."""
from pathlib import Path
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
from .protocol import PROTOCOL_SHA, ALLOCATION_SHA, PACKAGE, sha256


def clean(value):
    if is_dataclass(value): return clean(asdict(value))
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ('+infinity' if value > 0 else '-infinity' if value < 0 else None)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, columns=None):
    rows=list(rows)
    columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    if not columns: columns=['no_observations']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for row in rows:
            row=clean(row)
            writer.writerow({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def window_key(w): return (w.subject,w.record,float(w.start),float(w.end),w.wrist)


def rank_key(value, purpose='quick'):
    return hashlib.sha256(('20260921|P8|'+purpose+'|'+repr(value)).encode()).hexdigest()


def environment():
    hardware={'CPU':platform.processor(),'RAM_bytes':None}
    if sys.platform=='win32':
        cmd="@{CPU=(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name); RAM_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Compress"
        hardware.update(json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',cmd], encoding='utf-8')))
    import torch
    from threadpoolctl import threadpool_info
    return {'date_UTC':datetime.now(timezone.utc).isoformat(), 'OS':platform.platform(), **hardware,
            'Python':platform.python_version(),'interpreter':sys.executable,
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','wfdb','matplotlib','joblib']},
            'torch_build':torch.__version__,'actual_torch_CPU_threads':torch.get_num_threads(),
            'CNN_training_CPU_thread_cap':4,'numerical_threadpools':threadpool_info(),'device':'cpu'}


def create_run(path, task, family, settings):
    path=Path(path).resolve()
    if path.exists(): raise FileExistsError('Existing result directory must never be overwritten: '+str(path))
    path.mkdir(parents=True)
    inputs={'protocol_sha256':PROTOCOL_SHA,'allocation_sha256':ALLOCATION_SHA,
            'qualified_indices_manifest_sha256':sha256(PACKAGE.parents[1]/'03_Data/qualified_indices/manifest.json')}
    config={'version':'P8_quick_protocol_p6_v1','task':task,'model':family,'preset':'quick',
            'use_as_full_or_clinical_evidence':False,'formal_independent_cohort_evaluation':False,
            'selection_before_predictions':True,'settings':settings,'inputs':inputs,
            'code_files':{str(p.relative_to(PACKAGE)):sha256(p) for p in sorted(PACKAGE.rglob('*.py'))},
            'command':sys.argv,'environment':environment()}
    write_json(path/'run_config.json',config)
    (path/'README.md').write_text('# Quick reproducibility run\n\nThese are real-data smoke results only, not the full independent-cohort experiment or clinical validation. The frozen person roles are retained; the deterministic reduced windows/participants and CNN two-epoch cap are smoke-only. Do not pool with historical outputs. Raw third-party signals are not redistributed. Each result is linked to the P6 hashes and exact code version in run_config.json.\n',encoding='utf-8')
    return path


def seal_results(path):
    path=Path(path)
    write_json(path/'effective_environment.json',environment())
    rows=[{'file':str(p.relative_to(path)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(path.rglob('*')) if p.is_file() and p.name!='result_manifest.json']
    write_json(path/'result_manifest.json',{'files':rows,'status':'QUICK_RUN_ARTIFACTS_SAVED','not_full_validation':True})


def diagnostic_figures(path, summary, predictions, monitor=False):
    """Standard scientific figures plus exact plotted source data, no invented CIs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    path=Path(path); figures=path/'figures'; figures.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figures/(name+'.png'));fig.savefig(figures/(name+'.svg'));plt.close(fig)
    if not monitor:
        write_csv(figures/'budget_risk_source.csv', summary)
        groups=defaultdict(list)
        for row in summary: groups[(row['source'],row['strategy'])].append(row)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for (source,strategy),rows in groups.items():
            rows=sorted(rows,key=lambda r:r['budget'])
            ax.plot([r['mean_cost'] for r in rows],[np.nan if r['selective_risk'] is None else r['selective_risk'] for r in rows],marker='o',label=source+' / '+strategy)
        ax.set(xlabel='Normalized mean tool cost',ylabel='Selective risk among reported windows',title='Quick pipeline diagnostic (not full validation)')
        if all(r['selective_risk'] is None for r in summary):ax.text(.5,.5,'No reported windows: selective risk undefined',transform=ax.transAxes,ha='center',wrap=True)
        ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1));save(fig,'budget_risk')
        main=[r for r in summary if r['budget']==max(x['budget'] for x in summary)]
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        labels=[r['source']+'\n'+r['strategy'] for r in main]
        for ax,key,title in zip(axes,['report_fraction','secondary_call_fraction'],['Reported fraction','Secondary call fraction']):
            ax.bar(range(len(main)),[r[key] for r in main],color='#336B87');ax.set_xticks(range(len(main)),labels,rotation=70,fontsize=7);ax.set_ylim(0,1);ax.set_ylabel(title)
        save(fig,'reports_and_calls')
    else:
        write_csv(figures/'monitor_person_metrics_source.csv', summary)
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        for ax,key,title in zip(axes,['miss_penalized_delay','FAR_per_non_target_hour'],['Miss-penalized delay (s)','False alarms / non-target hour']):
            labels=[r['subject'][-6:]+' / '+r['strategy'] for r in summary]
            vals=[np.nan if r.get(key) is None else r[key] for r in summary]
            ax.bar(range(len(vals)),vals,color=['#336B87' if r['strategy']=='fixed' else '#C48445' for r in summary]);ax.set_xticks(range(len(vals)),labels,rotation=65,fontsize=7);ax.set_ylabel(title)
        fig.suptitle('Quick event-scoring diagnostic (not full validation)');save(fig,'monitor_events')
        if predictions:
            person=predictions[0]['subject'];rows=[r for r in predictions if r['subject']==person]
            fig,ax=plt.subplots(figsize=(8,3.5))
            for strategy in ('fixed','adaptive'):
                part=[r for r in rows if r['strategy']==strategy]
                ax.plot([r['end_seconds'] for r in part],[np.nan if r['probability'] is None else r['probability'] for r in part],'.-',label=strategy)
            ax.set(xlabel='Native record time (s)',ylabel='Heavy-model target probability',title='Same-person causal scheduled calls');ax.legend();save(fig,'monitor_timeline')
