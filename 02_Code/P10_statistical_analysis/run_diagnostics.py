"""Complete support and descriptive diagnostics without changing P9 decisions.

All secondary diagnostics are exploratory. Never equate person-macro p95 with
the pooled-window p95, or count two wrists as two independent participants.
"""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistical_core import bootstrap_mean
from run_statistics import (TASKS,FAMILIES,STRATEGIES,COLORS,LABELS,SHORT,
                            read_json,check_inputs,plot_style,save_figure,SCIENCE)
from pmtools.protocol import FrozenDesign
from pmtools.adapters import DatasetAdapters
from pmtools.signals import estimate_hr
from pmtools.safety import unreliable
from pmtools.experiment_io import write_csv,write_json


def csv(path):return pd.read_csv(path,dtype={'subject':str,'record':str,'wrist':str},low_memory=False)


def support_labels(root,data_root,design,out):
    adapter=DatasetAdapters(data_root,design=design);allrows=[];fitrows=[]
    for task in TASKS[:-1]:
        run=root/(task+'_classical_full');selection=read_json(run/'selection_before_training.json')
        dev={}
        if task!='af_afl':
            source='PPG-DaLiA' if task=='hr_reliability' else 'WESAD'
            for role in ('train','policy'):
                for person in design.select(source,role):
                    for w in adapter.iter_person(person['subject']):
                        key=(w.subject,w.record,w.start,w.end,w.wrist)
                        target=unreliable(estimate_hr(w.signals['BVP']) if w.technical_valid else None,w.reference_hr) if task=='hr_reliability' else w.label
                        dev[key]=target
                    print('DIAGNOSTIC_LABEL_REPLAY',task,person['subject'],flush=True)
        for filename in ('conformal_execution.csv','predictions.csv'):
            df=csv(run/filename);g=df[(df.strategy=='attempt')&(df.budget_hundredths==115)]
            for r in g.to_dict('records'):
                wrist=None if pd.isna(r['wrist']) else r['wrist']
                dev[(r['subject'],r['record'],r['start_seconds'],r['end_seconds'],wrist)]=int(r['label_primary'])
        for row in selection:
            key=tuple(row[k] for k in ('subject','record','start_seconds','end_seconds','wrist'))
            y=design.people[row['subject']]['label'] if task=='af_afl' else dev[key]
            assert y in (0,1)
            allrows.append(dict(task=task,**row,label=int(y),label_definition='AF_AFL_target' if task=='af_afl' else 'stress_state' if task=='stress' else 'PPG_error_gt10_or_unestimable'))
        for family in FAMILIES:
            paths=list((root/(task+'_'+family+'_full')/'models').glob('*/fit_windows.csv'))
            for path in paths:
                fit=csv(path)
                for (subject,included),g in fit.groupby(['subject','included_in_fit']):
                    fitrows.append(dict(task=task,model=family,modality=path.parent.name,subject=subject,
                                        candidate_windows=len(g),included_in_fit=bool(included)))
    frame=pd.DataFrame(allrows);summary=[]
    for (task,source,role,label),g in frame.groupby(['task','source','role','label']):
        summary.append(dict(task=task,source=source,role=role,label=int(label),
            label_definition=g.label_definition.iloc[0],contributing_people=g.subject.nunique(),
            wrist_windows=len(g),unique_person_time_windows=len(g.drop_duplicates(['subject','record','start_seconds','end_seconds'])),
            technical_unavailable_wrist_windows=int((~g.technical_valid).sum())))
    # Complete monitoring 10-second labels include unknown/mixed (-1).
    old=csv(out/'tables/label_distribution.csv');old=old[old.task=='continuous_monitoring']
    for r in old.to_dict('records'):
        summary.append(dict(task=r['task'],source=r['source'],role=r['role'],label=int(r['label']),
            label_definition=r['label_definition'],contributing_people=int(r['contributing_people']),
            wrist_windows=int(r['windows']),unique_person_time_windows=int(r['windows']),technical_unavailable_wrist_windows=None))
    write_csv(out/'tables/label_distribution_all_roles.csv',summary)
    write_csv(out/'tables/training_fit_support.csv',fitrows)
    write_csv(out/'tables/window_reference_labels.csv',allrows)


def diagnostics(root,out):
    latency=[];persons=[];hr=[]
    for task in TASKS:
        for model in FAMILIES:
            pred=csv(root/(task+'_'+model+'_full')/'predictions.csv')
            if task=='continuous_monitoring':pred['source']='SHDB-AF';pred['budget_hundredths']=0
            for (source,strategy,budget),g in pred.groupby(['source','strategy','budget_hundredths']):
                rows=[]
                for subject,v in g.groupby('subject',sort=True):
                    val=float(np.quantile(v.CPU_milliseconds,.95))
                    rows.append(val);persons.append(dict(task=task,model=model,source=source,strategy=strategy,
                        budget_hundredths=int(budget),subject=subject,p95_CPU_milliseconds=val,observed_calls=len(v)))
                y=[FrozenDesign().people[s]['label'] for s in sorted(g.subject.unique())] if task=='af_afl' else None
                r=bootstrap_mean(rows,y)
                latency.append(dict(task=task,model=model,source=source,strategy=strategy,budget_hundredths=int(budget),
                    pooled_window_p95_CPU_milliseconds=float(np.quantile(g.CPU_milliseconds,.95)),
                    mean_of_person_p95_CPU_milliseconds=r.pop('estimate'),aggregation='person_macro_p95_CI_not_pooled_p95_CI',
                    exploratory=True,physical_sensor_acquisition_latency=None,**r))
                if task!='hr_reliability':continue
                for subject,v in g.groupby('subject',sort=True):
                    available=v.PPG_HR_bpm.notna();accepted=~v.abstain
                    error=(v.PPG_HR_bpm-v.reference_HR_bpm).abs()
                    x=pd.DataFrame(dict(record=v.record,start=v.start_seconds,end=v.end_seconds,
                        estimable=available.astype(float),error_sum=error.fillna(0),
                        retained_count=accepted.astype(float),retained_sum=(v.post_final_absolute_error_bpm.fillna(0)*accepted)))
                    if source=='PPGSPORT':x=x.groupby(['record','start','end'])[['estimable','error_sum','retained_count','retained_sum']].mean()
                    denominator=x.estimable.mean();keep=x.retained_count.mean()
                    hr.append(dict(task=task,model=model,source=source,strategy=strategy,budget_hundredths=int(budget),subject=subject,
                        PPG_estimability_person_time=denominator,
                        all_estimable_PPG_MAE_person_time=x.error_sum.mean()/denominator if denominator else None,
                        retained_MAE_person_time=x.retained_sum.mean()/keep if keep else None,
                        wrists_averaged_within_person_time=source=='PPGSPORT'))
            print('DIAGNOSTICS_COMPLETE',task,model,flush=True)
    write_csv(out/'tables/latency_diagnostics.csv',latency);write_csv(out/'tables/person_latency.csv',persons)
    write_csv(out/'tables/person_HR_diagnostics.csv',hr)
    hrframe=pd.DataFrame(hr);ci=[]
    for (model,source,strategy,budget),g in hrframe.groupby(['model','source','strategy','budget_hundredths']):
        for metric in ('PPG_estimability_person_time','all_estimable_PPG_MAE_person_time','retained_MAE_person_time'):
            ci.append(dict(model=model,source=source,strategy=strategy,budget_hundredths=int(budget),metric=metric,
                exploratory=True,aggregation='wrists_then_person_time_then_person_macro',**bootstrap_mean(g[metric])))
    write_csv(out/'tables/HR_diagnostic_cluster_CI.csv',ci)
    return pd.DataFrame(latency),pd.DataFrame(ci)


def extra_figures(out,latency,hrci):
    plot_style();index=pd.read_csv(out/'figure_captions.csv').to_dict('records')
    e=csv(out/'tables/endpoint_cluster_CI.csv');sources=['VITALDB','Stress-Predict','PPGSPORT','GALAXYPPG']
    fig,axes=plt.subplots(2,4,figsize=(13.5,6))
    for i,model in enumerate(FAMILIES):
        for j,source in enumerate(sources):
            ax=axes[i,j]
            for strategy in STRATEGIES:
                g=e[(e.model==model)&(e.source==source)&(e.strategy==strategy)&(e.metric=='secondary_call_fraction')].sort_values('budget_hundredths')
                ax.plot(g.budget_hundredths/100,g.estimate,'o-',ms=3,label=LABELS[strategy],color=COLORS[strategy])
            ax.set_title(SHORT[source]+' / '+model);ax.set_xlabel('Available normalized budget');ax.set_ylim(-.02,1.02)
            if j==0:ax.set_ylabel('Person-macro secondary-call fraction')
    axes[0,0].legend(fontsize=7)
    save_figure(fig,out,'fig17_secondary_call_budget','endpoint_cluster_CI.csv',
        'Secondary-call fractions across all frozen budgets, separately by source and model. HR reference calls are offline simulations, not wearable ECG acquisition.',index)
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for i,model in enumerate(FAMILIES):
        for j,source in enumerate(('PPGSPORT','GALAXYPPG')):
            ax=axes[i,j];g=hrci[(hrci.model==model)&(hrci.source==source)&(hrci.budget_hundredths==420)&(hrci.metric=='retained_MAE_person_time')].set_index('strategy').reindex(STRATEGIES)
            ax.errorbar(range(4),g.estimate,yerr=np.maximum(0,np.array([g.estimate-g.CI95_low,g.CI95_high-g.estimate])),fmt='o',capsize=3,color='#0072B2')
            ax.set_xticks(range(4),['Attempt','Gate','Fixed','Adaptive'],rotation=30);ax.set_ylabel('Retained HR MAE (bpm)');ax.set_title(SHORT[source]+' / '+model)
    save_figure(fig,out,'fig18_retained_HR_error','HR_diagnostic_cluster_CI.csv',
        'Retained HR mean absolute error with person-cluster intervals at the main budget. Within-person simultaneous wrists are averaged before person aggregation. Zero error after secondary correction is the offline reference upper bound.',index)
    fig,axes=plt.subplots(2,4,figsize=(13.5,6))
    for i,model in enumerate(FAMILIES):
        for j,source in enumerate(sources):
            ax=axes[i,j]
            for strategy in STRATEGIES:
                g=latency[(latency.model==model)&(latency.source==source)&(latency.strategy==strategy)].sort_values('budget_hundredths')
                ax.plot(g.budget_hundredths/100,g.pooled_window_p95_CPU_milliseconds,'o-',ms=3,label=LABELS[strategy],color=COLORS[strategy])
            ax.set_title(SHORT[source]+' / '+model);ax.set_xlabel('Available normalized budget')
            if j==0:ax.set_ylabel('Observed pooled-window p95 CPU (ms)')
    axes[0,0].legend(fontsize=7)
    save_figure(fig,out,'fig19_measured_latency','latency_diagnostics.csv',
        'Original isolated P9 CPU timings, not newly timed sensitivity runs. Pooled-window p95 values are descriptive; the companion table separately gives person-p95 means and cluster intervals. No physical acquisition latency or mobile-energy claim is made.',index)
    write_csv(out/'figure_captions.csv',index)


def main():
    p=argparse.ArgumentParser();p.add_argument('--p9-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;assert not (out/'diagnostics_stage_status.json').exists()
    check_inputs(a.p9_root);design=FrozenDesign();support_labels(a.p9_root,a.data_root,design,out)
    lat,hr=diagnostics(a.p9_root,out);extra_figures(out,lat,hr);check_inputs(a.p9_root)
    write_json(out/'diagnostics_stage_status.json',dict(diagnostics_generated=True,independent_acceptance_pending=True,P10_complete=False))


if __name__=='__main__':main()
