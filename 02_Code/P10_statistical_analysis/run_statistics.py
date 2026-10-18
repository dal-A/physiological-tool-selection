"""Recalculate P10 statistics from immutable P9 full outputs, never old runs.

Usage: python run_statistics.py --p9-root ../../04_Results/P9_full_20261011_v1
       --output ../../04_Results/P10_statistics_20261011_v1
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from statistical_core import (bootstrap_mean, bootstrap_patient_auc, paired_contrast,
                              person_event_interval, holm_family)

SCIENCE = Path(__file__).resolve().parents[1] / 'physiological_tool_selection'
sys.path.insert(0, str(SCIENCE))
from pmtools.protocol import FrozenDesign, sha256
from pmtools.experiment_io import write_json, write_csv

TASKS = ('af_afl', 'hr_reliability', 'stress', 'continuous_monitoring')
FAMILIES = ('classical', 'cnn')
SOURCES = ('VITALDB', 'PPGSPORT', 'GALAXYPPG', 'Stress-Predict', 'WearableDevice_V1', 'WearableDevice_V2', 'SHDB-AF')
STRATEGIES = ('attempt', 'quality', 'fixed_fusion', 'adaptive')
COLORS = dict(attempt='#777777', quality='#0072B2', fixed_fusion='#E69F00', adaptive='#009E73', fixed='#0072B2')
LABELS = dict(attempt='Primary attempt', quality='Quality + conformal', fixed_fusion='Fixed secondary',
              adaptive='Adaptive', fixed='Fixed')
SHORT = {'VITALDB':'AF/AFL (VitalDB)', 'PPGSPORT':'HR (PPG-Sport)', 'GALAXYPPG':'HR (GalaxyPPG)',
         'Stress-Predict':'Stress-Predict', 'WearableDevice_V1':'Stress protocol V1', 'WearableDevice_V2':'Stress protocol V2',
         'SHDB-AF':'Continuous ECG (SHDB-AF)'}


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv(path):
    return pd.read_csv(path, dtype={'subject':str,'record':str,'source':str,'role':str,'wrist':str},low_memory=False)


def check_inputs(root):
    manifest=[]
    for task in TASKS:
        for family in FAMILIES:
            run=root/(task+'_'+family+'_full')
            config=read_json(run/'run_config.json')
            assert config['preset']=='full' and config['version']=='P9_full_protocol_p6_v1'
            assert not config['evaluation_used_for_fit_or_tuning']
            for item in read_json(run/'result_manifest.json')['files']:
                file=run/item['file']
                assert file.stat().st_size==item['bytes'] and sha256(file)==item['sha256'], str(file)
                manifest.append(dict(path=str(file.relative_to(root)),bytes=item['bytes'],sha256=item['sha256']))
    return manifest


def plot_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
        'savefig.dpi':300,'axes.grid':True,'grid.alpha':.15,'figure.constrained_layout.use':True})


def save_figure(fig, output, name, datafile, explanation, index):
    fig.savefig(output/'figures'/(name+'.png'),dpi=300)
    fig.savefig(output/'figures'/(name+'.svg'))
    plt.close(fig)
    index.append(dict(figure=name,source_table=datafile,caption_en=explanation,
                      PNG_dpi=300,vector_format='SVG',result_version=output.name,input_result_version='P9_full_20261011_v1'))


def support(root, design, out):
    people=[]; windows=[]; labels=[]
    # Count a single model/strategy/budget only. Repetitions are never new people.
    for task in TASKS:
        run=root/(task+'_classical_full')
        flow=csv(run/'contribution_flow.csv')
        if task=='continuous_monitoring':
            flow['source']='SHDB-AF'
            for (source,role),g in flow.groupby(['source','role']):
                people.append(dict(task=task,source=source,role=role,independent_source_IDs=g.subject.nunique(),
                    records=len(g),wrist_windows=int(g.complete_10s_blocks.sum()),
                    unique_person_time_windows=int(g.complete_10s_blocks.sum()),
                    valid_signal_hours=g.valid_signal_seconds.sum()/3600,
                    record_exposure_hours=g.record_exposure_seconds.sum()/3600,
                    negative_valid_hours=g.known_negative_valid_seconds.sum()/3600,
                    unknown_reference_hours=g.unknown_reference_seconds.sum()/3600,
                    qualifying_onsets=int(g.qualifying_onset_events.sum())))
            selections=csv(run/'selection_before_training.csv')
            for (role,y),g in selections.groupby(['role','reference_label_offline_only']):
                labels.append(dict(task=task,source='SHDB-AF',role=role,label=y,
                    label_definition='complete_native_10s_reference_target_0_1_or_unknown_minus1',
                    contributing_people=g.subject.nunique(),windows=len(g)))
        else:
            for (source,role),g in flow.groupby(['source','role']):
                people.append(dict(task=task,source=source,role=role,independent_source_IDs=g.subject.nunique(),
                    records=int(g.records.sum()),wrist_windows=int(g.wrist_windows.sum()),
                    unique_person_time_windows=int(g.unique_person_time_windows.sum()),
                    technical_unavailable_windows=int(g.technical_unavailable_windows.sum())))
            pred=csv(run/'predictions.csv');g=pred[(pred.strategy=='attempt')&(pred.budget_hundredths==115)]
            for (source,role,y),part in g.groupby(['source','role','label_primary']):
                labels.append(dict(task=task,source=source,role=role,label=y,
                    label_definition='AF_AFL_target' if task=='af_afl' else 'stress_state' if task=='stress' else 'PPG_error_gt10_or_unestimable',
                    contributing_people=part.subject.nunique(),windows=len(part)))
        for row in flow.to_dict('records'): windows.append(dict(task=task,**row))
    write_csv(out/'tables'/'cohort_distribution.csv',people)
    write_csv(out/'tables'/'individual_contribution_flow.csv',windows)
    write_csv(out/'tables'/'label_distribution.csv',labels)
    allocation=[]
    for subject,p in sorted(design.people.items()):
        allocation.append(dict(subject=subject,task=p['task'],source=p['source'],role=p['role'],
            label=p.get('label'),training_CV_fold=p.get('training_cv_fold'),source_subject=p['source_subject']))
    write_csv(out/'tables'/'frozen_person_roles.csv',allocation)
    # Qualification flow is separate from realized experimental support.
    qualification=[dict(source='VITALDB',qualified_people=161,main_evaluation_people=60,details='95 AF/AFL and 66 annotated sinus; evaluation 30 per class'),
        dict(source='PPGSPORT',qualified_people=29,main_evaluation_people=29,details='two wrists within each person; not 58 independent people'),
        dict(source='GALAXYPPG',qualified_people=23,main_evaluation_people=23,details='separate-source validation; not individually >=30'),
        dict(source='Stress-Predict',qualified_people=30,main_evaluation_people=30,details='same people contribute stress and non-stress'),
        dict(source='WearableDevice_V1',qualified_people=18,main_evaluation_people=0,details='supplementary separate protocol'),
        dict(source='WearableDevice_V2',qualified_people=17,main_evaluation_people=0,details='supplementary separate protocol'),
        dict(source='SHDB-AF',qualified_people=54,main_evaluation_people=30,details='60 acquired records; duplicate/conflicting 005/020 excluded; 58 qualified records')]
    write_csv(out/'tables'/'qualification_flow.csv',qualification)
    return pd.DataFrame(people)


CLASS_METRICS=('mean_decision_loss','report_fraction','selective_risk','dangerous_acceptance_all_windows',
    'mean_cost','secondary_call_fraction','set_coverage','retained_MAE','all_estimable_PPG_MAE','PPG_estimability',
    'primary_classifier_accuracy','AUROC','primary_classifier_AUROC')
MON_METRICS=('miss_penalized_delay','detection_rate','detected_only_delay','FAR_per_non_target_hour',
    'negative_alarm_occupancy','non_target_positive_decision_fraction','report_fraction','selective_risk',
    'set_coverage','AUROC','normalized_cost_per_valid_hour','heavy_calls_per_valid_hour','p95_CPU_milliseconds',
    'unknown_exposure_seconds','technical_missing_seconds','no_decision_seconds')


def infer(root,out):
    inference=[]; contrasts=[]; differences=[]; person_tables=[]; zero_rows=[]; strata=[]
    for task in TASKS:
        for family in FAMILIES:
            run=root/(task+'_'+family+'_full');p=csv(run/'person_metrics.csv')
            p['task']=task;p['model']=family
            if task=='continuous_monitoring':p['source']='SHDB-AF';p['budget_hundredths']=0
            person_tables.append(p)
            for (source,strategy,budget),g in p.groupby(['source','strategy','budget_hundredths'],sort=True):
                g=g.sort_values('subject')
                y=g.label.to_numpy(dtype=int) if task=='af_afl' else None
                for metric in MON_METRICS if task=='continuous_monitoring' else CLASS_METRICS:
                    if metric not in g:continue
                    if metric=='AUROC' and task=='af_afl':
                        r=bootstrap_patient_auc(y,g.final_mean_probability.to_numpy(float))
                        unit='patient_mean_probability_AUROC'
                    else:
                        r=bootstrap_mean(g[metric].to_numpy(float),y)
                        unit='person_macro' if metric!='p95_CPU_milliseconds' else 'mean_of_person_p95_not_pooled_p95'
                    inference.append(dict(task=task,model=family,source=source,strategy=strategy,
                        budget_hundredths=int(budget),metric=metric,aggregation=unit,
                        exploratory=family=='cnn' or source.startswith('WearableDevice'),**r))
            specs=[('SHDB-AF','adaptive','fixed','miss_penalized_delay','MONITOR')] if task=='continuous_monitoring' else (
                [(s,'quality','attempt','mean_decision_loss','HR_SPORT' if s=='PPGSPORT' else 'HR_GALAXY') for s in ('PPGSPORT','GALAXYPPG')]
                if task=='hr_reliability' else [('VITALDB' if task=='af_afl' else 'Stress-Predict','adaptive','quality','mean_decision_loss','AF' if task=='af_afl' else 'STRESS')])
            budget=0 if task=='continuous_monitoring' else 270 if task=='stress' else 420
            old=read_json(run/'paired_primary_contrasts.json');old=[old] if isinstance(old,dict) else old
            for source,left,right,metric,cid in specs:
                a=p[(p.source==source)&(p.strategy==left)&(p.budget_hundredths==budget)].set_index('subject').sort_index()
                b=p[(p.source==source)&(p.strategy==right)&(p.budget_hundredths==budget)].set_index('subject').sort_index()
                assert a.index.equals(b.index)
                y=a.label.to_numpy(int) if task=='af_afl' else None
                r,d=paired_contrast(a[metric].to_numpy(float),b[metric].to_numpy(float),y)
                prior=next(x for x in old if x.get('source',source)==source)
                assert abs(prior['estimate']-r['estimate'])<1e-12
                assert np.allclose(prior['CI95'],[r['CI95_low'],r['CI95_high']],atol=1e-12)
                contrasts.append(dict(id=cid,model=family,source=source,metric=metric,contrast=left+' minus '+right,
                    budget_hundredths=budget,exploratory=family=='cnn',negative_difference_is_better=True,**r))
                for sid,x,va,vb in zip(a.index,d,a[metric],b[metric]):
                    differences.append(dict(id=cid,model=family,subject=sid,left=va,right=vb,difference=x,label=design_label(sid)))
            if task!='continuous_monitoring':
                pred=csv(run/'predictions.csv')
                main=pred[pred.budget_hundredths==budget]
                for (source,strategy),g in main.groupby(['source','strategy']):
                    inds=g.assign(bad=(~g.abstain)&g.wrong_if_reported).groupby('subject').bad.any().astype(int)
                    zero_rows.append(dict(task=task,model=family,source=source,strategy=strategy,
                        endpoint='at_least_one_wrong_accepted_window',**person_event_interval(inds)))
                # Activities have separate within-person support; the same person may occur in multiple strata.
                for (source,activity,strategy),g in main.groupby(['source','activity','strategy'],dropna=False):
                    pp=[]
                    for sid,v in g.groupby('subject'):
                        a=(~v.abstain).astype(float);bad=a*v.wrong_if_reported
                        if source=='PPGSPORT':
                            tmp=pd.DataFrame({'record':v.record,'start':v.start_seconds,'end':v.end_seconds,
                                'a':a,'bad':bad,'cost':v.normalized_cost,'call':v.secondary_called.astype(float)})
                            avg=tmp.groupby(['record','start','end'])[['a','bad','cost','call']].mean()
                        else:avg=pd.DataFrame({'a':a,'bad':bad,'cost':v.normalized_cost,'call':v.secondary_called.astype(float)})
                        report=avg.a.mean();danger=avg.bad.mean()
                        pp.append(dict(subject=sid,report_fraction=report,dangerous_acceptance_all_windows=danger,
                            mean_decision_loss=danger+.2*(1-report),selective_risk=danger/report if report else np.nan,
                            mean_cost=avg.cost.mean(),secondary_call_fraction=avg['call'].mean()))
                    for metric in ('report_fraction','dangerous_acceptance_all_windows','mean_decision_loss','selective_risk','mean_cost','secondary_call_fraction'):
                        rr=bootstrap_mean([v[metric] for v in pp])
                        strata.append(dict(task=task,model=family,source=source,activity=activity,strategy=strategy,
                            metric=metric,exploratory=True,**rr))
            else:
                for strategy,g in p.groupby('strategy'):
                    for event,flags in [('at_least_one_false_alarm',g.false_alarms>0),('at_least_one_missed_onset',g.missed>0)]:
                        zero_rows.append(dict(task=task,model=family,source='SHDB-AF',strategy=strategy,
                            endpoint=event,**person_event_interval(flags)))
            print('STATISTICS_COMPLETE',task,family,flush=True)
    classical=[r for r in contrasts if r['model']=='classical']
    assert len(classical)==5
    for r,adj in zip(classical,holm_family([r['sign_p_raw'] for r in classical])):
        r.update(sign_p_Holm_family_of_five=float(adj),inference_family='five_prespecified_classical_comparisons')
    for r in contrasts:
        if r['model']=='cnn':r.update(sign_p_Holm_family_of_five=None,inference_family='exploratory_not_in_primary_Holm_family')
    write_csv(out/'tables'/'endpoint_cluster_CI.csv',inference)
    write_csv(out/'tables'/'primary_paired_contrasts.csv',contrasts)
    write_csv(out/'tables'/'person_paired_differences.csv',differences)
    write_csv(out/'tables'/'person_error_event_bounds.csv',zero_rows)
    write_csv(out/'tables'/'activity_protocol_strata.csv',strata)
    pd.concat(person_tables,ignore_index=True).to_csv(out/'tables'/'all_person_endpoints.csv',index=False)
    return pd.DataFrame(inference),pd.DataFrame(contrasts),pd.concat(person_tables,ignore_index=True)


DESIGN=None
def design_label(subject):return DESIGN.people[subject].get('label')


def panel_curves(ax,g,metric):
    for strategy in STRATEGIES:
        use=g[(g.strategy==strategy)&(g.metric==metric)].sort_values('budget_hundredths')
        if use.empty:continue
        ax.plot(use.budget_hundredths/100,use.estimate,marker='o',ms=3,color=COLORS[strategy],label=LABELS[strategy])
        ax.fill_between(use.budget_hundredths/100,use.CI95_low,use.CI95_high,color=COLORS[strategy],alpha=.08)
    ax.set_xlabel('Available normalized budget')


def figures(out,support_rows,endpoints,contrasts,people,root):
    index=[];plot_style()
    fig,ax=plt.subplots(figsize=(10,4.8));source_order=list(dict.fromkeys(support_rows.source))
    pivot=support_rows.pivot(index='source',columns='role',values='independent_source_IDs').reindex(source_order).fillna(0)
    bottom=np.zeros(len(pivot))
    for role,color in [('train','#4477AA'),('policy','#EEAA33'),('conformal','#66CCAA'),('evaluation','#228833'),('supplementary_evaluation','#AA3377')]:
        if role not in pivot:continue
        vals=pivot[role].to_numpy();ax.barh(np.arange(len(pivot)),vals,left=bottom,label=role,color=color)
        for i,(v,b) in enumerate(zip(vals,bottom)):
            if v:ax.text(b+v/2,i,str(int(v)),ha='center',va='center',fontsize=8,color='white' if role!='policy' else 'black')
        bottom+=vals
    ax.set_yticks(np.arange(len(pivot)),pivot.index);ax.invert_yaxis();ax.set_xlabel('Independent source participant IDs')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18));
    save_figure(fig,out,'fig01_cohort_allocation','cohort_distribution.csv',
        'Frozen disjoint role counts. Wrists, records, windows and task states do not increase participant counts. Cross-source identity non-overlap remains an assumption.',index)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5));g=support_rows[support_rows.role.isin(['evaluation','supplementary_evaluation'])]
    for ax,key,title in zip(axes,['independent_source_IDs','unique_person_time_windows'],['Independent evaluation IDs','Unique person-time windows']):
        ax.barh([SHORT.get(x,x) for x in g.source],g[key],color='#4477AA');ax.set_xlabel(title);ax.invert_yaxis()
        if key.endswith('windows'):ax.set_xscale('log')
    save_figure(fig,out,'fig02_evaluation_support','cohort_distribution.csv',
        'Evaluation participant counts and unique source-time blocks are different quantities. Right panel is logarithmic; continuous monitoring includes the full native clock, not just pure-label decisions.',index)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[2.8,1]})
    for ax,ids in zip(axes,[['AF','STRESS','HR_SPORT','HR_GALAXY'],['MONITOR']]):
        for j,cid in enumerate(ids):
            for family,shift,color in [('classical',-.1,'#0072B2'),('cnn',.1,'#D55E00')]:
                r=contrasts[(contrasts.id==cid)&(contrasts.model==family)].iloc[0]
                ax.plot([r.CI95_low,r.CI95_high],[j+shift]*2,color=color,lw=2)
                ax.scatter([r.estimate],[j+shift],color=color,s=25,label=family if j==0 else None)
        ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks(range(len(ids)),ids);ax.invert_yaxis()
        ax.set_xlabel('Paired decision-loss difference' if len(ids)>1 else 'Paired delay difference (s)')
        ax.legend();ax.set_title('Negative favors first strategy')
    save_figure(fig,out,'fig03_primary_paired_effects','primary_paired_contrasts.csv',
        'Five frozen contrasts with 10,000 paired person-cluster percentile intervals. AF/AFL resampling is class stratified. Delay uses a separate seconds axis. Classical comparisons share one exploratory sign-test Holm family; CNN is exploratory.',index)
    main_sources=['VITALDB','Stress-Predict','PPGSPORT','GALAXYPPG']
    for metric,name,ylabel in [('mean_decision_loss','fig04_loss_budget','Person-macro decision loss'),
            ('report_fraction','fig05_report_budget','Person-macro report fraction'),
            ('mean_cost','fig06_cost_budget','Person-macro normalized cost'),
            ('set_coverage','fig07_set_coverage_budget','Pre-quality set coverage')]:
        fig,axes=plt.subplots(2,4,figsize=(13.5,6))
        for i,family in enumerate(FAMILIES):
            for j,source in enumerate(main_sources):
                ax=axes[i,j];g=endpoints[(endpoints.model==family)&(endpoints.source==source)]
                panel_curves(ax,g,metric);ax.set_title(SHORT[source]+' / '+family);ax.set_ylabel(ylabel if j==0 else '')
                if metric in ('report_fraction','set_coverage'):ax.set_ylim(-.02,1.02)
                if metric=='report_fraction':ax.axhline(.2,color='black',ls=':',lw=.7)
                if metric=='set_coverage':ax.axhline(.9,color='black',ls=':',lw=.7)
        axes[0,0].legend(fontsize=7)
        save_figure(fig,out,name,'endpoint_cluster_CI.csv',
            ylabel+' across all frozen budgets. Shading denotes person-cluster 95% CI. Repeated or overlapping strategies are retained. Set coverage is empirical under dependence/domain shift, not a guaranteed selected or patient-level coverage. HR secondary paths are offline reference simulations.',index)
    fig,axes=plt.subplots(2,4,figsize=(13.5,6))
    for i,family in enumerate(FAMILIES):
        for j,source in enumerate(main_sources):
            ax=axes[i,j];g=endpoints[(endpoints.model==family)&(endpoints.source==source)]
            for strategy in STRATEGIES:
                rr=g[(g.strategy==strategy)&(g.metric=='report_fraction')].set_index('budget_hundredths').sort_index()
                risk=g[(g.strategy==strategy)&(g.metric=='selective_risk')].set_index('budget_hundredths').reindex(rr.index)
                ax.plot(rr.estimate,risk.estimate,'o-',ms=3,color=COLORS[strategy],label=LABELS[strategy])
                if risk.estimate.isna().all():ax.plot([],[],color=COLORS[strategy],label=LABELS[strategy]+' (risk NA)')
            ax.set_title(SHORT[source]+' / '+family);ax.set_xlabel('Report fraction');ax.set_ylabel('Selective risk' if j==0 else '')
            ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
    axes[0,0].legend(fontsize=7)
    save_figure(fig,out,'fig08_risk_report_operating_points','endpoint_cluster_CI.csv',
        'Risk-report operating points for the frozen budget sweep; lines guide the eye and are not a tuned continuous frontier. No-report risk is undefined and omitted, not displayed as zero.',index)
    fig,axes=plt.subplots(2,4,figsize=(13.5,6))
    for i,family in enumerate(FAMILIES):
        for j,source in enumerate(main_sources):
            ax=axes[i,j];budget=270 if source=='Stress-Predict' else 420
            g=people[(people.model==family)&(people.source==source)&(people.budget_hundredths==budget)]
            for k,strategy in enumerate(STRATEGIES):
                vals=g[g.strategy==strategy].mean_decision_loss.to_numpy(float)
                ax.scatter(k+np.linspace(-.12,.12,len(vals)),vals,s=8,alpha=.5,color=COLORS[strategy])
                if len(vals):ax.plot([k-.18,k+.18],[np.mean(vals)]*2,color='black')
            ax.set_xticks(range(4),['Attempt','Gate','Fixed','Adaptive'],rotation=35);ax.set_title(SHORT[source]+' / '+family)
            ax.set_ylabel('Individual decision loss' if j==0 else '')
    save_figure(fig,out,'fig09_person_error_distributions','all_person_endpoints.csv',
        'All independent-person decision losses at the main budget. Each point is one person; black segments show means. Technical failures and all-abstain people remain included.',index)
    fig,axes=plt.subplots(1,3,figsize=(11.5,4))
    for j,(source,n) in enumerate([('Stress-Predict',30),('WearableDevice_V1',18),('WearableDevice_V2',17)]):
        ax=axes[j]
        for i,family in enumerate(FAMILIES):
            g=endpoints[(endpoints.source==source)&(endpoints.model==family)&(endpoints.budget_hundredths==270)&(endpoints.metric=='mean_decision_loss')]
            x=np.arange(4)+(i-.5)*.22;g=g.set_index('strategy').reindex(STRATEGIES)
            ax.plot(x,g.estimate,'o-',label=family,color=['#0072B2','#D55E00'][i])
        ax.set_xticks(range(4),['Attempt','Gate','Fixed','Adaptive'],rotation=35);ax.set_title(SHORT[source]+f' (n={n})');ax.set_ylabel('Decision loss')
    axes[0].legend()
    save_figure(fig,out,'fig10_stress_protocols','endpoint_cluster_CI.csv',
        'Main Stress-Predict and two supplementary protocols are reported separately, not pooled as one validation group. Protocol shifts may reverse apparent benefit.',index)
    fig,ax=plt.subplots(figsize=(5,4.5));roc_rows=[]
    for family,color in zip(FAMILIES,['#0072B2','#D55E00']):
        g=people[(people.source=='VITALDB')&(people.model==family)&(people.strategy=='attempt')&(people.budget_hundredths==420)]
        good=g.primary_mean_probability.notna();fpr,tpr,thresholds=roc_curve(g.loc[good,'label'],g.loc[good,'primary_mean_probability'])
        ax.plot(fpr,tpr,color=color,label=family)
        roc_rows.extend(dict(model=family,FPR=a,TPR=b,threshold=c) for a,b,c in zip(fpr,tpr,thresholds))
    ax.plot([0,1],[0,1],'--',color='#777777');ax.set_xlabel('False-positive fraction');ax.set_ylabel('True-positive fraction');ax.legend();ax.set_title('Primary PPG patient-level discrimination')
    write_csv(out/'tables'/'patient_ROC_points.csv',roc_rows)
    save_figure(fig,out,'fig11_AF_patient_ROC','patient_ROC_points.csv',
        'Patient-level primary PPG scores are means of saved window probabilities. This perioperative finger-PPG AF/AFL task is not pure-AF or wrist-device external validation.',index)
    mon=people[people.source=='SHDB-AF'];fig,axes=plt.subplots(2,3,figsize=(12,6))
    for i,family in enumerate(FAMILIES):
        for j,(metric,ylabel) in enumerate([('miss_penalized_delay','Miss-penalized delay (s)'),('FAR_per_non_target_hour','False alarms / negative hour'),('detection_rate','Onset detection fraction')]):
            ax=axes[i,j]
            for k,strategy in enumerate(('fixed','adaptive')):
                vals=mon[(mon.model==family)&(mon.strategy==strategy)][metric].to_numpy(float)
                ax.scatter(k+np.linspace(-.1,.1,len(vals)),vals,s=10,alpha=.6,color=COLORS[strategy])
                ax.plot([k-.15,k+.15],[np.nanmean(vals)]*2,color='black')
            ax.set_xticks([0,1],['Fixed','Adaptive']);ax.set_ylabel(ylabel);ax.set_title(family)
    save_figure(fig,out,'fig12_monitor_person_endpoints','all_person_endpoints.csv',
        'Complete-record monitoring person endpoints. Misses receive 120 seconds; detected-only latency is unavailable when none are detected. FAR denominator includes abstained negative exposure. Zero alarms cannot establish effective monitoring.',index)
    fig,axes=plt.subplots(1,3,figsize=(11.5,4))
    for j,(metric,ylabel) in enumerate([('normalized_cost_per_valid_hour','Normalized cost / valid hour'),('heavy_calls_per_valid_hour','Heavy calls / valid hour'),('report_fraction','Reported pure heavy-call fraction')]):
        ax=axes[j]
        for i,family in enumerate(FAMILIES):
            g=endpoints[(endpoints.source=='SHDB-AF')&(endpoints.model==family)&(endpoints.metric==metric)].set_index('strategy').reindex(['fixed','adaptive'])
            ax.plot(np.arange(2)+(i-.5)*.1,g.estimate,'o-',label=family,color=['#0072B2','#D55E00'][i])
        ax.set_xticks([0,1],['Fixed','Adaptive']);ax.set_ylabel(ylabel)
    axes[0].legend()
    save_figure(fig,out,'fig13_monitor_operations','endpoint_cluster_CI.csv',
        'Monitoring cost and heavy-call rates use valid full-clock hours; a 30-second heavy call costs three times a 10-second call. Report fraction uses scheduled pure-reference heavy decisions, not full-clock coverage.',index)
    # Activity error and invocation distributions, each panel avoids combining source protocols.
    activity=pd.read_csv(out/'tables'/'activity_protocol_strata.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,source in enumerate(('PPGSPORT','GALAXYPPG')):
        for j,metric in enumerate(('mean_decision_loss','secondary_call_fraction')):
            ax=axes[i,j]
            g=activity[(activity.source==source)&(activity.model=='classical')&(activity.metric==metric)]
            activities=sorted(g.activity.dropna().astype(str).unique());positions={x:k for k,x in enumerate(activities)}
            for strategy in STRATEGIES:
                v=g[g.strategy==strategy].copy();v['position']=v.activity.astype(str).map(positions);v=v.sort_values('position')
                ax.plot(v.position,v.estimate,'o-',label=LABELS[strategy],color=COLORS[strategy],ms=3)
            ax.set_xticks(range(len(activities)),activities,rotation=35);ax.set_title(source);ax.set_ylabel(metric.replace('_',' '));ax.set_xlabel('Source activity label')
    axes[0,0].legend(fontsize=7)
    save_figure(fig,out,'fig14_HR_activity_errors_calls','activity_protocol_strata.csv',
        'Exploratory activity-stratified HR error losses and offline reference-call rates. People may contribute to several activities; activity-specific contributor counts and CIs are in the source table. Activities from different datasets are not merged.',index)
    write_csv(out/'figure_captions.csv',index)
    return index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--p9-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.p9_root.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError('Never overwrite P9 or existing P10 results')
    manifest=check_inputs(root)
    global DESIGN;DESIGN=FrozenDesign()
    out.mkdir(parents=True);(out/'tables').mkdir();(out/'figures').mkdir()
    write_json(out/'analysis_config.json',dict(stage='P10',created_UTC=datetime.now(timezone.utc).isoformat(),
        input_version=root.name,protocol_sha256=sha256(SCIENCE/'config/protocol.json'),
        allocation_sha256=sha256(SCIENCE/'config/person_allocations.json'),input_manifest=manifest,
        statistics=DESIGN.protocol['statistics'],sensitivities=DESIGN.protocol['sensitivity'],
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        main_settings_unchanged=True,source_domains_separate=True,not_clinical_validation=True))
    s=support(root,DESIGN,out);e,c,p=infer(root,out);figures(out,s,e,c,p,root)
    check_inputs(root)
    write_json(out/'main_analysis_status.json',dict(main_statistics_and_figures_generated=True,
        sensitivity_complete=False,P10_complete=False,P11_started=False,figures=14,
        statistical_tests_are_exploratory_sign_tests=True,clinical_success_claim=False))
    print('P10_MAIN_GENERATED_SENSITIVITY_AND_FINAL_ACCEPTANCE_PENDING',out,flush=True)


if __name__=='__main__':main()
