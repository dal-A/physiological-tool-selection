"""All frozen people/windows, disjoint policy/conformal pools, lazy full evaluation."""
from collections import defaultdict
import numpy as np
from .protocol import FrozenDesign
from .adapters import DatasetAdapters
from .quick_classification import (TASK_SOURCES, STRATEGIES, copy_window, Tools, execute,
                                   fitted_conformal, calibration_metadata, summarize)
from .full_io import create_full_run, seal_full_run, CsvSink, selection_row, flow_rows, full_figures
from .full_training import full_fit
from .profiling import profile_tool, latency_ratio
from .safety import Planner, Conformal, fuse, unreliable, person_diagnostic_conformal
from .statistics import balanced_person_mean, paired_bootstrap, safe_auroc
from .experiment_io import write_json, write_csv, window_key


def all_people(design,task):
    development,sources=TASK_SOURCES[task]
    result=[p for role in ('train','policy','conformal') for p in design.select(development,role)]
    for source in sources:
        role='supplementary_evaluation' if source.startswith('WearableDevice') else 'evaluation'
        result.extend(design.select(source,role))
    if len({p['subject'] for p in result})!=len(result): raise AssertionError('Duplicate frozen identity')
    return result


def load_full(adapter,design,task,path):
    windows=[]; expected=all_people(design,task)
    for p in expected:
        loaded=[copy_window(w) for w in adapter.iter_person(p['subject'])]
        if not loaded: raise AssertionError('Frozen person has no source windows: '+p['subject'])
        windows.extend(loaded)
        print('READ_FULL',task,p['role'],p['subject'],len(loaded),flush=True)
    actual={w.subject for w in windows}
    if actual!={p['subject'] for p in expected}: raise AssertionError('Full cohort changed')
    keys=[window_key(w) for w in windows]
    if len(set(keys))!=len(keys): raise AssertionError('Duplicate full window')
    for w in windows:
        if design.people[w.subject]['role']!=w.role: raise AssertionError('Role mismatch')
    write_json(path/'selection_before_training.json',[selection_row(w) for w in windows])
    write_csv(path/'contribution_flow.csv',flow_rows(windows))
    return windows


def paired_time_means(rows,values):
    bins=defaultdict(list)
    for r,v in zip(rows,values): bins[(r['record'],r['start_seconds'],r['end_seconds'])].append(v)
    return np.asarray([np.mean(v) for v in bins.values()],dtype=float)


def full_summary(rows,task):
    summary,persons=summarize(rows,task)
    # P8 summary is reused for its checked formulas, not its smoke qualification.
    summary.pop('not_full_or_clinical_validation',None)
    summary['not_clinical_validation']=True
    if rows[0]['source']=='PPGSPORT':
        for p in persons:
            g=[r for r in rows if r['subject']==p['subject']]
            a=np.asarray([not r['abstain'] for r in g],dtype=float)
            w=np.asarray([r['wrong_if_reported'] for r in g],dtype=float)
            report=float(paired_time_means(g,a).mean())
            bad=float(paired_time_means(g,a*w).mean())
            p.update(report_fraction=report,dangerous_acceptance_all_windows=bad,
                     selective_risk=bad/report if report else None)
            err=np.asarray([0 if r['post_final_absolute_error_bpm'] is None else r['post_final_absolute_error_bpm'] for r in g])
            p['retained_MAE']=float(paired_time_means(g,a*err).mean()/report) if report else None
            for field,key in [('normalized_cost','mean_cost'),('secondary_called','secondary_call_fraction'),('set_covered','set_coverage')]:
                p[key]=float(paired_time_means(g,[r[field] for r in g]).mean())
        for key in ('report_fraction','dangerous_acceptance_all_windows','selective_risk','retained_MAE',
                    'mean_cost','secondary_call_fraction','set_coverage'):
            values=[p[key] for p in persons if p[key] is not None]
            summary[key]=float(np.mean(values)) if values else None
            summary[key+'_contributing_people']=len(values)
        summary['abstention_fraction']=1-summary['report_fraction']
        summary['accuracy_reported']=1-summary['selective_risk'] if summary['selective_risk'] is not None else None
    final=[r for r in rows if r['probability'] is not None]
    summary['window_pooled_AUROC_diagnostic']=safe_auroc([r['label_final_tool'] for r in final],[r['probability'] for r in final]) if final else None
    summary['minimum_report_fraction_guard_pass']=summary['report_fraction']>=.2
    summary['eligible_unique_person_time_windows']=len({(r['subject'],r['record'],r['start_seconds'],r['end_seconds']) for r in rows})
    summary['physical_reference_acquisition_latency']=None
    return summary,persons


def primary_contrasts(person_rows,task,family):
    sources=TASK_SOURCES[task][1]
    budget=270 if task=='stress' else 420
    left,right=('quality','attempt') if task=='hr_reliability' else ('adaptive','quality')
    results=[]
    for source in sources:
        if source.startswith('WearableDevice'): continue
        parts={s:{r['subject']:r for r in person_rows if r['source']==source and r['strategy']==s and r['budget_hundredths']==budget} for s in (left,right)}
        if set(parts[left])!=set(parts[right]): raise AssertionError('Unpaired primary contrast')
        ids=sorted(parts[left]); labels=[parts[left][s]['label'] for s in ids] if task=='af_afl' else None
        result=paired_bootstrap([parts[left][s]['mean_decision_loss'] for s in ids],
                                [parts[right][s]['mean_decision_loss'] for s in ids],labels)
        results.append(dict(result,source=source,metric='mean_decision_loss',contrast=left+' minus '+right,
                            subjects=ids,model=family,exploratory=family=='cnn',
                            P10_familywise_inference_not_yet_performed=True))
    return results


def run_full_classification(data_root,output,task,family):
    design=FrozenDesign();adapter=DatasetAdapters(data_root,design=design)
    path=create_full_run(output,task,family)
    windows=load_full(adapter,design,task,path)
    roles={role:[w for w in windows if w.role==role] for role in ('train','policy','conformal')}
    evaluation=[w for w in windows if w.role in ('evaluation','supplementary_evaluation')]
    primary,pq,_=full_fit(roles['train'],'BVP',family,design,path/'models'/'primary',task=='hr_reliability')
    secondary=None;sq=0.
    if task!='hr_reliability':
        secondary,sq,_=full_fit(roles['train'],'ECG' if task=='af_afl' else 'EDA',family,design,path/'models'/'secondary')
    tools=Tools(primary,secondary,family,task);states=[];before=[];after=[]
    policy=roles['policy']
    for w in policy:
        state,hr=tools.primary(w.runtime())
        y=unreliable(hr,w.reference_hr) if task=='hr_reliability' else w.label
        if state is None:
            states.append((np.nan,np.nan));before.append(1);after.append(1);continue
        states.append((state.probability,state.quality))
        before.append(y if task=='hr_reliability' else int((state.probability>=.5)!=y))
        if task=='hr_reliability':after.append(0)
        else:
            secondary_result=tools.secondary(w.runtime())
            after.append(1 if not secondary_result.technical_valid or secondary_result.probability is None else
                         int((fuse(state.probability,secondary_result.probability,state.quality)>=.5)!=y))
    primary_profile=profile_tool(policy,lambda r:tools.primary(r,False),design)
    if task=='hr_reliability':
        secondary_profile={'physical_acquisition_latency':None,'ratio':1.,'conceptual_offline_reference_only':True};ratio=1.
    else:
        state_lookup={(w.subject,w.record,w.start,w.end):s for w,s in zip(policy,states)}
        def secondary_fusion(runtime):
            result=tools.secondary(runtime);p,q=state_lookup[(runtime.subject,runtime.record,runtime.start,runtime.end)]
            return fuse(p,result.probability,q) if result.technical_valid and result.probability is not None and np.isfinite(p) else None
        secondary_profile=profile_tool(policy,secondary_fusion,design)
        ratio=latency_ratio(primary_profile,secondary_profile)
    planner=Planner('REFERENCE' if task=='hr_reliability' else tools.secondary_modality).fit(
        np.asarray(states)[:,0],np.asarray(states)[:,1],before,after,[w.subject for w in policy],design,ratio)
    write_json(path/'policy.json',{'fit_people':planner.fit_people,'global_gain':planner.global_gain,'bins':planner.bins,
        'latency_ratio':ratio,'quality_threshold_primary':pq,'quality_threshold_secondary':sq,
        'quality_fit_people':sorted({w.subject for w in roles['train']}),
        'primary_profile':primary_profile,'secondary_plus_fusion_profile':secondary_profile,
        'gain_before':before,'gain_after':after,'policy_states':states,'policy_window_ids':[window_key(w) for w in policy]})
    budgets=[115,170,220,270] if task=='stress' else [115,200,300,420]
    calibrations=[];diagnostics=[];summaries=[];person_rows=[]
    with CsvSink(path/'predictions.csv') as prediction_sink,CsvSink(path/'conformal_execution.csv') as calibration_sink:
        for strategy in STRATEGIES:
            for budget in budgets:
                dummy=Conformal(strategy=strategy,budget=budget).fit([],[],[],design)
                rows=[execute(w,tools,strategy,budget,planner,dummy,pq,sq) for w in roles['conformal']]
                valid=[r for r in rows if r['probability'] is not None]
                c=fitted_conformal([r['probability'] for r in valid],[r['label_final_tool'] for r in valid],
                                   [r['subject'] for r in valid],design,strategy,budget)
                calibration_sink.add(rows);calibrations.append(calibration_metadata(c,rows))
                if valid:
                    diagnostic,ids=person_diagnostic_conformal([r['probability'] for r in valid],[r['label_final_tool'] for r in valid],
                        [r['subject'] for r in valid],[(r['record'],r['start_seconds'],r['wrist']) for r in valid],design)
                    diagnostics.append({'strategy':strategy,'budget_hundredths':budget,'people':len(ids),
                        'rank':diagnostic.rank,'quantile':diagnostic.quantile,
                        'selected_window_ids':[(valid[i]['subject'],valid[i]['record'],valid[i]['start_seconds'],valid[i]['wrist']) for i in ids],
                        'only_person_diagnostic_not_patient_safety_guarantee':True})
                for source in TASK_SOURCES[task][1]:
                    source_windows=[w for w in evaluation if w.source==source]
                    result=[]
                    for i,w in enumerate(source_windows):
                        result.append(execute(w,tools,strategy,budget,planner,c,pq,sq))
                        if (i+1)%5000==0:print('EVALUATE_FULL',task,family,source,strategy,budget,i+1,flush=True)
                    prediction_sink.add(result);summary,people=full_summary(result,task);summaries.append(summary)
                    person_rows.extend([dict(p,source=source,strategy=strategy,budget_hundredths=budget) for p in people])
                print('DECISIONS_FULL',task,family,strategy,budget,flush=True)
    write_json(path/'conformal.json',calibrations);write_json(path/'person_conformal_diagnostic.json',diagnostics)
    write_csv(path/'person_metrics.csv',person_rows)
    write_json(path/'paired_primary_contrasts.json',primary_contrasts(person_rows,task,family))
    write_json(path/'metrics.json',{'stage':'P9_full','summaries':summaries,
        'actual_role_people':{r:len({w.subject for w in windows if w.role==r}) for r in ('train','policy','conformal','evaluation','supplementary_evaluation')},
        'not_clinical_safety_proof':True,'physical_reference_acquisition_latency':None,
        'P10_statistics_sensitivity_and_familywise_audit_pending':True})
    full_figures(path,summaries);seal_full_run(path)
    return path
