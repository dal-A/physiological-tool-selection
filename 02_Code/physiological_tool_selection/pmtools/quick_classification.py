"""Reduced real-data end-to-end runs; no full-cohort inference or old models."""
from collections import defaultdict
from dataclasses import replace
from time import perf_counter_ns
import numpy as np
from .adapters import DatasetAdapters
from .schema import Signal
from .protocol import FrozenDesign
from .models import ClassicalModel, fit_cnn
from .signals import primary_features, cnn_input, eda_features, quality, estimate_hr, UnavailableSignal
from .safety import Conformal, Planner, PrimaryState, SecondaryResult, decide, fuse, unreliable, corrected_error, quality_threshold, person_diagnostic_conformal
from .persistence import save_model, load_model
from .profiling import profile_tool, latency_ratio
from .statistics import report_metrics, person_means, balanced_person_mean, safe_auroc
from .experiment_io import create_run, write_json, write_csv, window_key, rank_key, seal_results, diagnostic_figures

TASK_SOURCES={'af_afl':('VITALDB',['VITALDB']), 'hr_reliability':('PPG-DaLiA',['PPGSPORT','GALAXYPPG']),
              'stress':('WESAD',['Stress-Predict','WearableDevice_V1','WearableDevice_V2'])}
STRATEGIES=('attempt','quality','fixed_fusion','adaptive')


def quick_people(design,source,role):
    rows=design.select(source,role)
    if role=='train':
        if source!='VITALDB': return rows
        groups=defaultdict(list)
        for r in rows: groups[(r['training_cv_fold'],r['label'])].append(r)
        return [values[0] for _,values in sorted(groups.items())]
    if source=='VITALDB':
        return [r for label in (0,1) for r in [x for x in rows if x['label']==label][:2]]
    return rows[:3 if role!='supplementary_evaluation' else 2]


def copy_window(w):
    # Do not keep entire multi-hour source arrays alive via a short-window view.
    return replace(w,signals={k:Signal(np.array(v.values,copy=True),v.fs,v.unit) for k,v in w.signals.items()})


def selected_windows(adapter,person,maximum=48):
    groups=defaultdict(list)
    for w in adapter.iter_person(person):
        groups[(w.subject,w.record,w.start,w.end)].append(w)
    keys=sorted(groups,key=lambda k:rank_key(k))[:maximum]
    return [copy_window(w) for k in keys for w in groups[k]]


def inputs(runtime,modality,family):
    if modality=='EDA':
        if family=='classical': return eda_features(runtime.signals['EDA'])
        signal=runtime.signals['EDA']
        if signal.unit!='microSiemens' or not np.isfinite(signal.values).all():raise UnavailableSignal('Invalid EDA')
        return np.asarray(signal.values,dtype='float32').reshape(1,-1)  # Fit fold-specific EDA scale inside fit_cnn.
    signals={modality:runtime.signals[modality]}
    if modality=='BVP' and 'ACC' in runtime.signals: signals['ACC']=runtime.signals['ACC']
    return primary_features(signals,modality) if family=='classical' else cnn_input(signals,modality)


def fit_tool(windows,modality,family,design,directory,is_hr=False):
    arrays=[];labels=[];subjects=[];fit_rows=[];qualities=[];qpeople=[]
    for w in windows:
        runtime=w.runtime()
        q=quality(runtime.signals[modality],modality,runtime.signals.get('ACC') if modality=='BVP' else None)
        qualities.append(q);qpeople.append(w.subject)
        try:
            if not runtime.technical_valid:raise UnavailableSignal('Source technical unavailable')
            x=inputs(runtime,modality,family)
            target=unreliable(estimate_hr(runtime.signals['BVP']),w.reference_hr) if is_hr else w.label
            if target not in (0,1):raise UnavailableSignal('No pure training target')
            arrays.append(x);labels.append(target);subjects.append(w.subject)
            fit_rows.append({'subject':w.subject,'record':w.record,'start':w.start,'duration':w.end-w.start,'label':target,'included_in_fit':True})
        except UnavailableSignal as exc:
            fit_rows.append({'subject':w.subject,'record':w.record,'start':w.start,'duration':w.end-w.start,'included_in_fit':False,'reason':str(exc)})
    X=np.stack(arrays);y=np.array(labels,dtype=int)
    model=ClassicalModel().fit(X,y,subjects,design) if family=='classical' else fit_cnn(X,y,subjects,design,max_epochs=2,modality=modality)
    save_model(model,directory);reloaded=load_model(directory)
    before=model.predict(X[:8]);after=reloaded.predict(X[:8]);delta=float(np.max(np.abs(before-after)))
    if delta>1e-7:raise AssertionError('Saved-model prediction changed')
    write_csv(directory/'fit_windows.csv',fit_rows)
    write_json(directory/'roundtrip.json',{'maximum_probability_difference':delta,'checked_windows':len(before),'passed':True})
    return reloaded,quality_threshold(qualities,qpeople,design),fit_rows


class Tools:
    """Runtime interface sees no annotation/reference; reference oracle is a separate callback."""
    def __init__(self,primary,secondary,family,task):
        self.primary_model=primary;self.secondary_model=secondary;self.family=family;self.task=task
        self.secondary_modality='ECG' if task=='af_afl' else 'EDA'
    def primary(self,runtime,with_quality=True):
        if not runtime.technical_valid:return None,None
        try:
            x=inputs(runtime,'BVP',self.family);p=float(self.primary_model.predict(x[None])[0])
            q=quality(runtime.signals['BVP'],'BVP',runtime.signals.get('ACC')) if with_quality else 0.
            hr=estimate_hr(runtime.signals['BVP']) if self.task=='hr_reliability' else None
            return PrimaryState(p,q),hr
        except UnavailableSignal:return None,None
    def secondary(self,runtime):
        try:
            x=inputs(runtime,self.secondary_modality,self.family)
            p=float(self.secondary_model.predict(x[None])[0]);q=quality(runtime.signals[self.secondary_modality],self.secondary_modality)
            return SecondaryResult(p,q)
        except UnavailableSignal:return SecondaryResult(None,0.,False)


def execute(w,tools,strategy,budget,planner,calibrator,pq,sq):
    begin=perf_counter_ns();state,hr=tools.primary(w.runtime(),strategy!='attempt');calls=[];secondary_result=[]
    def secondary_callback():
        calls.append('REFERENCE' if tools.task=='hr_reliability' else tools.secondary_modality)
        result=SecondaryResult(0.,1.,reference_hr=w.reference_hr) if tools.task=='hr_reliability' else tools.secondary(w.runtime())
        secondary_result.append(result);return result
    decision=decide(state,pq,sq,{'primary':calibrator,'fusion':calibrator},budget,planner,secondary_callback,strategy,hr,tools.task=='hr_reliability')
    dt=(perf_counter_ns()-begin)/1e6
    if len(calls)!=int(decision.secondary_called) or decision.cost>budget:raise AssertionError('Lazy-call/cost contract failed')
    base_label=unreliable(hr,w.reference_hr) if tools.task=='hr_reliability' else w.label
    final_error=corrected_error(decision.reported_hr,w.reference_hr) if tools.task=='hr_reliability' else None
    final_label=0 if tools.task=='hr_reliability' and decision.secondary_called and decision.reported_hr is not None else base_label
    wrong=(final_error is None or final_error>10) if tools.task=='hr_reliability' else decision.predicted!=base_label
    return {'subject':w.subject,'source':w.source,'record':w.record,'start_seconds':w.start,'end_seconds':w.end,'role':w.role,
            'wrist':w.wrist,'activity':w.activity,'strategy':strategy,'budget':budget/100,'budget_hundredths':budget,
            'label_primary':base_label,'label_final_tool':final_label,'primary_probability':None if state is None else state.probability,
            'primary_quality':None if state is None else state.quality,'secondary_probability':secondary_result[0].probability if secondary_result else None,
            'secondary_quality':secondary_result[0].quality if secondary_result else None,'probability':decision.probability,
            'prediction_set':list(decision.prediction_set),'predicted':decision.predicted,'abstain':decision.abstain,
            'wrong_if_reported':bool(wrong),'set_covered':final_label in decision.prediction_set,
            'normalized_cost':decision.cost/100,'cost_hundredths':decision.cost,'secondary_called':decision.secondary_called,
            'runtime_secondary_accesses':len(calls),'tools_called':(['PPG'] if state is not None else [])+(['QUALITY'] if state is not None and strategy!='attempt' else [])+calls,
            'fusion_executed':bool(decision.secondary_called and tools.task!='hr_reliability' and secondary_result and secondary_result[0].technical_valid and secondary_result[0].probability is not None),
            'CPU_milliseconds':dt,'physical_reference_acquisition_latency':None,'PPG_HR_bpm':hr,
            'reference_HR_bpm':w.reference_hr,'reported_HR_bpm':decision.reported_hr,'post_final_absolute_error_bpm':final_error,
            'reason':decision.reason,'technical_valid':w.technical_valid}


def fitted_conformal(p,y,subjects,design,strategy,budget):
    return Conformal(strategy=strategy,budget=budget).fit(p,y,subjects,design)


def calibration_metadata(calibrator, rows):
    return {'key':calibrator.key,'alpha':calibrator.alpha,'rank':calibrator.rank,'quantile':calibrator.quantile,
            'fit_people':calibrator.subjects,'scores':calibrator.scores,'eligible_score_windows':len(calibrator.scores),
            'excluded_probability_unavailable':sum(r['probability'] is None for r in rows)}


def summarize(rows,task):
    people=[]
    for subject in sorted({r['subject'] for r in rows}):
        group=[r for r in rows if r['subject']==subject]
        a=np.array([not r['abstain'] for r in group]);wrong=np.array([r['wrong_if_reported'] for r in group])
        out=report_metrics(a,wrong,[np.nan if r['post_final_absolute_error_bpm'] is None else r['post_final_absolute_error_bpm'] for r in group] if task=='hr_reliability' else None)
        times=[(r['record'],r['start_seconds'],r['end_seconds']) for r in group]
        loss=np.where(a,wrong.astype(float),.2)
        out['mean_decision_loss']=person_means(loss,[subject]*len(group),times)[subject]
        out.update({'subject':subject,'label':group[0]['label_primary'] if task=='af_afl' else None,
                    'mean_cost':float(np.mean([r['normalized_cost'] for r in group])),
                    'secondary_call_fraction':float(np.mean([r['secondary_called'] for r in group])),
                    'set_coverage':float(np.mean([r['set_covered'] for r in group]))})
        good=[r for r in group if r['primary_probability'] is not None]
        out['primary_mean_probability']=float(np.mean([r['primary_probability'] for r in good])) if good else None
        out['primary_classifier_AUROC']=safe_auroc([r['label_primary'] for r in good],[r['primary_probability'] for r in good]) if good else None
        out['primary_classifier_accuracy']=float(np.mean([(r['primary_probability']>=.5)==r['label_primary'] for r in good])) if good else None
        final=[r for r in group if r['probability'] is not None]
        out['final_mean_probability']=float(np.mean([r['probability'] for r in final])) if final else None
        out['AUROC']=safe_auroc([r['label_final_tool'] for r in final],[r['probability'] for r in final]) if final else None
        if task=='hr_reliability':
            estimable=[abs(r['PPG_HR_bpm']-r['reference_HR_bpm']) for r in group if r['PPG_HR_bpm'] is not None]
            out['all_estimable_PPG_MAE']=float(np.mean(estimable)) if estimable else None
            out['PPG_estimability']=len(estimable)/len(group)
        people.append(out)
    keys=['report_fraction','selective_risk','dangerous_acceptance_all_windows','mean_decision_loss','mean_cost','secondary_call_fraction','set_coverage']
    summary={k:None for k in keys}
    for key in keys:
        available=[p for p in people if p[key] is not None]
        labels=[p['label'] for p in available] if task=='af_afl' else None
        if available and (labels is None or set(labels)=={0,1}):summary[key]=balanced_person_mean([p[key] for p in available],labels)
        summary[key+'_contributing_people']=len(available)
    if task=='af_afl':
        available=[p for p in people if p['final_mean_probability'] is not None]
        summary['AUROC']=safe_auroc([p['label'] for p in available],[p['final_mean_probability'] for p in available]) if available else None
        summary['AUROC_unit']='person_mean_probability'
    else:
        values=[p['AUROC'] for p in people if p['AUROC'] is not None]
        summary['AUROC']=float(np.mean(values)) if values else None
        summary['AUROC_contributing_people']=len(values);summary['AUROC_unit']='mean_defined_within_person_executed_tool_AUROC'
    summary['accuracy_reported']=None if summary['selective_risk'] is None else 1-summary['selective_risk']
    summary['abstention_fraction']=1-summary['report_fraction'] if summary['report_fraction'] is not None else None
    summary['primary_classifier_accuracy']=float(np.mean([p['primary_classifier_accuracy'] for p in people if p['primary_classifier_accuracy'] is not None])) if any(p['primary_classifier_accuracy'] is not None for p in people) else None
    defined=[p['primary_classifier_AUROC'] for p in people if p['primary_classifier_AUROC'] is not None]
    summary['primary_classifier_AUROC_mean_defined_people']=float(np.mean(defined)) if defined else None
    if task=='hr_reliability':
        for key in ['retained_MAE','all_estimable_PPG_MAE','PPG_estimability']:
            values=[p[key] for p in people if p.get(key) is not None];summary[key]=float(np.mean(values)) if values else None
            summary[key+'_contributing_people']=len(values)
    summary.update({'source':rows[0]['source'],'strategy':rows[0]['strategy'],'budget':rows[0]['budget'],
                    'independent_source_IDs':len(people),'eligible_windows':len(rows),'reported_windows':sum(not r['abstain'] for r in rows),
                    'p95_CPU_milliseconds':float(np.quantile([r['CPU_milliseconds'] for r in rows],.95)),
                    'selective_risk_undefined_reason':('zero_reported_windows' if all(r['abstain'] for r in rows) else 'no_reported_person_in_one_class_stratum') if summary['selective_risk'] is None else None,
                    'AUROC_undefined_reason':'insufficient_two_class_probability_support' if summary['AUROC'] is None else None,
                    'aggregation':'person_macro; VitalDB class-balanced; Sport same-time wrists paired',
                    'not_full_or_clinical_validation':True})
    return summary,people


def run_classification(data_root,output,task,family):
    design=FrozenDesign();adapter=DatasetAdapters(data_root,design=design);development,evaluation_sources=TASK_SOURCES[task]
    path=create_run(output,task,family,{'maximum_time_windows_per_person':48,'training_people':'all development train except VitalDB one per class/fold',
                                    'policy_conformal_people':'first three frozen ranks, VitalDB two per class','evaluation_people':'first three per source; supplementary first two',
                                    'CNN_max_epochs':2,'purpose':'pipeline smoke only','budgets_hundredths':[115,170,220,270] if task=='stress' else [115,200,300,420]})
    windows=[]
    for role in ('train','policy','conformal'):
        for p in quick_people(design,development,role):
            windows.extend(selected_windows(adapter,p['subject']))
            print('READ',task,family,role,p['subject'],flush=True)
    for source in evaluation_sources:
        role='supplementary_evaluation' if source.startswith('WearableDevice') else 'evaluation'
        for p in quick_people(design,source,role):windows.extend(selected_windows(adapter,p['subject']))
    selection=[{'subject':w.subject,'source':w.source,'role':w.role,'record':w.record,'start':w.start,'end':w.end,'wrist':w.wrist} for w in windows]
    write_json(path/'selection_before_training.json',selection)
    train=[w for w in windows if w.role=='train'];policy=[w for w in windows if w.role=='policy'];calibration=[w for w in windows if w.role=='conformal'];evaluation=[w for w in windows if w.role in ('evaluation','supplementary_evaluation')]
    for a in ('train','policy','conformal','evaluation','supplementary_evaluation'):
        for b in ('train','policy','conformal','evaluation','supplementary_evaluation'):
            if a!=b and {w.subject for w in windows if w.role==a}&{w.subject for w in windows if w.role==b}:raise AssertionError('Role overlap')
    primary,pq,_=fit_tool(train,'BVP',family,design,path/'models'/'primary',task=='hr_reliability')
    secondary=None;sq=0.
    if task!='hr_reliability':secondary,sq,_=fit_tool(train,'ECG' if task=='af_afl' else 'EDA',family,design,path/'models'/'secondary')
    tools=Tools(primary,secondary,family,task);states=[];gain_before=[];gain_after=[]
    for w in policy:
        state,hr=tools.primary(w.runtime());y=unreliable(hr,w.reference_hr) if task=='hr_reliability' else w.label
        if state is None: states.append((np.nan,np.nan));gain_before.append(1);gain_after.append(1);continue
        states.append((state.probability,state.quality));gain_before.append(y if task=='hr_reliability' else int((state.probability>=.5)!=y))
        if task=='hr_reliability': gain_after.append(0)
        else:
            result=tools.secondary(w.runtime());gain_after.append(1 if not result.technical_valid else int((fuse(state.probability,result.probability,state.quality)>=.5)!=y))
    primary_profile=profile_tool(policy,lambda r:tools.primary(r,False),design)
    if task=='hr_reliability': secondary_profile={'physical_acquisition_latency':None,'ratio':1,'conceptual_offline_reference_only':True};ratio=1.
    else:
        bykey={(w.subject,w.record,w.start,w.end):(p,q) for w,(p,q) in zip(policy,states)}
        def secondary_plus_fusion(runtime):
            result=tools.secondary(runtime);p,q=bykey[(runtime.subject,runtime.record,runtime.start,runtime.end)]
            return fuse(p,result.probability,q) if result.technical_valid and np.isfinite(p) else None
        secondary_profile=profile_tool(policy,secondary_plus_fusion,design);ratio=latency_ratio(primary_profile,secondary_profile)
    planner=Planner('REFERENCE' if task=='hr_reliability' else tools.secondary_modality).fit(np.array(states)[:,0],np.array(states)[:,1],gain_before,gain_after,[w.subject for w in policy],design,ratio)
    write_json(path/'policy.json',{'fit_people':planner.fit_people,'global_gain':planner.global_gain,'bins':planner.bins,'latency_ratio':ratio,
                                 'quality_threshold_primary':pq,'quality_threshold_secondary':sq,'quality_fit_people':sorted({w.subject for w in train}),
                                 'primary_profile':primary_profile,'secondary_plus_fusion_profile':secondary_profile,'gain_before':gain_before,'gain_after':gain_after,
                                 'policy_states':states,'policy_window_ids':[window_key(w) for w in policy]})
    predictions=[];calibration_rows=[];calibration_records=[];diagnostics=[]
    budgets=[115,170,220,270] if task=='stress' else [115,200,300,420]
    for strategy in STRATEGIES:
        for budget in budgets:
            dummy=Conformal(strategy=strategy,budget=budget).fit([],[],[],design)
            rows=[execute(w,tools,strategy,budget,planner,dummy,pq,sq) for w in calibration]
            valid=[r for r in rows if r['probability'] is not None]
            c=fitted_conformal([r['probability'] for r in valid],[r['label_final_tool'] for r in valid],[r['subject'] for r in valid],design,strategy,budget)
            for r in rows:r['stage']='conformal_execution_for_scores'
            calibration_rows.extend(rows);calibration_records.append(calibration_metadata(c,rows))
            if valid:
                diagnostic,indices=person_diagnostic_conformal([r['probability'] for r in valid],[r['label_final_tool'] for r in valid],[r['subject'] for r in valid],[(r['record'],r['start_seconds'],r['wrist']) for r in valid],design)
                diagnostics.append({'strategy':strategy,'budget_hundredths':budget,'people':len(indices),'rank':diagnostic.rank,'quantile':diagnostic.quantile,'selected_indices':indices,'only_person_diagnostic_not_patient_safety_guarantee':True})
            for w in evaluation:predictions.append(execute(w,tools,strategy,budget,planner,c,pq,sq))
            print('DECISIONS',task,family,strategy,budget,flush=True)
    write_csv(path/'predictions.csv',predictions);write_csv(path/'conformal_execution.csv',calibration_rows)
    write_json(path/'conformal.json',calibration_records);write_json(path/'person_conformal_diagnostic.json',diagnostics)
    summaries=[];person_rows=[]
    for source in evaluation_sources:
        for strategy in STRATEGIES:
            for budget in budgets:
                rows=[r for r in predictions if r['source']==source and r['strategy']==strategy and r['budget_hundredths']==budget]
                summary,persons=summarize(rows,task);summaries.append(summary)
                person_rows.extend([dict(p,source=source,strategy=strategy,budget_hundredths=budget) for p in persons])
    write_json(path/'metrics.json',{'stage':'P8_quick','not_formal_performance_evidence':True,'summaries':summaries,'selected_role_counts':{r:len({w.subject for w in windows if w.role==r}) for r in ('train','policy','conformal','evaluation','supplementary_evaluation')},'physical_reference_acquisition_latency':None})
    write_csv(path/'person_metrics.csv',person_rows);diagnostic_figures(path,summaries,predictions)
    seal_results(path);return path
