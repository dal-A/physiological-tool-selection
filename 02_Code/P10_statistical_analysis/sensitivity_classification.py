"""One-factor sensitivity on full person support and frozen fit roles.

Saved P9 probabilities may be replayed only for unchanged models/inputs.
Changed HR targets are newly trained with full training-person CV, never relabeled
test scores from the old 10-bpm reliability classifier.
"""
from pathlib import Path
import ast, json, sys
import numpy as np
import pandas as pd
from statistical_core import bootstrap_mean, paired_contrast

SCIENCE=Path(__file__).resolve().parents[1]/'physiological_tool_selection'
sys.path.insert(0,str(SCIENCE))
from pmtools.protocol import FrozenDesign,sha256
from pmtools.adapters import DatasetAdapters
from pmtools.signals import quality,estimate_hr,UnavailableSignal
from pmtools.quick_classification import inputs
from pmtools.models import ClassicalModel,fit_cnn
from pmtools.full_training import persist_fit
from pmtools.safety import (Conformal,Planner,PrimaryState,SecondaryResult,decide,
                           quality_threshold,unreliable)
from pmtools.experiment_io import write_csv,write_json
from pmtools.full_classification import all_people

KEY=['subject','record','start_seconds','end_seconds','wrist']
BASE={'alpha':.1,'quality_quantile':.2,'planner_lambda':.02,'HR_error_threshold_bpm':10.,'dalia_reference':'center'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_csv(path):
    return pd.read_csv(path,dtype={'subject':str,'record':str,'wrist':str},low_memory=False)


def cached_observations(run):
    rows=[]
    for name in ('conformal_execution.csv','predictions.csv'):
        df=load_csv(run/name);budget=int(df.budget_hundredths.max())
        primary=df[(df.strategy=='quality')&(df.budget_hundredths==budget)].copy()
        secondary=df[(df.strategy=='fixed_fusion')&(df.budget_hundredths==budget)][KEY+['secondary_probability','secondary_quality']]
        assert not primary.duplicated(KEY).any() and not secondary.duplicated(KEY).any()
        primary=primary.drop(columns=['secondary_probability','secondary_quality']).merge(secondary,on=KEY,validate='one_to_one')
        assert len(primary)==len(secondary)
        rows.append(primary)
    return pd.concat(rows,ignore_index=True)


def training_quality(adapter,design,task):
    source='VITALDB' if task=='af_afl' else 'PPG-DaLiA' if task=='hr_reliability' else 'WESAD'
    primary=[];secondary=[];subjects=[]
    for person in design.select(source,'train'):
        for w in adapter.iter_person(person['subject']):
            primary.append(quality(w.signals['BVP'],'BVP',w.signals.get('ACC')));subjects.append(w.subject)
            if task!='hr_reliability':
                modality='ECG' if task=='af_afl' else 'EDA'
                secondary.append(quality(w.signals[modality],modality))
        print('SENS_TRAIN_QUALITY',task,person['subject'],flush=True)
    return primary,secondary,subjects


def restore_planner(policy,task,design,penalty):
    states=np.array([[np.nan if x is None else x for x in row] for row in policy['policy_states']])
    subjects=[row[0] for row in policy['policy_window_ids']]
    modality='REFERENCE' if task=='hr_reliability' else 'ECG' if task=='af_afl' else 'EDA'
    return Planner(modality,penalty).fit(states[:,0],states[:,1],policy['gain_before'],policy['gain_after'],subjects,design,policy['latency_ratio'])


def execute_cache(row,task,strategy,budget,planner,c,pq,sq,hr_threshold=10):
    p=row['primary_probability'];q=row['primary_quality']
    state=PrimaryState(float(p),float(q)) if pd.notna(p) else None
    hr=row.get('PPG_HR_bpm');hr=None if pd.isna(hr) else float(hr)
    ref=row.get('reference_HR_bpm');ref=None if pd.isna(ref) else float(ref)
    def secondary():
        if task=='hr_reliability':return SecondaryResult(0.,1.,reference_hr=ref)
        sp=row.get('secondary_probability');sqval=row.get('secondary_quality')
        return SecondaryResult(float(sp) if pd.notna(sp) else None,float(sqval) if pd.notna(sqval) else 0.,pd.notna(sp))
    decision=decide(state,pq,sq,{'primary':c,'fusion':c},budget,planner,secondary,strategy,hr,task=='hr_reliability')
    target=unreliable(hr,ref,hr_threshold) if task=='hr_reliability' else int(row['label_primary'])
    error=None if task!='hr_reliability' or decision.reported_hr is None else abs(decision.reported_hr-ref)
    wrong=(error is None or error>hr_threshold) if task=='hr_reliability' else decision.predicted!=target
    final_target=0 if task=='hr_reliability' and decision.secondary_called and decision.reported_hr is not None else target
    return dict(subject=row['subject'],source=row['source'],record=row['record'],start_seconds=row['start_seconds'],
        end_seconds=row['end_seconds'],wrist=row['wrist'],role=row['role'],activity=row.get('activity'),strategy=strategy,
        budget_hundredths=budget,label_primary=target,accepted=not decision.abstain,wrong=wrong,
        probability=decision.probability,predicted=decision.predicted,prediction_set=list(decision.prediction_set),
        report_fraction=float(not decision.abstain),dangerous_acceptance_all_windows=float(not decision.abstain and wrong),
        mean_decision_loss=float(wrong) if not decision.abstain else .2,
        mean_cost=decision.cost/100,secondary_call_fraction=float(decision.secondary_called),
        set_coverage=float(final_target in decision.prediction_set),retained_error=error,
        reason=decision.reason)


def person_summary(rows):
    df=pd.DataFrame(rows);output=[]
    for (source,strategy,subject),g in df.groupby(['source','strategy','subject']):
        if source=='PPGSPORT':
            x=g.groupby(['record','start_seconds','end_seconds'])[['report_fraction','dangerous_acceptance_all_windows',
                    'mean_decision_loss','mean_cost','secondary_call_fraction','set_coverage']].mean()
        else:x=g
        record=dict(source=source,strategy=strategy,subject=subject,label=g.label_primary.iloc[0] if source=='VITALDB' else None)
        for metric in ('report_fraction','dangerous_acceptance_all_windows','mean_decision_loss','mean_cost','secondary_call_fraction','set_coverage'):
            record[metric]=float(x[metric].mean())
        record['selective_risk']=record['dangerous_acceptance_all_windows']/record['report_fraction'] if record['report_fraction'] else None
        output.append(record)
    return pd.DataFrame(output)


def run_variant(observations,task,family,policy,design,training_q,variant,out):
    settings=dict(BASE,**variant);vq=settings['quality_quantile'];ps,ss,subjects=training_q
    pq=quality_threshold(ps,subjects,design,vq);sq=quality_threshold(ss,subjects,design,vq) if ss else 0.
    planner=restore_planner(policy,task,design,settings['planner_lambda'])
    budget=270 if task=='stress' else 420
    cal=observations[observations.role=='conformal'].to_dict('records')
    evaluation=observations[observations.role.isin(['evaluation','supplementary_evaluation'])].to_dict('records')
    allrows=[];cal_meta=[]
    variant_name='baseline' if not variant else next(iter(variant))+'_'+str(next(iter(variant.values())))
    targetdir=out/task/family/variant_name
    if targetdir.exists():raise FileExistsError(targetdir)
    targetdir.mkdir(parents=True)
    for strategy in ('attempt','quality','fixed_fusion','adaptive'):
        dummy=Conformal(settings['alpha'],strategy,budget).fit([],[],[],design)
        executed=[execute_cache(row,task,strategy,budget,planner,dummy,pq,sq,settings['HR_error_threshold_bpm']) for row in cal]
        good=[r for r in executed if r['probability'] is not None]
        # The corrected reference route has a reliable final-tool label.
        labels=[0 if task=='hr_reliability' and r['reason']=='offline_reference_simulation' else r['label_primary'] for r in good]
        c=Conformal(settings['alpha'],strategy,budget).fit([r['probability'] for r in good],labels,[r['subject'] for r in good],design)
        cal_meta.append(dict(strategy=strategy,alpha=c.alpha,rank=c.rank,quantile=c.quantile,fit_people=c.subjects,
            primary_quality_threshold=pq,secondary_quality_threshold=sq,scores=c.scores))
        result=[execute_cache(row,task,strategy,budget,planner,c,pq,sq,settings['HR_error_threshold_bpm']) for row in evaluation]
        allrows.extend(result)
    write_json(targetdir/'settings_and_calibration.json',dict(settings=settings,exploratory=True,
        no_evaluation_fitting=True,policy_fit_people=planner.fit_people,quality_fit_people=sorted(set(subjects)),
        calibrations=cal_meta,latency_ratio=policy['latency_ratio'],
        timings='Saved P9 policy latency ratio. Cached-probability sensitivity is not a new measured end-to-end CPU profile.'))
    write_csv(targetdir/'predictions.csv',allrows);p=person_summary(allrows);p.to_csv(targetdir/'person_metrics.csv',index=False)
    summary=[];contrast=[]
    for (source,strategy),g in p.groupby(['source','strategy']):
        y=g.label.to_numpy(int) if source=='VITALDB' else None
        for metric in ('mean_decision_loss','report_fraction','selective_risk','dangerous_acceptance_all_windows','mean_cost','secondary_call_fraction','set_coverage'):
            summary.append(dict(task=task,model=family,source=source,strategy=strategy,variant=variant_name,
                factor=next(iter(variant),'baseline'),factor_value=next(iter(variant.values()),'baseline'),
                metric=metric,exploratory=True,**bootstrap_mean(g[metric].to_numpy(float),y)))
    left,right=('quality','attempt') if task=='hr_reliability' else ('adaptive','quality')
    for source in p.source.unique():
        if source.startswith('WearableDevice'):continue
        a=p[(p.source==source)&(p.strategy==left)].set_index('subject').sort_index()
        b=p[(p.source==source)&(p.strategy==right)].set_index('subject').sort_index()
        assert a.index.equals(b.index)
        r,_=paired_contrast(a.mean_decision_loss,b.mean_decision_loss,a.label.to_numpy(int) if task=='af_afl' else None)
        contrast.append(dict(task=task,model=family,source=source,variant=variant_name,metric='mean_decision_loss',
            contrast=left+' minus '+right,exploratory=True,**r))
    write_csv(targetdir/'summary.csv',summary);write_csv(targetdir/'paired_contrasts.csv',contrast)
    print('SENS_VARIANT_COMPLETE',task,family,variant_name,flush=True)
    return summary,contrast


def hr_new_model_inputs(adapter,design,family,scratch):
    rows=[];arrays=[];training=[];qs=[];qsubjects=[]
    for person in all_people(design,'hr_reliability'):
        for w in adapter.iter_person(person['subject']):
            row=dict(subject=w.subject,source=w.source,record=w.record,role=w.role,start_seconds=w.start,end_seconds=w.end,
                wrist=w.wrist,activity=w.activity,reference_HR_bpm=w.reference_hr,
                wholly_inside_reference=w.metadata.get('reference_wholly_inside',w.reference_hr),
                PPG_HR_bpm=estimate_hr(w.signals['BVP']) if w.technical_valid else None,
                primary_quality=quality(w.signals['BVP'],'BVP',w.signals.get('ACC')),primary_probability=np.nan,
                secondary_probability=np.nan,secondary_quality=np.nan,input_index=None)
            if w.role=='train':qs.append(row['primary_quality']);qsubjects.append(w.subject)
            if w.technical_valid:
                try:row['input_index']=len(arrays);arrays.append(inputs(w.runtime(),'BVP',family))
                except UnavailableSignal:pass
            rows.append(row)
        print('SENS_HR_INPUT',family,person['subject'],flush=True)
    X=np.stack(arrays)
    # Feature/tensor cache is internal scratch, not redistributed raw sensor windows.
    np.savez_compressed(scratch/('hr_inputs_'+family+'.npz'),X=X)
    write_csv(scratch/('hr_input_rows_'+family+'.csv'),rows)
    return rows,X,qs,qsubjects


def train_hr_variant(rows,X,family,design,target_threshold,wholly,out,qs,qsubjects):
    name='dalia_reference_wholly_inside' if wholly else 'HR_error_threshold_bpm_'+str(target_threshold)
    directory=out/'hr_reliability'/family/(name+'_model')
    arrays=[];y=[];subjects=[];fit_rows=[]
    for row in rows:
        if row['role']!='train':continue
        ref=row['wholly_inside_reference'] if wholly else row['reference_HR_bpm']
        target=unreliable(row['PPG_HR_bpm'],ref,target_threshold)
        included=row['input_index'] is not None
        fit_rows.append(dict(subject=row['subject'],record=row['record'],start=row['start_seconds'],duration=10,
                             label=target,included_in_fit=included))
        if included:arrays.append(X[row['input_index']]);y.append(target);subjects.append(row['subject'])
    train=np.stack(arrays)
    model=(ClassicalModel().fit(train,y,subjects,design) if family=='classical' else
           fit_cnn(train,y,subjects,design,max_epochs=25,patience=5,modality='BVP'))
    model,_,_=persist_fit(model,train,fit_rows,qs,qsubjects,design,directory)
    write_json(directory/'sensitivity_target.json',dict(error_threshold_bpm=target_threshold,
        dalia_reference='wholly_inside' if wholly else 'center',new_training=True,role='train',
        no_evaluation_for_optimization=True,exploratory=True))
    probabilities=model.predict(X);observations=[];pol=[]
    for row in rows:
        r=dict(row);idx=r.pop('input_index');r['primary_probability']=float(probabilities[idx]) if idx is not None else np.nan
        if wholly and r['source']=='PPG-DaLiA':r['reference_HR_bpm']=r['wholly_inside_reference']
        observations.append(r)
        if r['role']=='policy':pol.append(r)
    return pd.DataFrame(observations),pol


def new_hr_policy(policy,rows,threshold):
    result=dict(policy)
    result['policy_states']=[[None if pd.isna(r['primary_probability']) else r['primary_probability'],r['primary_quality']] for r in rows]
    result['policy_window_ids']=[[r[k] for k in KEY] for r in rows]
    result['gain_before']=[unreliable(r['PPG_HR_bpm'],r['reference_HR_bpm'],threshold) if pd.notna(r['primary_probability']) else 1 for r in rows]
    result['gain_after']=[0 if pd.notna(r['primary_probability']) else 1 for r in rows]
    return result


def run_classification_sensitivity(root,data_root,out,scratch):
    design=FrozenDesign();adapter=DatasetAdapters(data_root,design=design);summary=[];contrasts=[]
    for task in ('af_afl','hr_reliability','stress'):
        q=training_quality(adapter,design,task)
        for family in ('classical','cnn'):
            run=root/(task+'_'+family+'_full');base=cached_observations(run);policy=read(run/'policy.json')
            # Train-only recomputation must reproduce the frozen quality threshold.
            assert abs(quality_threshold(q[0],q[2],design)-policy['quality_threshold_primary'])<1e-12
            if q[1]:assert abs(quality_threshold(q[1],q[2],design)-policy['quality_threshold_secondary'])<1e-12
            variants=[{}]+[{factor:v} for factor,values in [('alpha',[.05,.2]),('quality_quantile',[.1,.3]),('planner_lambda',[.005,.05])] for v in values]
            for variant in variants:
                s,c=run_variant(base,task,family,policy,design,q,variant,out);summary.extend(s);contrasts.extend(c)
                if not variant:
                    fresh=load_csv(out/task/family/'baseline'/'person_metrics.csv')
                    old=load_csv(run/'person_metrics.csv');budget=270 if task=='stress' else 420
                    old=old[old.budget_hundredths==budget].set_index(['subject','strategy']).sort_index()
                    fresh=fresh.set_index(['subject','strategy']).sort_index();assert fresh.index.equals(old.index)
                    for metric in ('mean_decision_loss','report_fraction','selective_risk','dangerous_acceptance_all_windows','mean_cost','secondary_call_fraction','set_coverage'):
                        assert np.allclose(fresh[metric],old[metric],atol=1e-12,equal_nan=True), (task,family,metric)
            if task=='hr_reliability':
                rows,X,qs,qsubjects=hr_new_model_inputs(adapter,design,family,scratch)
                for threshold,wholly in [(5.,False),(15.,False),(10.,True)]:
                    observations,pol=train_hr_variant(rows,X,family,design,threshold,wholly,out,qs,qsubjects)
                    variant={'dalia_reference':'wholly_inside'} if wholly else {'HR_error_threshold_bpm':threshold}
                    s,c=run_variant(observations,task,family,new_hr_policy(policy,pol,threshold),design,q,variant,out)
                    summary.extend(s);contrasts.extend(c)
                del X
    write_csv(out.parent/'tables'/'classification_sensitivity_CI.csv',summary)
    write_csv(out.parent/'tables'/'classification_sensitivity_contrasts.csv',contrasts)
    return summary,contrasts
