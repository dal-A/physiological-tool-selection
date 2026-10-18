"""Replay frozen complete monitoring clock; never reset using rhythm truth.

Alpha/quality/cap do not change the frozen light-driven sampling schedule.
Recorded probability and decision-time replay is exact for that fixed pipeline.
Event/horizon sensitivity is offline scoring only, not algorithm retuning.
"""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import sys,json
import numpy as np
import pandas as pd
from statistical_core import bootstrap_mean,paired_contrast
SCIENCE=Path(__file__).resolve().parents[1]/'physiological_tool_selection'
sys.path.insert(0,str(SCIENCE))
from pmtools.protocol import FrozenDesign
from pmtools.adapters import DatasetAdapters
from pmtools.signals import quality
from pmtools.safety import quality_threshold,Conformal
from pmtools.full_monitor import thirty_contexts,full_policy_records,score_record,aggregate_records
from pmtools.monitoring import calibrate_alarm,score_events
from pmtools.experiment_io import write_csv,write_json


def csv(path):return pd.read_csv(path,dtype={'subject':str,'record':str},low_memory=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def monitor_quality(data_root,design,out):
    adapter=DatasetAdapters(data_root,design=design);rows=[]
    for person in design.select('SHDB-AF','train'):
        def blocks():yield from adapter.iter_person(person['subject'])
        for w in blocks():
            rows.append(dict(subject=w.subject,record=w.record,start_seconds=w.start,end_seconds=w.end,
                             duration_seconds=10,quality=quality(w.signals['ECG'],'ECG')))
        for w in thirty_contexts(blocks()):
            rows.append(dict(subject=w.subject,record=w.record,start_seconds=w.start,end_seconds=w.end,
                             duration_seconds=30,quality=quality(w.signals['ECG'],'ECG')))
        print('SENS_MONITOR_TRAIN_QUALITY',person['subject'],len(rows),flush=True)
    write_csv(out/'monitor_training_quality.csv',rows)
    return pd.DataFrame(rows)


def changes_to_calls(frame,qth):
    df=frame.copy()
    df['quality_pass']=(df.quality>=df.duration_seconds.map(qth))&df.probability.notna()
    # CSV missing probabilities are unavailable tools, not zero probabilities.
    rows=df.to_dict('records')
    for row in rows:
        for key in ('probability','label','technical_failure_reason'):
            if key in row and pd.isna(row[key]):row[key]=None
    return rows


def grouped(rows):
    output=defaultdict(list)
    for row in rows:output[(row['subject'],row['record'],row['strategy'])].append(row)
    return output


def run_monitor_sensitivity(root,data_root,out,inputs_out):
    design=FrozenDesign();training=monitor_quality(data_root,design,inputs_out);summaries=[];contrasts=[];support=[]
    variants=[{}]+[{factor:value} for factor,values in [('alpha',[.05,.2]),('quality_quantile',[.1,.3]),
        ('monitor_policy_cap',[.01,.1]),('monitor_horizon_seconds',[60,300]),('monitor_event_duration_seconds',[60])] for value in values]
    for family in ('classical','cnn'):
        run=root/('continuous_monitoring_'+family+'_full')
        refs={(r['subject'],r['record']):r for r in read(run/'scoring_intervals.json')}
        ticks=csv(run/'stream_ticks.csv');eval_ticks=ticks[ticks.role=='evaluation'].to_dict('records')
        old_scores=read(run/'event_scores.json');policy_scores=read(run/'policy_final_scoring.json')
        reset_lookup={(r['subject'],r['record'],r['strategy']):r['reset_seconds'] for r in old_scores+policy_scores}
        policy=csv(run/'policy_execution.csv');cal=csv(run/'conformal_execution.csv');evaluation=csv(run/'predictions.csv')
        old_quality=read(run/'training_quality.json')
        for variant in variants:
            alpha=variant.get('alpha',.1);fraction=variant.get('quality_quantile',.2)
            cap=variant.get('monitor_policy_cap',.05);horizon=variant.get('monitor_horizon_seconds',120)
            min_duration=variant.get('monitor_event_duration_seconds',30)
            name='baseline' if not variant else next(iter(variant))+'_'+str(next(iter(variant.values())))
            directory=out/'continuous_monitoring'/family/name
            if directory.exists():raise FileExistsError(directory)
            directory.mkdir(parents=True)
            qth={}
            for seconds in (10,30):
                g=training[training.duration_seconds==seconds]
                qth[seconds]=quality_threshold(g.quality,g.subject,design,fraction)
                if fraction==.2:assert abs(qth[seconds]-old_quality[str(seconds)])<1e-12
            pcalls=grouped(changes_to_calls(policy,qth));records=full_policy_records(pcalls,refs)
            records=[replace(r,resets=tuple(reset_lookup[(r.subject,r.record,r.strategy)])) for r in records]
            # Threshold remains fitted on policy people. Changing event/horizon
            # is a scoring sensitivity, so it must not feed back into calibration.
            alarm=calibrate_alarm(records,design,cap=cap)
            theta=alarm['selected']['threshold']
            if not variant:
                prior=read(run/'policy_alarm.json');assert theta==prior['selected']['threshold']
            cal_rows=changes_to_calls(cal,qth);calibrators={};cal_metadata=[]
            for strategy in ('fixed','adaptive'):
                for seconds in (10,30):
                    g=[r for r in cal_rows if r['strategy']==strategy and r['duration_seconds']==seconds
                       and r['label'] in (0,1) and r['probability'] is not None]
                    c=Conformal(alpha,strategy,0,seconds).fit([r['probability'] for r in g],[r['label'] for r in g],[r['subject'] for r in g],design)
                    calibrators[(strategy,seconds)]=c
                    cal_metadata.append(dict(strategy=strategy,duration_seconds=seconds,alpha=alpha,rank=c.rank,
                                             quantile=c.quantile,fit_people=c.subjects,scores=c.scores))
            calls_by=grouped(changes_to_calls(evaluation,qth));scores=[];predictions=[];matched=[]
            for key,calls in calls_by.items():
                reference=dict(refs[key[:2]])
                reference['events']=[e for e in reference['events'] if e[1]-e[0]>=min_duration]
                annotated,score=score_record(calls,reset_lookup[key],reference,calibrators,theta,key[2])
                rescored=score_events(reference['events'],score['alarm_onsets'],reference['negative'],reference['valid'],
                                     score['alarm_spans'],horizon=horizon)
                score.update(rescored);scores.append(score);predictions.extend(annotated)
                matched.extend(dict(subject=key[0],record=key[1],strategy=key[2],**r) for r in rescored['matches'])
            persons=pd.DataFrame(aggregate_records(scores,predictions,eval_ticks))
            # No-onset people stay in support; their event-specific metric is NA.
            expected={p['subject'] for p in design.select('SHDB-AF','evaluation')}
            assert set(persons.subject)==expected
            if not variant:
                prior=csv(run/'person_metrics.csv').set_index(['subject','strategy']).sort_index()
                fresh=persons.set_index(['subject','strategy']).sort_index()
                for field in ('events','detected','missed','miss_penalized_delay','FAR_per_non_target_hour','report_fraction','set_coverage','normalized_cost_per_valid_hour'):
                    assert np.allclose(prior[field],fresh[field],atol=1e-10,equal_nan=True),(family,field)
            persons.to_csv(directory/'person_metrics.csv',index=False);write_csv(directory/'event_matches.csv',matched)
            write_json(directory/'settings_and_calibration.json',dict(alpha=alpha,quality_quantile=fraction,alarm_cap=cap,
                horizon_seconds=horizon,minimum_event_duration_seconds=min_duration,quality_thresholds=qth,
                policy=alarm,conformal=cal_metadata,schedule='unchanged frozen P9 complete clock',
                decision_latency='unchanged recorded per-call P9 latency',exploratory=True,no_evaluation_fitting=True,
                event_and_horizon_changes_are_scoring_only=True))
            write_json(directory/'event_scores.json',scores)
            for strategy,g in persons.groupby('strategy'):
                support.append(dict(model=family,variant=name,strategy=strategy,evaluation_people=g.subject.nunique(),
                    event_contributing_people=int((g.events>0).sum()),qualifying_events=int(g.events.sum()),
                    detected_events=int(g.detected.sum()),negative_valid_hours=float(g.negative_hours.sum()),
                    horizon_seconds=horizon,minimum_event_duration_seconds=min_duration))
                for metric in ('miss_penalized_delay','detection_rate','detected_only_delay','FAR_per_non_target_hour',
                    'report_fraction','set_coverage','selective_risk','normalized_cost_per_valid_hour','heavy_calls_per_valid_hour'):
                    r=bootstrap_mean(g[metric].to_numpy(float))
                    summaries.append(dict(task='continuous_monitoring',model=family,source='SHDB-AF',strategy=strategy,
                        variant=name,factor=next(iter(variant),'baseline'),factor_value=next(iter(variant.values()),'baseline'),
                        metric=metric,exploratory=True,**r))
            a=persons[persons.strategy=='adaptive'].set_index('subject').sort_index()
            b=persons[persons.strategy=='fixed'].set_index('subject').sort_index();assert a.index.equals(b.index)
            complete=a.miss_penalized_delay.notna()&b.miss_penalized_delay.notna()
            if complete.any():
                r,_=paired_contrast(a.loc[complete,'miss_penalized_delay'],b.loc[complete,'miss_penalized_delay'])
            else:r=dict(estimate=None,CI95_low=None,CI95_high=None,independent_people=0,contributing_people=0)
            contrasts.append(dict(task='continuous_monitoring',model=family,source='SHDB-AF',variant=name,
                contrast='adaptive minus fixed',metric='miss_penalized_delay',all_evaluation_people=len(a),
                event_contributing_people=int(complete.sum()),exploratory=True,**r))
            print('SENS_VARIANT_COMPLETE','continuous_monitoring',family,name,flush=True)
    write_csv(out.parent/'tables'/'monitor_sensitivity_CI.csv',summaries)
    write_csv(out.parent/'tables'/'monitor_sensitivity_contrasts.csv',contrasts)
    write_csv(out.parent/'tables'/'monitor_sensitivity_support.csv',support)
    return summaries,contrasts
