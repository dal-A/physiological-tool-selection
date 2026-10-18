"""Complete-record causal full monitoring, independent offline references and person clusters."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import numpy as np
from .protocol import FrozenDesign
from .adapters import DatasetAdapters, truth
from .schema import Signal, contiguous_triples
from .signals import quality, light_features, UnavailableSignal
from .models import ClassicalModel
from .persistence import save_model, load_model
from .full_training import full_fit
from .quick_classification import copy_window, inputs
from .quick_monitor import stream, score_final
from .full_io import create_full_run, seal_full_run, CsvSink, selection_row, full_figures
from .experiment_io import write_csv, write_json
from .safety import quality_threshold, Conformal, person_diagnostic_conformal
from .profiling import profile_tool
from .monitoring import (duration, intersect_intervals, union_intervals, calibrate_alarm,
                         final_alarm, alarm_episodes, PolicyMonitorRecord)
from .statistics import safe_auroc, paired_bootstrap


class ReferenceIntervals:
    """Offline interval integral; unknown/mixed targets remain None, never runtime state."""
    def __init__(self,table):
        self.spans={}
        for target in (0,1):
            intervals=union_intervals([(float(r.start_seconds),float(r.end_seconds))
                       for r in table.itertuples() if r.target_label==target])
            a=np.asarray(intervals,dtype=float).reshape(-1,2)
            self.spans[target]=(a[:,0],a[:,1],np.r_[0.,np.cumsum(a[:,1]-a[:,0])])
    @staticmethod
    def integral(x,spans):
        starts,ends,sums=spans
        count=int(np.searchsorted(ends,x,side='right'))
        total=float(sums[count])
        if count<len(starts):total+=max(0.,min(x,float(ends[count]))-float(starts[count]))
        return total
    def target(self,start,end):
        for target,spans in self.spans.items():
            if abs(self.integral(end,spans)-self.integral(start,spans)-(end-start))<1e-7:
                return target
        return None


def references(adapter,design,path):
    table=adapter.table('P5','candidate_windows.csv'); refs={};flow=[]
    all_events=adapter.table('P5','target_episodes.csv')
    with CsvSink(path/'selection_before_training.csv') as selection:
        for person in design.select('SHDB-AF'):
            intervals,events=adapter.monitor_reference(person['subject'])
            for record in sorted(person['records']):
                rows=table[(table.subject==person['subject']) & (table.record==record)]
                if rows.empty: raise AssertionError('Frozen monitoring record absent')
                start=float(rows.start_seconds.min());end=float(rows.end_seconds.max())
                valid=union_intervals([(float(r.start_seconds),float(r.end_seconds)) for r in rows.itertuples() if truth(r.signal_valid)])
                timeline=intervals[intervals.record==record].copy()
                negative=intersect_intervals([(float(r.start_seconds),float(r.end_seconds)) for r in timeline.itertuples() if r.target_label==0],[(start,end)])
                unknown=intersect_intervals([(float(r.start_seconds),float(r.end_seconds)) for r in timeline.itertuples() if r.target_label==-1],[(start,end)])
                eligible=[(float(r.start_seconds),float(r.end_seconds)) for r in events[events.record==record].itertuples()]
                censor=all_events[(all_events.subject==person['subject']) & (all_events.record==record)].to_dict('records')
                ref={'subject':person['subject'],'record':record,'role':person['role'],
                     'start':start,'end':end,'valid':valid,'negative':negative,'unknown':unknown,
                     'events':eligible,'censoring':censor,'interval_table':timeline,'window_count':len(rows)}
                refs[(person['subject'],record)]=ref
                selection.add([{'subject':person['subject'],'record':record,'role':person['role'],
                    'start_seconds':float(r.start_seconds),'end_seconds':float(r.end_seconds),
                    'signal_valid':truth(r.signal_valid),'reference_label_offline_only':int(r.target_label)} for r in rows.itertuples()])
                flow.append({'subject':person['subject'],'record':record,'role':person['role'],
                    'complete_10s_blocks':len(rows),'valid_signal_seconds':duration(valid),
                    'record_exposure_seconds':end-start,'unknown_reference_seconds':duration(unknown),
                    'known_negative_valid_seconds':duration(intersect_intervals(negative,valid)),
                    'qualifying_onset_events':len(eligible)})
    write_csv(path/'contribution_flow.csv',flow)
    write_json(path/'scoring_intervals.json',[{k:v for k,v in r.items() if k!='interval_table'} for r in refs.values()])
    return refs


def person_windows(adapter,design,role):
    for person in design.select('SHDB-AF',role):
        print('STREAM_TRAIN_SOURCE',role,person['subject'],flush=True)
        yield from adapter.iter_person(person['subject'])


def record_windows(adapter,subject,record):
    # Complete signal time axis. Do NOT use rhythm labels to filter/reset this iterator.
    for w in adapter.iter_person(subject):
        if w.record==record:yield w
        elif w.record>record:break


def thirty_contexts(windows):
    """Label-independent nonoverlapping quality/profiling contexts, including unknown rhythm."""
    buf=[]
    for w in windows:
        if buf and (w.subject!=buf[-1].subject or w.record!=buf[-1].record or w.start!=buf[-1].end):buf=[]
        if not w.technical_valid:buf=[];continue
        buf.append(w)
        if len(buf)==3:
            native=buf[0].signals['ECG']
            yield replace(buf[0],end=w.end,label=None,metadata={},signals={
                'ECG':Signal(np.concatenate([x.signals['ECG'].values for x in buf]),native.fs,native.unit)})
            buf=[]


def profile_selection(windows,limit=20):
    """Same policy hash selection as profile_tool, held in O(people*20) signal memory."""
    groups=defaultdict(list)
    for w in windows:
        key=(w.subject,w.record,w.start,w.end,w.wrist)
        rank=hashlib.sha256(('20260921|profile|'+repr(key)).encode()).hexdigest()
        values=groups[w.subject]
        if len(values)<limit or rank<values[-1][0]:
            values.append((rank,copy_window(w)));values.sort(key=lambda item:item[0]);del values[limit:]
    return [w for subject in sorted(groups) for _,w in groups[subject]]


def annotate(calls,reference):
    lookup=ReferenceIntervals(reference['interval_table'])
    return [dict(r,label=lookup.target(r['start_seconds'],r['end_seconds'])) for r in calls]


def full_policy_records(calls_by,refs):
    # A scheduled call has a technically valid signal context. Tool failure is a
    # zero positive, not permission to shrink the known-negative denominator.
    # NaN remains a missing probability, excluded from threshold candidates.
    records=[]
    for (subject,record,strategy),calls in calls_by.items():
        ref=refs[(subject,record)]
        if ref['role']!='policy':continue
        rows=tuple((r['decision_seconds'],np.nan if r['probability'] is None else r['probability'],
                    r['quality_pass'] and r['probability'] is not None,r['label']==0) for r in calls)
        resets=tuple(r['decision_seconds'] for r in calls if r['probability'] is None)
        records.append(PolicyMonitorRecord(subject,record,strategy,rows,tuple(ref['events']),
                       tuple(ref['negative']),tuple(ref['valid']),ref['start'],ref['end'],resets))
    return records


def run_record(adapter,ref,light,heavy,family,thresholds,strategy):
    count=[0]
    def blocks():
        for w in record_windows(adapter,ref['subject'],ref['record']):
            count[0]+=1
            yield w.runtime()
    calls,ticks,resets=stream(blocks(),light,heavy,family,thresholds,strategy)
    if count[0]!=ref['window_count']:raise AssertionError('Full stream skipped or duplicated blocks')
    print('RECORD_FULL',family,ref['role'],ref['subject'],ref['record'],strategy,
          'ticks',count[0],'heavy_calls',len(calls),flush=True)
    return annotate(calls,ref),ticks,resets


def aggregate_records(scores,predictions,ticks):
    """Score each native record first, then combine all records of ONE participant."""
    groups=defaultdict(list)
    for row in scores:groups[(row['subject'],row['strategy'])].append(row)
    output=[]
    by_calls=defaultdict(list);by_ticks=defaultdict(list)
    for r in predictions:by_calls[(r['subject'],r['strategy'])].append(r)
    for r in ticks:by_ticks[(r['subject'],r['strategy'])].append(r)
    for key,records in sorted(groups.items()):
        calls=by_calls[key];runtime=by_ticks[key]
        matches=[m for r in records for m in r['matches']]
        events=len(matches);detected=[m['delay'] for m in matches if m['detected']]
        negative_hours=sum(r['negative_hours'] for r in records)
        valid_hours=sum(r['valid_signal_seconds'] for r in records)/3600
        false=sum(r['false_alarms'] for r in records)
        pure=[r for r in calls if r['label'] in (0,1)]
        accepted=[r for r in pure if not r['abstain']]
        finite=[r for r in pure if r['probability'] is not None]
        negative=[r for r in pure if r['label']==0 and r['unavailable_reason'] is None]
        cost=sum(r['light_cost']+r['heavy_cost'] for r in runtime)
        p={'subject':key[0],'strategy':key[1],'records':len(records),'events':events,
           'detected':len(detected),'missed':events-len(detected),
           'detection_rate':len(detected)/events if events else None,
           'miss_penalized_delay':float(np.mean([m['delay'] for m in matches])) if events else None,
           'detected_only_delay':float(np.mean(detected)) if detected else None,
           'detected_only_delay_undefined_reason':None if detected else 'no_detected_onset',
           'negative_hours':negative_hours,'false_alarms':false,
           'FAR_per_non_target_hour':false/negative_hours if negative_hours else None,
           'FAR_undefined_reason':None if negative_hours else 'zero_known_negative_valid_exposure',
           'negative_alarm_occupancy':sum(r['negative_alarm_seconds'] for r in records)/(negative_hours*3600) if negative_hours else None,
           'non_target_positive_decision_fraction':sum(r['active_alarm'] for r in negative)/len(negative) if negative else None,
           'known_negative_decision_denominator':len(negative),
           'eligible_pure_heavy_decisions':len(pure),'reported_heavy_decisions':len(accepted),
           'report_fraction':len(accepted)/len(pure) if pure else None,
           'selective_risk':float(np.mean([r['predicted']!=r['label'] for r in accepted])) if accepted else None,
           'selective_risk_undefined_reason':None if accepted else 'zero_reported_decisions',
           'set_coverage':float(np.mean([r['set_covered'] for r in pure])) if pure else None,
           'AUROC':safe_auroc([r['label'] for r in finite],[r['probability'] for r in finite]) if finite else None,
           'valid_signal_hours':valid_hours,'normalized_cost':cost,
           'normalized_cost_per_valid_hour':cost/valid_hours if valid_hours else None,
           'heavy_calls':len(calls),'heavy_calls_per_valid_hour':len(calls)/valid_hours if valid_hours else None,
           'p95_CPU_milliseconds':float(np.quantile([r['CPU_milliseconds'] for r in calls],.95)) if calls else None,
           'complete_10s_ticks':len(runtime),
           'no_accepted_heavy_decision':not any(not r['abstain'] for r in calls),
           'no_decision_seconds':sum(r['no_decision_seconds'] for r in records),
           'unknown_exposure_seconds':sum(r['unknown_exposure_seconds'] for r in records),
           'technical_missing_seconds':sum(r['technical_missing_seconds'] for r in records)}
        output.append(p)
    return output


def score_record(calls,resets,reference,calibrators,threshold,strategy):
    predictions,score=score_final(calls,resets,reference,calibrators,threshold)
    score['strategy']=strategy
    score['valid_signal_seconds']=duration(reference['valid'])
    score['negative_alarm_seconds']=duration(intersect_intervals(score['alarm_spans'],
                                    intersect_intervals(reference['negative'],reference['valid'])))
    accepted_decisions=[(r['decision_seconds'],not r['abstain']) for r in predictions]+[(t,False) for t in resets]
    _,accepted_spans=alarm_episodes(accepted_decisions,reference['start'],reference['end'])
    score['no_decision_seconds']=duration(reference['valid'])-duration(intersect_intervals(accepted_spans,reference['valid']))
    return predictions,score


def run_full_monitor(data_root,output,family):
    design=FrozenDesign();adapter=DatasetAdapters(data_root,design=design)
    path=create_full_run(output,'continuous_monitoring',family);refs=references(adapter,design,path)
    heavy10,q10,_=full_fit(person_windows(adapter,design,'train'),'ECG',family,design,path/'models'/'heavy10')
    heavy30,_,_=full_fit(contiguous_triples(person_windows(adapter,design,'train')),'ECG',family,design,path/'models'/'heavy30')
    q30values=[];q30people=[]
    for w in thirty_contexts(person_windows(adapter,design,'train')):
        q30values.append(quality(w.signals['ECG'],'ECG'));q30people.append(w.subject)
    q30=quality_threshold(q30values,q30people,design)
    light_x=[];light_y=[];light_people=[];light_rows=[]
    for w in person_windows(adapter,design,'train'):
        if w.technical_valid and w.label in (0,1):
            try:
                light_x.append(light_features(w.signals['ECG']));light_y.append(w.label);light_people.append(w.subject)
                light_rows.append({'subject':w.subject,'record':w.record,'start':w.start,'label':w.label})
            except UnavailableSignal:continue
    light=ClassicalModel().fit(np.stack(light_x),light_y,light_people,design)
    save_model(light,path/'models'/'light');reloaded=load_model(path/'models'/'light')
    if not np.array_equal(light.predict(np.stack(light_x[:32])),reloaded.predict(np.stack(light_x[:32]))):raise AssertionError('Light model reload mismatch')
    light=reloaded;del light_x
    write_csv(path/'models'/'light'/'fit_windows.csv',light_rows)
    write_json(path/'training_quality.json',{'10':q10,'30':q30,'fit_people':sorted({w['subject'] for w in light_rows}),
         '30s_quality_uses_label_independent_valid_contexts':True,'30s_quality_contexts':len(q30values)})
    policy10=profile_selection(person_windows(adapter,design,'policy'))
    policy30=profile_selection(thirty_contexts(person_windows(adapter,design,'policy')))
    profiles={'light10':profile_tool(policy10,lambda r:light.predict(light_features(r.signals['ECG'])[None]),design),
              'heavy10':profile_tool(policy10,lambda r:heavy10.predict(inputs(r,'ECG',family)[None]),design),
              'heavy30':profile_tool(policy30,lambda r:heavy30.predict(inputs(r,'ECG',family)[None]),design)}
    write_json(path/'policy_CPU_profiles.json',profiles)
    heavy={10:heavy10,30:heavy30};thresholds={10:q10,30:q30};calls_by={};resets_by={}
    # Policy completes BEFORE conformal streaming; never use conformal/test feedback to select theta.
    with CsvSink(path/'stream_ticks.csv') as tick_sink,CsvSink(path/'policy_execution.csv') as policy_sink:
        for ref in refs.values():
            if ref['role']!='policy':continue
            for strategy in ('fixed','adaptive'):
                calls,ticks,resets=run_record(adapter,ref,light,heavy,family,thresholds,strategy)
                key=(ref['subject'],ref['record'],strategy);calls_by[key]=calls;resets_by[key]=resets
                tick_sink.add([dict(r,role='policy') for r in ticks]);policy_sink.add(calls)
        records=full_policy_records(calls_by,refs)
        records=[replace(r,resets=tuple(sorted(set(r.resets+tuple(resets_by[(r.subject,r.record,r.strategy)]))))) for r in records]
        policy=calibrate_alarm(records,design);threshold=policy['selected']['threshold']
        write_json(path/'policy_alarm.json',policy)
        print('POLICY_FULL_FROZEN',family,threshold,'target_feasible',policy['selected']['target_feasible'],flush=True)
        calibration_calls=[]
        with CsvSink(path/'conformal_execution.csv') as cal_sink:
            for ref in refs.values():
                if ref['role']!='conformal':continue
                for strategy in ('fixed','adaptive'):
                    calls,ticks,resets=run_record(adapter,ref,light,heavy,family,thresholds,strategy)
                    cal_sink.add(calls);calibration_calls.extend(calls)
                    tick_sink.add([dict(r,role='conformal') for r in ticks])
        calibrators={};metadata=[];diagnostics=[]
        for strategy in ('fixed','adaptive'):
            for seconds in (10,30):
                rows=[r for r in calibration_calls if r['strategy']==strategy and r['duration_seconds']==seconds and r['label'] in (0,1) and r['probability'] is not None]
                c=Conformal(strategy=strategy,budget=0,duration=seconds).fit([r['probability'] for r in rows],
                          [r['label'] for r in rows],[r['subject'] for r in rows],design)
                calibrators[(strategy,seconds)]=c
                metadata.append({'key':c.key,'fit_people':c.subjects,'rank':c.rank,'quantile':c.quantile,
                     'scores':c.scores,'score_windows':len(rows),'no_observations_reason':'no_scheduled_pure_calls_for_duration' if not rows else None})
                if rows:
                    d,ids=person_diagnostic_conformal([r['probability'] for r in rows],[r['label'] for r in rows],
                        [r['subject'] for r in rows],[(r['record'],r['start_seconds'],seconds) for r in rows],design)
                    diagnostics.append({'strategy':strategy,'duration_seconds':seconds,'rank':d.rank,'quantile':d.quantile,
                        'fit_people':d.subjects,'selected_window_ids':[(rows[i]['subject'],rows[i]['record'],rows[i]['start_seconds']) for i in ids],
                        'diagnostic_only_not_patient_safety_guarantee':True})
        write_json(path/'conformal.json',metadata);write_json(path/'person_conformal_diagnostic.json',diagnostics)
        del calibration_calls
        policy_final=[]
        for key,calls in calls_by.items():
            ref=refs[key[:2]];annotated,score=score_record(calls,resets_by[key],ref,calibrators,threshold,key[2])
            for row in annotated:
                if row['active_alarm'] and not (row['quality_pass'] and row['probability']>=threshold):raise AssertionError('Final gate added preliminary positive')
            policy_final.append(score)
        write_json(path/'policy_final_scoring.json',policy_final);del calls_by,resets_by
        scores=[];person_rows=[];matches=[]
        with CsvSink(path/'predictions.csv') as prediction_sink,CsvSink(path/'event_matches.csv') as match_sink:
            for person in design.select('SHDB-AF','evaluation'):
                person_scores=[];person_predictions=[];person_ticks=[]
                for record in sorted(person['records']):
                    ref=refs[(person['subject'],record)]
                    for strategy in ('fixed','adaptive'):
                        calls,ticks,resets=run_record(adapter,ref,light,heavy,family,thresholds,strategy)
                        annotated,score=score_record(calls,resets,ref,calibrators,threshold,strategy)
                        prediction_sink.add(annotated);tick_sink.add([dict(r,role='evaluation') for r in ticks])
                        match_sink.add([dict(m,subject=person['subject'],record=record,strategy=strategy) for m in score['matches']])
                        person_scores.append(score);person_predictions.extend(annotated);person_ticks.extend(ticks)
                scores.extend(person_scores);person_rows.extend(aggregate_records(person_scores,person_predictions,person_ticks))
    write_json(path/'event_scores.json',scores)
    write_csv(path/'person_metrics.csv',person_rows)
    lookup={(r['subject'],r['strategy']):r for r in person_rows};ids=sorted({r['subject'] for r in person_rows})
    if set(ids)!={p['subject'] for p in design.select('SHDB-AF','evaluation')}:raise AssertionError('Evaluation people changed')
    contrast=paired_bootstrap([lookup[(s,'adaptive')]['miss_penalized_delay'] for s in ids],
                             [lookup[(s,'fixed')]['miss_penalized_delay'] for s in ids])
    write_json(path/'paired_primary_contrasts.json',dict(contrast,subjects=ids,contrast='adaptive minus fixed',
        metric='miss_penalized_delay',model=family,exploratory=family=='cnn',P10_familywise_inference_not_yet_performed=True))
    summaries=[]
    for strategy in ('fixed','adaptive'):
        rows=[r for r in person_rows if r['strategy']==strategy]
        out={'strategy':strategy,'independent_people':len(rows),'records':sum(r['records'] for r in rows),
             'qualifying_events':sum(r['events'] for r in rows),'detected_events':sum(r['detected'] for r in rows),
             'missed_events':sum(r['missed'] for r in rows),'negative_hours':sum(r['negative_hours'] for r in rows),
             'valid_signal_hours':sum(r['valid_signal_hours'] for r in rows)}
        for key in ('miss_penalized_delay','detected_only_delay','detection_rate','FAR_per_non_target_hour',
                    'negative_alarm_occupancy','report_fraction','selective_risk','set_coverage','AUROC',
                    'normalized_cost_per_valid_hour','heavy_calls_per_valid_hour'):
            values=[r[key] for r in rows if r[key] is not None]
            out[key]=float(np.mean(values)) if values else None;out[key+'_contributing_people']=len(values)
        out['FAR_pooled_exposure_diagnostic']=sum(r['false_alarms'] for r in rows)/out['negative_hours'] if out['negative_hours'] else None
        out['no_alarm_zero_FAR_not_success']=True;summaries.append(out)
    write_json(path/'metrics.json',{'stage':'P9_full','common_alarm_threshold':threshold,
        'policy_target_feasible':policy['selected']['target_feasible'],'summaries':summaries,
        'per_person':person_rows,'selected_source_people_by_role':{r:len(design.select('SHDB-AF',r)) for r in ('train','policy','conformal','evaluation')},
        'physical_acquisition_latency':None,'runtime_light_cost_per_10s':.15,
        'complete_record_followup':True,'no_truth_driven_runtime_reset':True,'not_clinical_safety_proof':True,
        'P10_statistics_sensitivity_and_familywise_audit_pending':True})
    full_figures(path,person_rows,True);seal_full_run(path);return path
