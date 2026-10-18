"""Same-person fixed/adaptive real-stream smoke evaluation; reference is offline."""
from dataclasses import replace
from time import perf_counter_ns
import numpy as np
from .adapters import DatasetAdapters
from .protocol import FrozenDesign
from .schema import Signal, RuntimeBlock, contiguous_triples
from .signals import quality, light_features, UnavailableSignal
from .models import ClassicalModel
from .profiling import profile_tool
from .statistics import safe_auroc
from .persistence import save_model, load_model
from .safety import Conformal, quality_threshold
from .monitoring import MonitorScheduler, PolicyMonitorRecord, calibrate_alarm, final_alarm, alarm_episodes, score_events, duration, intersect_intervals, union_intervals
from .quick_classification import inputs, fit_tool, copy_window
from .experiment_io import create_run, write_json, write_csv, seal_results, diagnostic_figures


def clip_intervals(intervals,start,end): return intersect_intervals(intervals,[(start,end)])


def context_target(intervals,start,end):
    for target in (0,1):
        spans=[(float(r.start_seconds),float(r.end_seconds)) for r in intervals.itertuples() if r.target_label==target]
        if abs(duration(clip_intervals(spans,start,end))-(end-start))<1e-7:return target
    return None


def select_clips(adapter,design):
    """Qualification-annotation-based smoke clips, never used as full efficacy evidence."""
    clips=[];selection=[];references={}
    for role in ('train','policy','conformal','evaluation'):
        people=design.select('SHDB-AF',role)
        if role!='train':people=people[:3]
        for person in people:
            intervals,events=adapter.monitor_reference(person['subject'])
            event=events.sort_values(['duration_seconds','record','start_seconds'],ascending=[False,True,True]).iloc[0]
            start=max(0.,float(np.floor((event.start_seconds-180)/10)*10));end=float(np.ceil((event.start_seconds+180)/10)*10)
            record=str(event.record);windows=[]
            iterator=adapter.iter_person(person['subject'])
            try:
                for w in iterator:
                    if w.record!=record:continue
                    if w.start>=end:break
                    if w.start>=start and w.end<=end:windows.append(copy_window(w))
            finally:iterator.close()
            if not windows:raise ValueError('Selected smoke clip has no complete blocks')
            end=windows[-1].end;start=windows[0].start
            rows=intervals[intervals.record==record].copy()
            negative=clip_intervals([(float(r.start_seconds),float(r.end_seconds)) for r in rows.itertuples() if r.target_label==0],start,end)
            unknown=clip_intervals([(float(r.start_seconds),float(r.end_seconds)) for r in rows.itertuples() if r.target_label==-1],start,end)
            valid=union_intervals([(w.start,w.end) for w in windows if w.technical_valid])
            eligible=[];censored=[]
            for ev in events[events.record==record].itertuples():
                lo,hi=float(ev.start_seconds),float(ev.end_seconds)
                if not start<=lo<end:continue
                complete_followup=end>=min(hi,lo+120) and duration(clip_intervals(negative,lo-30,lo))>=30-1e-7
                if complete_followup:eligible.append((lo,min(hi,end)))
                censored.append({'onset':lo,'original_end':hi,'clip_end':end,'right_clipped':hi>end,'included_smoke_event':complete_followup})
            key=(person['subject'],record)
            references[key]={'subject':person['subject'],'record':record,'role':role,'start':start,'end':end,'negative':negative,'unknown':unknown,'valid':valid,'events':eligible,'censoring':censored,'interval_table':rows}
            clips.append(windows);selection.append({k:v for k,v in references[key].items() if k!='interval_table'})
            print('READ MONITOR',role,person['subject'],record,len(windows),flush=True)
    return clips,selection,references


def training_quality_30(windows,design):
    buffer=[];qualities=[];people=[]
    for w in windows:
        if buffer and (w.record!=buffer[-1].record or w.subject!=buffer[-1].subject or w.start!=buffer[-1].end):buffer=[]
        if not w.technical_valid:buffer=[];continue
        buffer.append(w)
        if len(buffer)==3:
            signal=Signal(np.concatenate([x.signals['ECG'].values for x in buffer]),buffer[0].signals['ECG'].fs,buffer[0].signals['ECG'].unit)
            qualities.append(quality(signal,'ECG'));people.append(w.subject);buffer=[]
    return quality_threshold(qualities,people,design)


def stream(runtime_blocks,light,heavy,family,thresholds,strategy):
    """Strict runtime-only input: not labels, activities, events or independent references."""
    scheduler=MonitorScheduler(strategy=='adaptive');calls=[];ticks=[];resets=[];previous=None
    for block in runtime_blocks:
        begin=perf_counter_ns()
        gap=previous is not None and (block.record!=previous.record or block.subject!=previous.subject or block.start!=previous.end)
        if gap or not block.technical_valid:resets.append(block.start)
        light_p=None;scheduled=None
        try:
            if block.technical_valid:light_p=float(light.predict(light_features(block.signals['ECG'])[None])[0])
            scheduled=scheduler.push(block,light_p if light_p is not None else .5)
        except UnavailableSignal:
            block=replace(block,technical_valid=False);resets.append(block.start);scheduler.push(block,.5)
        if scheduled is not None:
            context=RuntimeBlock(block.subject,block.record,scheduled.start,scheduled.end,{'ECG':scheduled.ecg},True)
            p=None;q=0.;error=None
            try:
                x=inputs(context,'ECG',family);p=float(heavy[scheduled.duration].predict(x[None])[0]);q=quality(scheduled.ecg,'ECG')
            except UnavailableSignal as exc:error=str(exc)
            latency=(perf_counter_ns()-begin)/1e6
            calls.append({'subject':block.subject,'record':block.record,'strategy':strategy,'start_seconds':scheduled.start,'end_seconds':scheduled.end,
                          'duration_seconds':scheduled.duration,'probability':p,'quality':q,'quality_pass':q>=thresholds[scheduled.duration] and p is not None,
                          'decision_seconds':scheduled.end+latency/1000,'CPU_milliseconds':latency,'heavy_cost':scheduled.cost/100,
                          'change_score':scheduled.change_score,'technical_failure_reason':error})
        ticks.append({'subject':block.subject,'record':block.record,'strategy':strategy,'start_seconds':block.start,'end_seconds':block.end,
                      'technical_valid':block.technical_valid,'light_probability':light_p,'heavy_called':scheduled is not None,
                      'light_cost':.15 if block.technical_valid else 0.,'heavy_cost':scheduled.cost/100 if scheduled else 0.,
                      'CPU_milliseconds':(perf_counter_ns()-begin)/1e6,'reference_used_by_runtime':False})
        previous=block
    return calls,ticks,sorted(set(resets))


def annotate_calls(calls,reference):
    # Offline labeling happens only after the causal stream has completed.
    return [dict(row,label=context_target(reference['interval_table'],row['start_seconds'],row['end_seconds'])) for row in calls]


def policy_records(calls_by,refs):
    records=[]
    for (subject,record,strategy),calls in calls_by.items():
        ref=refs[(subject,record)]
        if ref['role']!='policy':continue
        rows=tuple((r['decision_seconds'],r['probability'],r['quality_pass'],r['label']==0) for r in calls if r['probability'] is not None)
        resets=tuple(r['decision_seconds'] for r in calls if r['probability'] is None)
        records.append(PolicyMonitorRecord(subject,record,strategy,rows,tuple(ref['events']),tuple(ref['negative']),tuple(ref['valid']),ref['start'],ref['end'],resets))
    return records


def score_final(calls,resets,reference,calibrators,threshold):
    decisions=[];output=[]
    for row in calls:
        c=calibrators[(row['strategy'],row['duration_seconds'])]
        prediction_set=c.prediction_set(row['probability']) if row['probability'] is not None else ()
        accepted=len(prediction_set)==1 and row['quality_pass'] and row['decision_seconds']<reference['end']
        active=accepted and final_alarm(prediction_set,row['quality_pass'],row['probability'],threshold) if row['probability'] is not None else False
        decisions.append((row['decision_seconds'],active))
        output.append(dict(row,prediction_set=list(prediction_set),abstain=not accepted,active_alarm=active,
                           set_covered=row['label'] in prediction_set if row['label'] in (0,1) else None,
                           predicted=prediction_set[0] if accepted else None,role=reference['role'],
                           unavailable_reason='decision_after_clip_end' if row['decision_seconds']>=reference['end'] else None))
    decisions.extend((t,False) for t in resets)
    onsets,spans=alarm_episodes(decisions,reference['start'],reference['end'])
    scored=score_events(reference['events'],onsets,reference['negative'],reference['valid'],spans)
    scored.update({'subject':reference['subject'],'record':reference['record'],'unknown_exposure_seconds':duration(reference['unknown']),
                   'technical_missing_seconds':reference['end']-reference['start']-duration(reference['valid']),
                   'no_accepted_heavy_decision':not any(not r['abstain'] for r in output),
                   'alarm_onsets':onsets,'alarm_spans':spans,'reset_seconds':resets})
    return output,scored


def run_monitor(data_root,output,family):
    design=FrozenDesign();adapter=DatasetAdapters(data_root,design=design)
    path=create_run(output,'continuous_monitoring',family,{'training_people':'all 10 training people','other_roles':'first three frozen ranks each',
                    'clip_rule':'longest prequalified episode per person; onset-180s to onset+180s rounded to complete 10s blocks',
                    'reference_selection_only_for_smoke_coverage':True,'CNN_max_epochs':2,'formal_monitoring_followup':False})
    clips,selection,refs=select_clips(adapter,design);write_json(path/'selection_before_training.json',selection)
    train=[w for clip in clips for w in clip if w.role=='train'];triples=list(contiguous_triples(train))
    heavy10,q10,_=fit_tool(train,'ECG',family,design,path/'models'/'heavy10')
    heavy30,_,_=fit_tool(triples,'ECG',family,design,path/'models'/'heavy30');q30=training_quality_30(train,design)
    light_rows=[];light_x=[];light_y=[];light_people=[]
    for w in train:
        if w.technical_valid and w.label in (0,1):
            try:light_x.append(light_features(w.signals['ECG']));light_y.append(w.label);light_people.append(w.subject);light_rows.append({'subject':w.subject,'record':w.record,'start':w.start,'label':w.label})
            except UnavailableSignal:continue
    light=ClassicalModel().fit(np.stack(light_x),light_y,light_people,design);save_model(light,path/'models'/'light');reloaded=load_model(path/'models'/'light')
    if not np.array_equal(light.predict(np.stack(light_x[:8])),reloaded.predict(np.stack(light_x[:8]))):raise AssertionError('Light reload mismatch')
    light=reloaded;write_csv(path/'models'/'light'/'fit_windows.csv',light_rows)
    write_json(path/'training_quality.json',{'10':q10,'30':q30,'fit_people':sorted({w.subject for w in train}),'30s_quality_uses_label_independent_valid_contexts':True})
    policy10=[w for clip in clips if clip[0].role=='policy' for w in clip]
    policy30=[]
    for clip in clips:
        if clip[0].role!='policy':continue
        for lo in range(0,len(clip)-2,3):
            part=clip[lo:lo+3]
            if all(w.technical_valid for w in part) and part[0].end==part[1].start and part[1].end==part[2].start:
                native=part[0].signals['ECG'];signal=Signal(np.concatenate([w.signals['ECG'].values for w in part]),native.fs,native.unit)
                policy30.append(replace(part[0],end=part[-1].end,signals={'ECG':signal},label=None,metadata={}))
    profiles={'light10':profile_tool(policy10,lambda r:light.predict(light_features(r.signals['ECG'])[None]),design),
              'heavy10':profile_tool(policy10,lambda r:heavy10.predict(inputs(r,'ECG',family)[None]),design),
              'heavy30':profile_tool(policy30,lambda r:heavy30.predict(inputs(r,'ECG',family)[None]),design)}
    write_json(path/'policy_CPU_profiles.json',profiles)
    calls_by={};resets_by={};ticks=[]
    for clip in clips:
        if clip[0].role=='evaluation':continue
        for strategy in ('fixed','adaptive'):
            calls,tick,reset=stream([w.runtime() for w in clip],light,{10:heavy10,30:heavy30},family,{10:q10,30:q30},strategy)
            key=(clip[0].subject,clip[0].record,strategy);calls_by[key]=annotate_calls(calls,refs[key[:2]]);resets_by[key]=reset;ticks.extend(tick)
    records=policy_records(calls_by,refs)
    write_csv(path/'policy_execution.csv',[dict(r,role='policy') for key,rows in calls_by.items() if refs[key[:2]]['role']=='policy' for r in rows])
    # Include all observed runtime resets, not reference-unknown boundaries.
    records=[replace(r,resets=tuple(sorted(set(r.resets+tuple(resets_by[(r.subject,r.record,r.strategy)]))))) for r in records]
    policy=calibrate_alarm(records,design);write_json(path/'policy_alarm.json',policy);threshold=policy['selected']['threshold']
    calibration=[];calibrators={};calibration_execution=[]
    for strategy in ('fixed','adaptive'):
        for seconds in (10,30):
            rows=[r for key,values in calls_by.items() if key[2]==strategy and refs[key[:2]]['role']=='conformal' for r in values if r['duration_seconds']==seconds and r['label'] in (0,1) and r['probability'] is not None]
            c=Conformal(strategy=strategy,budget=0,duration=seconds).fit([r['probability'] for r in rows],[r['label'] for r in rows],[r['subject'] for r in rows],design)
            calibrators[(strategy,seconds)]=c;calibration_execution.extend(rows)
            calibration.append({'key':c.key,'fit_people':c.subjects,'rank':c.rank,'quantile':c.quantile,'scores':c.scores,'score_windows':len(rows),
                                'no_observations_reason':'no_calls_with_pure_target_in_this_smoke_duration' if not rows else None})
    write_json(path/'conformal.json',calibration);write_csv(path/'conformal_execution.csv',calibration_execution)
    evaluation_calls=[];scores=[];match_rows=[];policy_final=[]
    for clip in clips:
        key=(clip[0].subject,clip[0].record);ref=refs[key]
        for strategy in ('fixed','adaptive'):
            run_key=key+(strategy,)
            if clip[0].role=='evaluation':
                calls,tick,reset=stream([w.runtime() for w in clip],light,{10:heavy10,30:heavy30},family,{10:q10,30:q30},strategy)
                calls=annotate_calls(calls,ref);ticks.extend(tick)
            elif clip[0].role=='policy':calls=calls_by[run_key];reset=resets_by[run_key]
            else:continue
            annotated,scored=score_final(calls,reset,ref,calibrators,threshold);scored['strategy']=strategy
            known_negative=[r for r in annotated if r['label']==0 and r['decision_seconds']<ref['end']]
            scored['non_target_positive_decision_fraction']=sum(r['active_alarm'] for r in known_negative)/len(known_negative) if known_negative else None
            if clip[0].role=='policy':policy_final.append(scored);continue
            evaluation_calls.extend(annotated);scores.append(scored)
            pure=[r for r in annotated if r['label'] in (0,1)]
            accepted=[r for r in pure if not r['abstain']]
            finite=[r for r in pure if r['probability'] is not None]
            scored.update({'eligible_pure_heavy_decisions':len(pure),'reported_heavy_decisions':len(accepted),
                           'report_fraction':len(accepted)/len(pure) if pure else None,
                           'selective_risk':float(np.mean([r['predicted']!=r['label'] for r in accepted])) if accepted else None,
                           'set_coverage':float(np.mean([r['set_covered'] for r in pure])) if pure else None,
                           'AUROC':safe_auroc([r['label'] for r in finite],[r['probability'] for r in finite]) if finite else None})
            rt=[r for r in ticks if r['subject']==key[0] and r['record']==key[1] and r['strategy']==strategy]
            seconds=duration(ref['valid']);cost=sum(r['light_cost']+r['heavy_cost'] for r in rt)
            scored['normalized_cost']=cost;scored['normalized_cost_per_valid_hour']=cost/(seconds/3600) if seconds else None
            scored['heavy_calls']=len(annotated);scored['heavy_calls_per_valid_hour']=len(annotated)/(seconds/3600) if seconds else None
            scored['p95_CPU_milliseconds']=float(np.quantile([r['CPU_milliseconds'] for r in annotated],.95)) if annotated else None
            match_rows.extend([dict(m,subject=key[0],record=key[1],strategy=strategy) for m in scored['matches']])
    for strategy in ('fixed','adaptive'):
        rows=[r for r in policy_final if r['strategy']==strategy]
        # Verify final gating cannot add preliminary policy positives, call by call.
        for row in rows:
            for call in calls_by[(row['subject'],row['record'],strategy)]:
                c=calibrators[(strategy,call['duration_seconds'])]
                if call['probability'] is not None:
                    active=final_alarm(c.prediction_set(call['probability']),call['quality_pass'],call['probability'],threshold)
                    if active and not (call['quality_pass'] and call['probability']>=threshold):raise AssertionError('Final gate raised policy positive count')
    score_rows=[{k:v for k,v in r.items() if k not in ('matches','alarm_spans','alarm_onsets','reset_seconds')} for r in scores]
    write_csv(path/'predictions.csv',evaluation_calls);write_csv(path/'stream_ticks.csv',ticks);write_csv(path/'event_matches.csv',match_rows)
    write_csv(path/'person_metrics.csv',score_rows);write_json(path/'event_scores.json',scores);write_json(path/'policy_final_scoring.json',policy_final)
    write_json(path/'scoring_intervals.json',[{k:v for k,v in ref.items() if k!='interval_table'} for ref in refs.values()])
    write_json(path/'metrics.json',{'stage':'P8_quick','not_full_monitoring_evidence':True,'common_alarm_threshold':threshold,
                                   'policy_target_feasible':policy['selected']['target_feasible'],'selected_source_people_by_role':{role:len({c[0].subject for c in clips if c[0].role==role}) for role in ('train','policy','conformal','evaluation')},
                                   'per_person':score_rows,'physical_acquisition_latency':None,'runtime_light_cost_per_10s':.15})
    diagnostic_figures(path,score_rows,evaluation_calls,True);seal_results(path);return path
