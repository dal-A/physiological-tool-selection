"""Full-cohort scope and aggregation regression tests, independent of performance."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from pmtools.protocol import FrozenDesign
from pmtools.schema import Window,Signal
from pmtools.full_classification import all_people,load_full,full_summary,paired_time_means
from pmtools.full_io import CsvSink,create_full_run,seal_full_run
from pmtools.full_monitor import ReferenceIntervals,thirty_contexts,profile_selection,aggregate_records,full_policy_records
from pmtools.quick_monitor import context_target
from pmtools.full_training import full_fit
from pmtools.models import ClassicalModel


@pytest.mark.parametrize('task,counts',[
    ('af_afl',{'train':71,'policy':15,'conformal':15,'evaluation':60}),
    ('hr_reliability',{'train':9,'policy':3,'conformal':3,'evaluation':52}),
    ('stress',{'train':9,'policy':3,'conformal':3,'evaluation':30,'supplementary_evaluation':35})])
def test_full_all_frozen_people(task,counts):
    rows=all_people(FrozenDesign(),task)
    assert {role:sum(r['role']==role for r in rows) for role in counts}==counts
    assert len({r['subject'] for r in rows})==sum(counts.values())


def test_full_load_no_48_window_cap(tmp_path,bvp,acc):
    class Design:
        people={'p':{'role':'train'}}
        def select(self,source,role):return [{'subject':'p','role':role}] if role=='train' else []
    class Adapter:
        def iter_person(self,subject):
            for i in range(73):
                yield Window('p','VITALDB','r','train',i*10,i*10+10,{'BVP':bvp,'ACC':acc},label=i%2)
    rows=load_full(Adapter(),Design(),'af_afl',tmp_path)
    assert len(rows)==73 and rows[-1].start==720


def test_full_load_duplicate_rejected(tmp_path,bvp):
    class Design:
        people={'p':{'role':'train'}}
        def select(self,source,role):return [{'subject':'p','role':role}] if role=='train' else []
    class Adapter:
        def iter_person(self,subject):
            w=Window('p','VITALDB','r','train',0,10,{'BVP':bvp},label=1)
            yield w;yield w
    with pytest.raises(AssertionError,match='Duplicate'):load_full(Adapter(),Design(),'af_afl',tmp_path)


def test_full_cnn_epoch_cap_not_smoke(monkeypatch,tmp_path,toy_design,bvp,acc):
    captured={}
    def fake_fit(X,y,subjects,design,**kwargs):
        captured.update(kwargs)
        return object()
    # Test the invocation before optimization, no fake scientific checkpoint is delivered.
    monkeypatch.setattr('pmtools.full_training.fit_cnn',fake_fit)
    monkeypatch.setattr('pmtools.full_training.persist_fit',lambda *args:args)
    rows=[Window('train:'+str(i%3),'toy','r','train',i*10,i*10+10,{'BVP':bvp,'ACC':acc},label=i%2) for i in range(6)]
    full_fit(rows,'BVP','cnn',toy_design,tmp_path)
    assert captured=={'max_epochs':25,'patience':5,'modality':'BVP'}


def test_full_create_no_overwrite_and_provenance(tmp_path):
    out=tmp_path/'run';create_full_run(out,'stress','cnn')
    config=json.loads((out/'run_config.json').read_text(encoding='utf-8'))
    assert config['preset']=='full' and config['CNN_max_epochs']==25 and not config['warm_start']
    with pytest.raises(FileExistsError):create_full_run(out,'stress','cnn')
    seal_full_run(out)
    assert json.loads((out/'result_manifest.json').read_text())['status'].endswith('AWAITING_INDEPENDENT_ACCEPTANCE')


def test_csv_sink_preserves_null_zero_and_all_rows(tmp_path):
    out=tmp_path/'p.csv'
    with CsvSink(out) as sink:
        sink.add([{'id':i,'risk':None,'zero':0.,'set':[0,1]} for i in range(70)])
        sink.add([{'id':70,'risk':None,'zero':0.,'set':[]}])
    table=pd.read_csv(out)
    assert len(table)==71 and table.risk.isna().all() and (table.zero==0).all()
    assert json.loads(table.iloc[-1]['set'])==[]


def test_full_interval_integrals_against_independent_union():
    table=pd.DataFrame({'start_seconds':[0.,4.,6.,10.,15.],'end_seconds':[4.,6.,10.,15.,30.],
                        'target_label':[0,-1,0,1,1]})
    lookup=ReferenceIntervals(table)
    rng=np.random.default_rng(20260921)
    for _ in range(120):
        start,end=sorted(rng.uniform(0,30,2))
        assert lookup.target(start,end)==context_target(table,start,end)
    assert lookup.target(10,30)==1 and lookup.target(0,10) is None


def test_full_30_quality_contexts_ignore_rhythm(bvp):
    signal=Signal(np.sin(np.arange(2000)/200*2*np.pi),200,'mV')
    windows=[Window('train:0','SHDB-AF','r','train',i*10,i*10+10,{'ECG':signal},label=[0,-1,1][i%3]) for i in range(7)]
    rows=list(thirty_contexts(iter(windows)))
    assert len(rows)==2 and [(r.start,r.end) for r in rows]==[(0,30),(30,60)]
    assert all(r.label is None and len(r.signals['ECG'].values)==6000 for r in rows)
    windows[1]=replace(windows[1],technical_valid=False)
    assert [(r.start,r.end) for r in thirty_contexts(windows)]==[(20,50)]


def test_bounded_profile_selection_matches_sealed_hash(bvp):
    import hashlib
    windows=[Window('policy:0','toy','r','policy',i*10,i*10+10,{'BVP':bvp}) for i in range(71)]
    expected=sorted(windows,key=lambda w:hashlib.sha256(('20260921|profile|'+repr((w.subject,w.record,w.start,w.end,w.wrist))).encode()).hexdigest())[:20]
    assert [w.start for w in profile_selection(iter(windows))]==[w.start for w in expected]


def test_same_time_wrists_not_independent():
    rows=[{'record':'r','start_seconds':s,'end_seconds':s+10} for s in [0,0,10]]
    assert paired_time_means(rows,[0,1,0]).tolist()==[.5,0.]


def test_full_failed_scheduled_call_does_not_shrink_negative_denominator():
    calls={('policy:0','r','fixed'):[{'decision_seconds':30.001,'probability':None,
                                    'quality_pass':False,'label':0}]}
    refs={('policy:0','r'):{'role':'policy','events':[(60,90)],'negative':[(0,60)],
                           'valid':[(0,120)],'start':0,'end':120}}
    rows=full_policy_records(calls,refs)
    assert len(rows[0].calls)==1 and np.isnan(rows[0].calls[0][1])
    assert rows[0].calls[0][2:] == (False,True) and rows[0].resets==(30.001,)


def test_monitor_records_combined_not_independent_people():
    # Two native records from one participant. FAR is sum(onsets)/sum(exposure),
    # and delay is over that participant's events, not an unweighted record mean.
    common={'subject':'evaluation:0','strategy':'fixed','unknown_exposure_seconds':0.,
            'technical_missing_seconds':0.,'no_decision_seconds':10.,'negative_alarm_seconds':0.}
    records=[dict(common,record='a',matches=[{'delay':10.,'detected':True}],negative_hours=1.,
                  valid_signal_seconds=3600.,false_alarms=1),
             dict(common,record='b',matches=[{'delay':120.,'detected':False}]*3,negative_hours=3.,
                  valid_signal_seconds=10800.,false_alarms=0)]
    rows=aggregate_records(records,[],[])
    assert len(rows)==1 and rows[0]['records']==2 and rows[0]['events']==4
    assert rows[0]['miss_penalized_delay']==92.5 and rows[0]['FAR_per_non_target_hour']==.25
    assert rows[0]['selective_risk'] is None and rows[0]['no_accepted_heavy_decision']


def test_full_monitor_eval_has_30_person_clusters_and_167_events():
    from pmtools.adapters import DatasetAdapters
    design=FrozenDesign();people=design.select('SHDB-AF','evaluation')
    adapter=DatasetAdapters(Path('D:/大学学习/论文/论文项目/URL'),design=design)
    assert len(people)==30
    events=adapter.table('P5','target_episodes.csv')
    eligible=events.eligible.astype(str).str.lower().eq('true')
    selected=events[eligible & events.subject.isin([p['subject'] for p in people])]
    assert len(selected)==167 and selected.subject.nunique()==30
