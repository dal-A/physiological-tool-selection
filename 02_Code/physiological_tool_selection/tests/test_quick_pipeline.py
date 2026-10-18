import json
from dataclasses import replace
import numpy as np
import pytest
from pmtools.schema import Signal,Window,RuntimeBlock
from pmtools.safety import Conformal,Planner
from pmtools.quick_classification import Tools,execute,inputs,selected_windows,copy_window,summarize
from pmtools.quick_monitor import stream,score_final
from pmtools.experiment_io import clean,create_run


class StubModel:
    def predict(self,X):return np.full(len(X),.8)


def sample(bvp,acc,role='evaluation:0'):
    return Window(role,'toy','r','evaluation',0,10,{'BVP':bvp,'ACC':acc,'EDA':Signal(np.ones(40),4,'microSiemens')},label=1,reference_hr=20)


@pytest.mark.parametrize('strategy',['quality','fixed_fusion','adaptive'])
def test_quick_lazy_low_budget(toy_design,bvp,acc,strategy):
    w=sample(bvp,acc);tools=Tools(StubModel(),StubModel(),'classical','stress')
    planner=Planner('EDA').fit([.8],[.9],[1],[0],['policy:0'],toy_design,1)
    c=Conformal(strategy=strategy,budget=115).fit([.9]*10,[1]*10,['conformal:0']*10,toy_design)
    row=execute(w,tools,strategy,115,planner,c,0,0)
    assert not row['secondary_called'] and row['runtime_secondary_accesses']==0 and row['cost_hundredths']==115


def test_quick_primary_reference_invariance(toy_design,bvp,acc):
    w=sample(bvp,acc);tools=Tools(StubModel(),StubModel(),'classical','stress')
    c=Conformal(strategy='quality',budget=270).fit([.9]*10,[1]*10,['conformal:0']*10,toy_design)
    a=execute(w,tools,'quality',270,None,c,0,0);b=execute(replace(w,label=0,reference_hr=999),tools,'quality',270,None,c,0,0)
    for key in ['primary_probability','probability','prediction_set','abstain','secondary_called','cost_hundredths']:
        assert a[key]==b[key]


def test_quick_oracle_post_error_target(toy_design,bvp,acc):
    tools=Tools(StubModel(),None,'classical','hr_reliability');w=sample(bvp,acc)
    planner=Planner('REFERENCE').fit([.8],[.8],[1],[0],['policy:0'],toy_design,1)
    c=Conformal(strategy='fixed_fusion',budget=420).fit([],[],[],toy_design)
    row=execute(w,tools,'fixed_fusion',420,planner,c,0,0)
    assert row['label_primary']==1 and row['label_final_tool']==0
    assert row['post_final_absolute_error_bpm']==0 and not row['wrong_if_reported']
    assert row['runtime_secondary_accesses']==1 and row['physical_reference_acquisition_latency'] is None


def test_quick_all_abstain_risk_null(toy_design,bvp,acc):
    tools=Tools(StubModel(),None,'classical','hr_reliability');w=sample(bvp,acc)
    c=Conformal(strategy='quality',budget=115).fit([],[],[],toy_design)
    row=execute(w,tools,'quality',115,None,c,0,0)
    summary,_=summarize([row],'hr_reliability')
    assert summary['selective_risk'] is None and summary['accuracy_reported'] is None and summary['report_fraction']==0


def test_quick_copy_does_not_keep_source_view(bvp,acc):
    w=sample(bvp,acc);new=copy_window(w)
    assert not np.shares_memory(new.signals['BVP'].values,w.signals['BVP'].values)
    assert np.array_equal(new.signals['BVP'].values,w.signals['BVP'].values)


def test_quick_pair_selection_keeps_both_wrists(bvp,acc):
    left=[replace(sample(bvp,acc),start=i*10,end=i*10+10,wrist='left') for i in range(6)]
    class Adapter:
        def iter_person(self,person):yield from left;yield from [replace(w,wrist='right') for w in left]
    selected=selected_windows(Adapter(),'evaluation:0',3)
    assert len(selected)==6
    for start in {w.start for w in selected}:assert {w.wrist for w in selected if w.start==start}=={'left','right'}


def test_quick_primary_inputs_do_not_use_eda(bvp,acc):
    w=sample(bvp,acc);r=w.runtime();x=inputs(r,'BVP','classical')
    altered=replace(r,signals={**r.signals,'EDA':Signal(np.full(40,np.nan),4,'microSiemens')})
    assert np.array_equal(x,inputs(altered,'BVP','classical'))


def test_quick_json_nonfinite_is_explicit():
    value=clean({'risk':np.nan,'quantile':np.inf,'empty':None,'zero':0.,'flag':np.bool_(True)})
    assert value=={'risk':None,'quantile':'+infinity','empty':None,'zero':0.,'flag':True}
    assert json.loads(json.dumps(value,allow_nan=False))==value


@pytest.mark.parametrize('strategy',['fixed','adaptive'])
def test_quick_monitor_runtime_has_no_reference(strategy):
    t=np.arange(2000)/200;sig=Signal(np.sin(2*np.pi*1.2*t),200,'mV')
    blocks=[RuntimeBlock('evaluation:0','r',10*i,10*i+10,{'ECG':sig},True) for i in range(12)]
    calls,ticks,resets=stream(blocks,StubModel(),{10:StubModel(),30:StubModel()},'classical',{10:0,30:0},strategy)
    assert len(ticks)==12 and calls and not resets
    assert all(r['decision_seconds']>=r['end_seconds'] for r in calls)
    assert all(not r['reference_used_by_runtime'] for r in ticks)


def test_quick_monitor_gap_not_spliced():
    t=np.arange(2000)/200;sig=Signal(np.sin(2*np.pi*1.2*t),200,'mV')
    blocks=[RuntimeBlock('evaluation:0','r',s,s+10,{'ECG':sig},True) for s in [0,10,40,50,60]]
    calls,ticks,resets=stream(blocks,StubModel(),{10:StubModel(),30:StubModel()},'classical',{10:0,30:0},'fixed')
    assert resets==[40] and len(calls)==1 and calls[0]['start_seconds']==40


def test_quick_monitor_last_decision_unavailable(toy_design):
    c=Conformal(strategy='fixed',budget=0,duration=10).fit([.99]*10,[1]*10,['conformal:0']*10,toy_design)
    row={'subject':'evaluation:0','record':'r','strategy':'fixed','duration_seconds':10,'start_seconds':10.,'end_seconds':20.,'decision_seconds':20.001,'probability':.99,'quality_pass':True,'label':1}
    ref={'subject':'evaluation:0','record':'r','role':'evaluation','start':0.,'end':20.,'events':[(10.,20.)],'negative':[(0.,10.)],'unknown':[],'valid':[(0.,20.)]}
    rows,score=score_final([row],[],ref,{('fixed',10):c},.8)
    assert rows[0]['abstain'] and not rows[0]['active_alarm'] and rows[0]['predicted'] is None
    assert rows[0]['unavailable_reason']=='decision_after_clip_end' and score['missed']==1 and score['alarm_onsets']==[]
