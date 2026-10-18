import numpy as np
import pandas as pd
from sensitivity_classification import execute_cache,person_summary
from sensitivity_monitoring import changes_to_calls
from pmtools.safety import Conformal,Planner
from pmtools.monitoring import score_events

class Design:
    def require(self,people,role):
        assert role in ('policy','conformal','train')

def row():
    return dict(subject='person',source='PPGSPORT',record='record',start_seconds=0,end_seconds=10,
        wrist='left',role='evaluation',activity='walk',primary_probability=.1,primary_quality=1,
        secondary_probability=np.nan,secondary_quality=np.nan,PPG_HR_bpm=80,reference_HR_bpm=90)

def test_exact_10_is_reliable_but_5_is_not():
    d=Design();c=Conformal(strategy='attempt',budget=420).fit([],[],[],d)
    planner=Planner('REFERENCE').fit([.1]*3,[1]*3,[1]*3,[0]*3,['a','b','c'],d,1)
    assert execute_cache(row(),'hr_reliability','attempt',420,planner,c,0,0,10)['mean_decision_loss']==0
    assert execute_cache(row(),'hr_reliability','attempt',420,planner,c,0,0,5)['mean_decision_loss']==1

def test_offline_reference_post_correction_zero():
    d=Design();c=Conformal(strategy='fixed_fusion',budget=420).fit([],[],[],d)
    planner=Planner('REFERENCE').fit([.1]*3,[1]*3,[1]*3,[0]*3,['a','b','c'],d,1)
    r=execute_cache(row(),'hr_reliability','fixed_fusion',420,planner,c,0,0,5)
    assert r['mean_decision_loss']==0 and r['mean_cost']==4.2 and r['retained_error']==0

def test_secondary_budget_short_does_not_read():
    d=Design();c=Conformal(strategy='adaptive',budget=115).fit([.1],[0],['cal'],d)
    planner=Planner('REFERENCE').fit([.1]*3,[1]*3,[1]*3,[0]*3,['a','b','c'],d,1)
    r=execute_cache(row(),'hr_reliability','adaptive',115,planner,c,0,0)
    assert r['secondary_call_fraction']==0 and r['mean_cost']==1.15

def test_two_wrists_not_two_people():
    g=[dict(subject='person',source='PPGSPORT',record='record',start_seconds=0,end_seconds=10,
        strategy='quality',label_primary=k,report_fraction=float(k),dangerous_acceptance_all_windows=float(k),
        mean_decision_loss=.2 if k==0 else 1,mean_cost=1.15,secondary_call_fraction=0,set_coverage=1) for k in (0,1)]
    p=person_summary(g)
    assert len(p)==1 and p.report_fraction.iloc[0]==.5 and p.selective_risk.iloc[0]==1

def test_missing_monitor_tool_remains_none():
    rows=changes_to_calls(pd.DataFrame([dict(duration_seconds=10,quality=1,probability=np.nan,label=np.nan)]),{10:0})
    assert rows[0]['probability'] is None and rows[0]['label'] is None and not rows[0]['quality_pass']

def test_horizon_changes_miss_penalty_without_alarm_backdating():
    a=score_events([(100,500)],[200],[(0,100)],[(0,600)],horizon=60)
    b=score_events([(100,500)],[200],[(0,100)],[(0,600)],horizon=120)
    assert a['detected']==0 and a['miss_penalized_delay']==60
    assert b['detected']==1 and b['miss_penalized_delay']==100

def test_event_boundary_half_open():
    r=score_events([(100,160)],[160],[(0,100)],[(0,600)])
    assert r['detected']==0
