import numpy as np
import pytest
from pmtools.monitoring import BOCPD,MonitorScheduler,alarm_episodes,score_events,intersect_intervals,PolicyMonitorRecord,calibrate_alarm,final_alarm
from pmtools.statistics import decision_loss,report_metrics,person_means,paired_bootstrap,holm,paired_sign_test,zero_event_bounds,safe_auroc,balanced_person_mean
from conftest import ecg_window

def test_bocpd_normalized_tail_and_startup():
    b=BOCPD(max_run_length=16)
    for i in range(300):
        score,change=b.update(.1 if i<150 else .9)
        assert len(b.posterior)<=17 and b.posterior.sum()==pytest.approx(1) and 0<=score<=1+1e-12
        if i<5:assert not change
def test_fixed_cadence_30_seconds_and_cost():
    s=MonitorScheduler(False);calls=[]
    for k in range(9):
        call=s.push(ecg_window(k*10).runtime(),.1)
        if call:calls.append(call)
    assert [c.end for c in calls]==[30,60,90]
    assert all(c.duration==30 and len(c.ecg.values)==6000 for c in calls)
    assert s.total_cost==9*15+3*900
def test_unknown_labels_do_not_reset_controller():
    a=MonitorScheduler(False);b=MonitorScheduler(False)
    for k in range(12):
        ca=a.push(ecg_window(k*10,label=0).runtime(),.2)
        cb=b.push(ecg_window(k*10,label=-1 if k%2 else 1).runtime(),.2)
        assert (ca.end if ca else None)==(cb.end if cb else None)
        assert np.array_equal(a.bocpd.posterior,b.bocpd.posterior)
def test_gap_and_technical_reset_no_splicing():
    s=MonitorScheduler(False)
    assert s.push(ecg_window(0).runtime(),.1) is None
    assert s.push(ecg_window(20).runtime(),.1) is None
    assert s.push(ecg_window(30,valid=False).runtime(),.1) is None
    for t in (40,50):assert s.push(ecg_window(t).runtime(),.1) is None
    assert s.push(ecg_window(60).runtime(),.1).start==40
def test_adaptive_six_ticks_then_new_stable_context(monkeypatch):
    s=MonitorScheduler(True);counter=[0]
    def update(p):counter[0]+=1;return .8,counter[0]==6
    monkeypatch.setattr(s.bocpd,'update',update);calls=[]
    for k in range(14):
        c=s.push(ecg_window(k*10).runtime(),.2)
        if c:calls.append(c)
    assert [(c.start,c.end,c.duration) for c in calls]==[(0,30,30),(50,60,10),(60,70,10),(70,80,10),(80,90,10),(90,100,10),(100,110,10),(110,140,30)]
@pytest.mark.parametrize('alarm,detected',[(29.999,False),(30,True),(59.999,True),(60,False),(150,False)])
def test_half_open_alarm_match(alarm,detected):
    r=score_events([(30,60)],[alarm],[(0,30),(60,180)],[(0,180)])
    assert bool(r['detected'])==detected
def test_horizon_equality_and_one_to_one():
    r=score_events([(30,300)],[150],[(0,30)],[(0,300)])
    assert r['detected']==1 and r['detected_only_delay']==120
    r=score_events([(30,70),(80,120)],[90],[(0,30),(70,80)],[(0,120)])
    assert r['missed']==1 and r['detected']==1
def test_already_held_alarm_not_new_detection():
    onset,spans=alarm_episodes([(20,True),(90,False)],0,120)
    r=score_events([(30,80)],onset,[(0,30),(80,120)],[(0,120)],spans)
    assert r['missed']==1 and r['false_alarms']==1 and r['negative_alarm_occupancy']>0
def test_latency_at_end_not_start_or_backdated():
    onsets,_=alarm_episodes([(30+.25,True),(60,False)],0,60)
    assert onsets==[30.25]
    assert alarm_episodes([(60.25,True)],0,60)[0]==[]
def test_negative_exposure_abstention_does_not_shrink_denominator():
    r=score_events([],[],[(0,3600)],[(0,3600)])
    assert r['negative_hours']==1 and r['FAR_per_non_target_hour']==0 and r['detected_only_delay'] is None
    assert score_events([],[],[],[(0,3600)])['FAR_per_non_target_hour'] is None
def test_technical_gap_cannot_match_alarm():
    r=score_events([(30,90)],[50],[(0,30)],[(0,40),(60,120)])
    assert r['detected']==0
def test_interval_intersection_union_not_duplicate_exposure():
    assert intersect_intervals([(0,20),(10,30)],[(5,25)])==[(5.,25.)]
def policy_fixture(p=.9):
    return [PolicyMonitorRecord('policy:0','r',strategy,((10,p,True,True),(40,p,True,False),(70,p,False,True)),((30,60),),((0,30),(60,90)),((0,90),),0,90) for strategy in ('fixed','adaptive')]
def test_alarm_policy_cap_infeasible_no_unsafe_floor(toy_design):
    r=calibrate_alarm(policy_fixture(),toy_design)
    assert np.isinf(r['selected']['threshold']) and not r['selected']['target_feasible']
    assert r['selected']['max_negative_positive_fraction']==0
def test_alarm_policy_labels_only_not_conformal(toy_design):
    records=policy_fixture()
    records=[PolicyMonitorRecord('conformal:0',r.record,r.strategy,r.calls,r.events,r.negative_intervals,r.valid_intervals,r.start,r.end) for r in records]
    with pytest.raises(ValueError):calibrate_alarm(records,toy_design)
def test_conformal_final_cannot_raise_preliminary_positive_fraction():
    for singleton in ((),(0,),(1,),(0,1)):
        for quality in (False,True):
            for p in (.1,.8,.9):assert int(final_alarm(singleton,quality,p,.8))<=int(quality and p>=.8)
def test_zero_reports_risk_null_not_success():
    r=report_metrics([False]*5,[True]*5,[np.nan]*5)
    assert r['selective_risk'] is None and r['retained_MAE'] is None and r['mean_decision_loss']==.2 and r['no_reports']
def test_one_class_auroc_null():assert safe_auroc([0,0],[.1,.2]) is None
def test_loss_and_person_wrist_aggregation():
    assert np.array_equal(decision_loss([True,True,False],[False,True,True]),[0,1,.2])
    m=person_means([0,1,1],['s','s','s'],['t1','t1','t2'])
    assert m['s']==.75 # wrists average first; two times, not three people
def test_vital_stratified_person_mean():assert balanced_person_mean([0,0,0,1],[0,0,0,1])==.5
def test_paired_cluster_bootstrap_identical_pair_draws():
    r=paired_bootstrap([1,2,3],[1,2,3],replicates=10000)
    assert r['CI95']==[0.,0.] and r['valid_replicates']==10000 and r['independent_people']==3
def test_stratified_bootstrap_exact_constant_effect():
    r=paired_bootstrap([1,1,3,3],[0,0,1,1],labels=[0,0,1,1])
    assert r['estimate']==1.5 and r['CI95']==[1.5,1.5]
def test_bootstrap_undefined_replicates_not_zero():
    r=paired_bootstrap([np.nan,1],[0,0],replicates=1000)
    assert r['CI95'] is None and r['valid_replicates']<950
def test_holm_and_sign_ties():
    assert np.allclose(holm([.01,.04,.03,.8,.9]),[.05,.12,.12,1.,1.])
    assert paired_sign_test([0,0])['p']==1
    assert paired_sign_test([-1,-1,-1])['effective_people']==3
def test_zero_error_not_proof_of_zero_risk():
    r=zero_event_bounds(30)
    assert .09<r['Clopper_Pearson_one_sided_upper']<.10 and r['not_zero_risk_proof']
