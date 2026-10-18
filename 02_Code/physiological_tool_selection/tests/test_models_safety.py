import numpy as np
import pytest
import torch
from pmtools.models import TinyCNN,ClassicalModel,fit_cnn,person_weights,balanced_weights,hierarchical_draws,eda_global_scaler
from pmtools.safety import Conformal,Planner,PrimaryState,SecondaryResult,decide,unreliable,corrected_error,quality_threshold,person_quantile

def calibrated(design,quantile=.2,strategy='adaptive',budget=420):
    c=Conformal(strategy=strategy,budget=budget);c.fit([.8]*9,[1]*9,['conformal:0']*9,design);c.quantile=quantile;return c

def planner(design,gain=.5,secondary='ECG',penalty=.02):
    return Planner(secondary,penalty).fit([.5]*3,[.1]*3,[gain]*3,[0]*3,['policy:0','policy:1','policy:2'],design,1.)

def test_equal_person_weights_and_class_balancing():
    s=['a']+['b']*9;y=[0]+[1]*9
    w=person_weights(s);b=balanced_weights(y,s)
    assert w[0]==.5 and sum(w[1:])==pytest.approx(.5)
    assert b[0]==pytest.approx(sum(b[1:]))
def test_person_quantile_not_window_quantile(toy_design):
    assert person_quantile([.1]+[.9]*99,['a']+['b']*99,.2)==.1
    with pytest.raises(ValueError):quality_threshold([.1],['evaluation:0'],toy_design)
def test_classical_scaler_train_only(toy_design):
    X=np.array([[0.],[0.],[2.],[2.]])
    m=ClassicalModel().fit(X,[0,0,1,1],['train:0']*2+['train:1']*2,toy_design)
    assert m.scaler.mean_[0]==1 and m.predict([[1]]).shape==(1,)
    with pytest.raises(ValueError):ClassicalModel().fit(X,[0,0,1,1],['evaluation:0']*4,toy_design)
def test_missing_training_class_fails(toy_design):
    with pytest.raises(ValueError):ClassicalModel().fit([[1],[2]],[0,0],['train:0']*2,toy_design)
def test_eda_person_global_tonic_retained(toy_design):
    mean,std=eda_global_scaler([np.full(40,1),np.full(40,3)],['train:0','train:1'],toy_design)
    assert mean==2 and std==1
@pytest.mark.parametrize('channels,length',[(1,640),(2,640),(1,40),(1,3750)])
def test_cnn_channel_duration_forward(channels,length):
    assert TinyCNN(channels)(torch.zeros(2,channels,length)).shape==(2,2)
def test_hierarchical_sampler_seed():
    y=np.array([0,1,1,1]);s=np.array(['train:0','train:1','train:1','train:2'])
    a=hierarchical_draws(y,s,2000,np.random.default_rng(1));b=hierarchical_draws(y,s,2000,np.random.default_rng(1))
    assert np.array_equal(a,b) and .45<np.mean(y[a])<.55
    assert .4<np.mean(s[a[y[a]==1]]=='train:2')<.6
def test_cnn_grouped_epoch_training_and_roundtrip(toy_design,tmp_path):
    rng=np.random.default_rng(1);X=rng.normal(size=(12,2,64)).astype('float32');y=np.tile([0,1],6);s=np.repeat([f'train:{i}' for i in range(6)],2)
    m=fit_cnn(X,y,s,toy_design,max_epochs=1)
    assert m.metadata['refit_epochs']==1
    for f in m.metadata['cv_people']:assert not set(f['train'])&set(f['validation'])
    path=tmp_path/'model.pt';torch.save(m.model.state_dict(),path);new=TinyCNN(2);new.load_state_dict(torch.load(path,weights_only=True));new.eval();m.model.eval()
    with torch.no_grad():assert torch.equal(new(torch.tensor(X)),m.model(torch.tensor(X)))
    with pytest.raises(ValueError):fit_cnn(X,y,['policy:0']*12,toy_design,max_epochs=1)
def test_conformal_infinite_rank(toy_design):
    c=Conformal().fit([.9,.8,.9],[1,1,1],['conformal:0','conformal:1','conformal:2'],toy_design)
    assert c.rank==4 and np.isinf(c.quantile) and c.prediction_set(.99)==(0,1)
def test_empty_duration_calibration_abstains_via_infinite_set(toy_design):
    c=Conformal().fit([],[],[],toy_design)
    assert c.rank==1 and np.isinf(c.quantile) and c.prediction_set(.99)==(0,1)
def test_conformal_equality_empty_doubleton(toy_design):
    c=calibrated(toy_design,.25)
    assert c.prediction_set(.75)==(1,) and c.prediction_set(.5)==()
    c.quantile=.6;assert c.prediction_set(.5)==(0,1)
def test_policy_conformal_disjoint(toy_design):
    with pytest.raises(ValueError):Conformal().fit([.9],[1],['policy:0'],toy_design)
    with pytest.raises(ValueError):Conformal().fit([.9],[1],['conformal:0'],toy_design,policy_frozen=False)
@pytest.mark.parametrize('error,expected',[(9.999,0),(10.,0),(10.001,1)])
def test_hr_target_boundary(error,expected):assert unreliable(60+error,60)==expected
def test_hr_unavailable_target():assert unreliable(None,60)==1
def test_post_oracle_recalculation():assert corrected_error(60,60)==0 and corrected_error(80,60)==20
def test_budget_equality_and_no_free_secondary(toy_design):
    p=planner(toy_design);calls=[];c=calibrated(toy_design)
    callback=lambda:(calls.append(1) or SecondaryResult(.95,.9))
    c419=calibrated(toy_design,budget=419)
    a=decide(PrimaryState(.5,.1),.2,.2,{'primary':c419,'fusion':c419},419,p,callback)
    assert not calls and not a.secondary_called and a.cost==115
    b=decide(PrimaryState(.5,.1),.2,.2,{'primary':c,'fusion':c},420,p,callback)
    assert calls==[1] and b.secondary_called and b.cost==420
def test_low_primary_secondary_rescue(toy_design):
    c=calibrated(toy_design);p=planner(toy_design)
    d=decide(PrimaryState(.5,.1),.2,.2,{'primary':c,'fusion':c},420,p,lambda:SecondaryResult(.99,.9))
    assert not d.abstain and d.predicted==1
def test_called_secondary_failure_abstains_not_switch(toy_design):
    c=calibrated(toy_design,strategy='fixed_fusion');p=planner(toy_design)
    d=decide(PrimaryState(.95,.9),.2,.2,{'primary':c,'fusion':c},420,p,lambda:SecondaryResult(None,.9,False),strategy='fixed_fusion')
    assert d.abstain and d.secondary_called and d.cost==420
def test_primary_quality_gate_when_not_called(toy_design):
    c=calibrated(toy_design,strategy='quality',budget=115);d=decide(PrimaryState(.9,.1),.2,.2,{'primary':c},115,strategy='quality')
    assert d.abstain and not d.secondary_called
def test_under_primary_budget_no_execution(toy_design):
    d=decide(PrimaryState(.9,.9),.2,.2,{},114,planner(toy_design),lambda:pytest.fail('secondary invoked'))
    assert d.abstain and d.cost==0
def test_negative_gain_not_clipped(toy_design):assert not planner(toy_design,-.2).should_call(PrimaryState(.5,.1),305)
def test_cost_benefit_strict_equality(toy_design):
    p=planner(toy_design,gain=.061)
    assert not p.should_call(PrimaryState(.5,.1),305)
def test_planner_fit_policy_only(toy_design):
    with pytest.raises(ValueError):Planner().fit([.5],[.5],[1],[0],['evaluation:0'],toy_design,1)
def test_oracle_only_access_after_budget(toy_design):
    p=planner(toy_design,secondary='REFERENCE');c=calibrated(toy_design);calls=[]
    callback=lambda:(calls.append(1) or SecondaryResult(None,0,True,reference_hr=75))
    a=decide(PrimaryState(.9,.1),.2,.2,{'primary':calibrated(toy_design,budget=419)},419,p,callback,is_hr=True,hr=110)
    assert not calls and a.abstain
    b=decide(PrimaryState(.9,.1),.2,.2,{'primary':c},420,p,callback,is_hr=True,hr=110)
    assert b.reported_hr==75 and corrected_error(b.reported_hr,75)==0 and not b.abstain
@pytest.mark.parametrize('budget',[0,99,100,114,115,170,200,220,270,300,419,420])
def test_cost_never_overspend(budget,toy_design):
    c=calibrated(toy_design,budget=budget);p=planner(toy_design)
    d=decide(PrimaryState(.5,.1),.2,.2,{'primary':c,'fusion':c},budget,p,lambda:SecondaryResult(.9,.9))
    assert d.cost<=budget
