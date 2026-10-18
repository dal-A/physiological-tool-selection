import numpy as np
import pytest
from pmtools.adapters import DatasetAdapters
from pmtools.models import ClassicalModel,fit_cnn
from pmtools.persistence import save_model,load_model
from pmtools.profiling import profile_tool,latency_ratio
from pmtools.safety import Conformal,Planner,PrimaryState,decide,person_diagnostic_conformal
from pmtools.schema import Window,Signal

def test_checkpoint_protocol_and_no_overwrite(toy_design,tmp_path):
    X=np.array([[0.],[1.],[2.],[3.]])
    model=ClassicalModel().fit(X,[0,0,1,1],['train:0']*2+['train:1']*2,toy_design)
    p=tmp_path/'classical';save_model(model,p);new=load_model(p)
    assert np.array_equal(new.predict(X),model.predict(X))
    with pytest.raises(FileExistsError):save_model(model,p)
def test_cnn_eda_fold_scaler_excludes_validation(toy_design,tmp_path):
    subjects=np.repeat([f'train:{i}' for i in range(6)],2);y=np.tile([0,1],6)
    X=np.stack([np.full((1,40),i+1,dtype='float32') for i in range(12)])
    m=fit_cnn(X,y,subjects,toy_design,max_epochs=1,modality='EDA')
    for fold,scaler in enumerate(m.metadata['cv_eda_scalers']):
        train=np.array([toy_design.fold(s)!=fold for s in subjects])
        assert scaler[0]==pytest.approx(float(X[train].mean()))
    p=tmp_path/'eda';save_model(m,p);new=load_model(p)
    assert np.array_equal(new.predict(X),m.predict(X))
def test_profile_recomputes_inside_callback_policy_only(toy_design):
    windows=[Window('policy:0','toy','r','policy',i*10,i*10+10,{'BVP':Signal(np.sin(np.arange(640)/10),64,'a')}) for i in range(25)]
    calls=[]
    def callback(runtime):calls.append(runtime.start);return np.sum(runtime.signals['BVP'].values**2)
    out=profile_tool(windows,callback,toy_design,warmup=2,repeats=10)
    assert len(calls)==12 and len(out['CPU_milliseconds'])==10 and out['physical_acquisition_latency'] is None
    assert latency_ratio(out,out)==1
    with pytest.raises(ValueError):profile_tool([Window('evaluation:0','toy','r','evaluation',0,10,{})],callback,toy_design)
def test_zip_never_discovered(tmp_path):
    # The fixed trusted-PKL path and frozen identity prevent duplicate ZIP discovery.
    (tmp_path/'S1.zip').write_bytes(b'not executed')
    adapter=DatasetAdapters.__new__(DatasetAdapters);adapter.root=tmp_path
    p={'source_subject':'S1','source':'PPG-DaLiA','subject':'PPG-DaLiA:S1','role':'train'}
    with pytest.raises(FileNotFoundError):next(adapter._pickle(p))
def test_reference_technical_failure_still_in_policy_global_loss(toy_design):
    p=Planner().fit([np.nan,.5],[np.nan,.1],[1,0],[1,0],['policy:0','policy:1'],toy_design,1)
    assert p.global_gain==0 and not p.should_call(PrimaryState(.5,.1),305)
def test_conformal_strategy_budget_duration_metadata(toy_design):
    a=Conformal(strategy='adaptive',budget=420,duration=10).fit([.9]*9,[1]*9,['conformal:0']*9,toy_design)
    b=Conformal(strategy='fixed',budget=420,duration=30).fit([.9]*9,[1]*9,['conformal:0']*9,toy_design)
    assert a.key!=b.key
def test_wrong_strategy_conformal_refused_before_tool(toy_design):
    c=Conformal(strategy='quality',budget=115).fit([.9]*9,[1]*9,['conformal:0']*9,toy_design)
    with pytest.raises(ValueError):decide(PrimaryState(.9,.9),.2,.2,{'primary':c},420)
def test_person_diagnostic_three_people_infinite(toy_design):
    c,indices=person_diagnostic_conformal([.9]*30,[1]*30,[f'conformal:{i%3}' for i in range(30)],list(range(30)),toy_design)
    assert len(indices)==3 and np.isinf(c.quantile)
