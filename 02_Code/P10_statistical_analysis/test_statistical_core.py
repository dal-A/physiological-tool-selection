import numpy as np
import pytest
from statistical_core import *

def test_all_abstain_missing_not_zero():
    r=bootstrap_mean([np.nan]*30)
    assert r['estimate'] is None and r['CI95_low'] is None and r['valid_replicates']==0

def test_paired_equal_zero_and_ties():
    r,d=paired_contrast([.2]*30,[.2]*30)
    assert r['estimate']==0 and r['sign_p_raw']==1 and r['ties']==30
    assert r['CI95_low']==r['CI95_high']==0

def test_stratified_unequal_class_weights():
    r=bootstrap_mean([1,1,1,0],[1,1,1,0])
    assert r['estimate']==.5 and r['CI95_low']==.5

def test_undefined_one_af_class():
    r=bootstrap_mean([np.nan,np.nan,0,0],[1,1,0,0])
    assert r['estimate'] is None and r['CI95_low'] is None

def test_paired_support_required():
    with pytest.raises(ValueError): paired_contrast([np.nan],[0])

def test_holm_known_values():
    assert np.allclose(holm_family([.04,.01,.03]),[.06,.03,.06])

def test_zero_people_event_bound():
    r=person_event_interval([0]*30)
    assert abs(r['exact_one_sided95_upper']-(1-.05**(1/30)))<1e-12
    assert r['exact_one_sided95_upper']>.09

def test_person_draw_deterministic_and_clusters():
    a,_=cluster_draws(3,replicates=50);b,_=cluster_draws(3,replicates=50)
    assert np.array_equal(a,b) and a.shape==(50,3)

def test_auc_stratified_perfect():
    r=bootstrap_patient_auc([0,0,1,1],[0,.1,.8,1],replicates=100)
    assert r['estimate']==r['CI95_low']==r['CI95_high']==1

def test_missing_replicates_contract():
    r=bootstrap_mean([.1,np.nan],replicates=10000)
    assert r['CI95_low'] is None and r['valid_replicates']<9500
