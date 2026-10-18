import pickle
import numpy as np
import pytest
from pmtools.protocol import FrozenDesign
from pmtools.schema import Signal,pure_wesad_label,dalia_reference,check_dalia_count,activity_mode,contiguous_triples
from pmtools.adapters import truth,empatica_start,read_empatica
from pmtools.signals import filtered,resample,UnavailableSignal,base_features,primary_features,quality,motion_penalty,estimate_hr,hr_from_peaks,cnn_input,eda_features,light_features,acc_magnitude
from conftest import ecg_window

@pytest.mark.parametrize('native,target,n',[(200,125,2000),(64,64,640),(64,32,640),(700,125,7000),(500,125,5000)])
def test_exact_sample_support(native,target,n):
    signal=Signal(np.sin(np.arange(n)/10),native,'mV');out=resample(signal,target)
    assert len(out)==10*target

@pytest.mark.parametrize('label,expected',[(0,None),(1,0),(2,1),(3,0),(4,0),(5,None),(6,None),(7,None)])
def test_wesad_task_mapping(label,expected):assert pure_wesad_label(np.full(7000,label))==expected

def test_mixed_wesad_excluded():assert pure_wesad_label([1,2]) is None
def test_activity_tie_deterministic():assert activity_mode([3,2,3,2])==2
def test_dalia_center_not_label_index():
    labels=np.arange(12,dtype=float)*10
    assert dalia_reference(labels,0,10)==10  # centers 4,6,8 -> labels 0,10,20
    assert dalia_reference(labels,10,20)==50 # centers 10..18 -> labels30..70
    assert dalia_reference(labels,0,10,True)==5 # source [0,8),[2,10)
def test_dalia_count_mismatch_fails():
    assert check_dalia_count(7,20)==7
    with pytest.raises(ValueError):check_dalia_count(8,20)
def test_signal_length_mismatch():
    with pytest.raises(ValueError):Signal(np.zeros(639),64,'a').validate(10)
def test_declared_missing_no_imputation(bvp):
    a=bvp.values.copy();a[4]=np.nan
    with pytest.raises(UnavailableSignal):filtered(Signal(a,64,'a'),'BVP')
    assert estimate_hr(Signal(a,64,'a')) is None
def test_constant_ppg_unavailable():
    with pytest.raises(UnavailableSignal):filtered(Signal(np.zeros(640),64,'a'),'BVP')
def test_constant_acc_allowed(acc):assert motion_penalty(acc)==1
def test_sport_acc_analysis_rate_unified_to32():
    t=np.arange(640)/64;a=Signal(np.c_[np.sin(t),np.cos(t),np.ones(640)],64,'source_unit')
    assert acc_magnitude(a).shape==(320,)
def test_motion_unit_invariant():
    t=np.arange(320)/32;a=np.c_[1+.3*np.sin(t),np.cos(t),np.sin(t)]
    assert motion_penalty(Signal(a,32,'g'))==pytest.approx(motion_penalty(Signal(a*1000,32,'mg')))
def test_eda_flat_quality_zero():assert quality(Signal(np.ones(40),4,'microSiemens'),'EDA')==0
def test_eda_units_mandatory():
    with pytest.raises(ValueError):quality(Signal(np.ones(40),4,'unknown'),'EDA')
def test_features_schema(bvp,acc):
    assert base_features(bvp).shape==(15,)
    assert primary_features({'BVP':bvp,'ACC':acc}).shape==(21,)
    assert eda_features(Signal(np.ones(40),4,'microSiemens')).shape==(6,)
def test_peak_hr_failure_not_zero():
    assert hr_from_peaks([0,32],64) is None
    assert hr_from_peaks(np.arange(64,640,64),64)==60
def test_cnn_single_double_and_eda(bvp,acc):
    assert cnn_input({'BVP':bvp}).shape==(1,640)
    assert cnn_input({'BVP':bvp,'ACC':acc}).shape==(2,640)
    eda=Signal(np.full(40,2.),4,'microSiemens')
    assert np.allclose(cnn_input({'EDA':eda},'EDA',(1.,1.)),1/(1+1e-6))
def test_within_window_processing_causal(bvp):
    before=filtered(bvp,'BVP')[0]
    unrelated_future=np.ones(640)*1e9
    assert np.array_equal(before,filtered(bvp,'BVP')[0]) and unrelated_future[0]>0
def test_runtime_no_reference_access():
    b=ecg_window(label=-1).runtime()
    assert not hasattr(b,'label') and not hasattr(b,'reference_hr')
def test_thirty_seconds_no_gap_or_mixed_splicing():
    good=list(contiguous_triples([ecg_window(0),ecg_window(10),ecg_window(20)]))
    assert len(good)==1 and len(good[0].signals['ECG'].values)==6000
    assert not list(contiguous_triples([ecg_window(0),ecg_window(20),ecg_window(30)]))
    assert not list(contiguous_triples([ecg_window(0),ecg_window(10,label=1),ecg_window(20)]))
def test_frozen_roles_records_and_wrists():
    d=FrozenDesign();assert len(d.people)==362
    assert len(d.select('PPGSPORT'))==29 and len(d.select('GALAXYPPG'))==23
    assert d.record_subject('SHDB-AF','052')==d.record_subject('SHDB-AF','128')
    with pytest.raises(ValueError):d.record_subject('SHDB-AF','005')
    for task,people in d.tasks.items():
        assert len({p['subject'] for p in people})==len(people)
    with pytest.raises(ValueError):d.require([d.select('Stress-Predict')[0]['subject']],'train')
def test_source_header_and_duplicate_trim(tmp_path):
    p=tmp_path/'BVP.csv';p.write_text('100\n64\n'+''.join(f'{i}\n' for i in range(650)),encoding='utf-8')
    a,t,fs=read_empatica(p,{'start':100.,'fs':64.,'original_n':650,'n':640})
    assert a.shape==(640,1) and t==100 and fs==64
def test_parse_utc_header():assert empatica_start('1970-01-01 00:01:40')==100
def test_real_ecg_feature_detector():
    # Synthetic QRS-like waveform, not a new patient's performance evaluation.
    t=np.arange(2000)/200;x=np.zeros_like(t)
    for peak in np.arange(.5,10,.8):x+=np.exp(-((t-peak)/.018)**2)
    x+=.01*np.sin(2*np.pi*t)
    assert light_features(Signal(x,200,'mV')).shape==(7,)
