"""Within-completed-window transforms and fixed engineering signal proxies."""
from fractions import Fraction
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly,welch,find_peaks
from scipy.stats import skew,kurtosis
from wfdb.processing import xqrs_detect
from .schema import Signal

class UnavailableSignal(ValueError): pass

BASE_FEATURE_NAMES=('mean','std','median','peak_to_peak','skewness','excess_kurtosis','dominant_frequency','band_power_ratio','normalized_spectral_entropy','peak_rate','heart_rate','RR_CV','RMSSD','valid_RR_fraction','HR_failure')
ACC_FEATURE_NAMES=('magnitude_mean','magnitude_std','magnitude_ptp','difference_std','dominant_frequency','normalized_spectral_entropy')
EDA_FEATURE_NAMES=('mean','std','median','ptp','difference_std','linear_slope')
LIGHT_FEATURE_NAMES=('heart_rate','RR_CV','RMSSD','pNN50','valid_RR_fraction','peak_count','HR_failure')

def valid_array(x,allow_constant=False):
    a=np.asarray(x,dtype=float)
    if not a.size or not np.isfinite(a).all(): raise UnavailableSignal('Missing/nonfinite signal; no imputation')
    if not allow_constant and np.ptp(a)==0: raise UnavailableSignal('Constant PPG/ECG')
    return a

def resample(signal,target_hz):
    a=valid_array(signal.values,allow_constant=True)
    ratio=Fraction(float(target_hz)/float(signal.fs)).limit_denominator(100000)
    if abs(signal.fs*ratio.numerator/ratio.denominator-target_hz)>1e-7: raise ValueError('Inexact rational resampling')
    expected=len(a)/signal.fs*target_hz
    if abs(expected-round(expected))>1e-7: raise ValueError('Incomplete time support')
    b=resample_poly(a,ratio.numerator,ratio.denominator,axis=0)
    if len(b)!=round(expected): raise ValueError('Resampled length mismatch')
    return b

def scale_window(a):
    a=np.asarray(a,dtype=float); m=np.median(a)
    return (a-m)/(np.median(np.abs(a-m))+1e-6)

def filtered(signal,modality):
    valid_array(signal.values)
    fs=125 if modality=='ECG' else 64
    x=resample(signal,fs).reshape(-1)
    band=(.5,40) if modality=='ECG' else (.5,8)
    x=sosfiltfilt(butter(3,band,btype='bandpass',fs=fs,output='sos'),x)
    return x,fs

def spectral(x,fs,band):
    f,p=welch(x,fs=fs,nperseg=min(len(x),int(4*fs)))
    total=p.sum();prob=p/total if total>0 else np.zeros_like(p)
    entropy=float(-np.sum(prob[prob>0]*np.log(prob[prob>0]))/np.log(len(p))) if len(p)>1 else 0.
    ratio=float(p[(f>=band[0])&(f<=band[1])].sum()/total) if total>0 else 0.
    dominant=float(f[np.argmax(p)]) if total>0 else 0.
    return dominant,ratio,entropy

def detect_peaks(x,fs,modality):
    if modality=='ECG': return np.asarray(xqrs_detect(x,fs,learn=True,verbose=False),dtype=int)
    n=scale_window(x)
    return find_peaks(n,distance=max(1,round(.25*fs)),prominence=max(.15,.25*np.std(n)))[0]

def hr_from_peaks(peaks,fs):
    peaks=np.asarray(peaks)
    if len(peaks)<4 or (peaks[-1]-peaks[0])/fs<6: return None
    rr=np.diff(peaks)/fs;good=rr[(rr>=.25)&(rr<=2)]
    if not len(good): return None
    hr=float(60/np.median(good))
    return hr if 30<=hr<=240 else None

def estimate_hr(signal):
    try:
        x,fs=filtered(signal,'BVP');return hr_from_peaks(detect_peaks(x,fs,'BVP'),fs)
    except UnavailableSignal: return None

def rr_features(peaks,fs):
    rr=np.diff(peaks)/fs
    cv=float(np.std(rr)/np.mean(rr)) if len(rr) and np.mean(rr)>0 else 0.
    rmssd=float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else 0.
    fraction=float(np.mean((rr>=.3)&(rr<=2))) if len(rr) else 0.
    return rr,cv,rmssd,fraction

def motion_penalty(acc):
    mag=acc_magnitude(acc);std=float(np.std(mag))
    if std==0: return 1.
    eps=1e-6*max(abs(float(np.mean(mag))),std,1e-12)
    j=float(np.std(np.diff(mag)))/max(std,eps)
    return 1/(1+j)

def acc_magnitude(signal):
    """Native tri-axis magnitude, then declared-rate polyphase reduction to32Hz."""
    a=valid_array(signal.values,True)
    if a.ndim!=2 or a.shape[1]!=3:raise ValueError('ACC requires three axes')
    return resample(Signal(np.linalg.norm(a,axis=1),signal.fs,signal.unit),32).reshape(-1)

def quality(signal,modality='BVP',acc=None):
    if modality=='EDA':
        if signal.unit!='microSiemens': raise ValueError('EDA units not microSiemens')
        try: std=float(np.std(valid_array(signal.values,True)))
        except UnavailableSignal:return 0.
        return std/(std+.01)
    try:
        x,fs=filtered(signal,modality);peaks=detect_peaks(x,fs,modality)
        # Band-energy proxy must see the resampled raw signal, not filter-created energy.
        _,band,ent=spectral(resample(signal,fs).reshape(-1),fs,(.5,40) if modality=='ECG' else (.5,8))
        rr,cv,_,v=rr_features(peaks,fs); stability=np.clip(1-cv/.5,0,1) if len(rr) else 0.
        q=float(np.clip((.30*band+.20*(1-ent)+.25*stability+.15*v)/.90,0,1))
        return q*(motion_penalty(acc) if acc is not None else 1)
    except UnavailableSignal:return 0.

def base_features(signal,modality='BVP'):
    x,fs=filtered(signal,modality);peaks=detect_peaks(x,fs,modality)
    hr=hr_from_peaks(peaks,fs);rr,cv,rmssd,v=rr_features(peaks,fs)
    dominant,band,ent=spectral(resample(signal,fs).reshape(-1),fs,(.5,40) if modality=='ECG' else (.5,8))
    result=np.array([np.mean(x),np.std(x),np.median(x),np.ptp(x),skew(x),kurtosis(x),dominant,band,ent,len(peaks)/(len(x)/fs),hr if hr is not None else 0,cv,rmssd,v,float(hr is None)])
    if not np.isfinite(result).all():raise UnavailableSignal('Nonfinite derived features')
    return result

def acc_features(signal):
    mag=acc_magnitude(signal);dom,_,ent=spectral(mag,32,(0,16))
    return np.array([mag.mean(),mag.std(),np.ptp(mag),np.std(np.diff(mag)),dom,ent])

def eda_features(signal):
    if signal.unit!='microSiemens': raise ValueError('EDA units unknown')
    x=valid_array(signal.values,True).reshape(-1);t=np.arange(len(x))/signal.fs
    return np.array([x.mean(),x.std(),np.median(x),np.ptp(x),np.std(np.diff(x)),np.polyfit(t,x,1)[0]])

def primary_features(signals,modality='BVP'):
    base=base_features(signals[modality],modality)
    return np.r_[base,acc_features(signals['ACC'])] if 'ACC' in signals else base

def light_features(signal):
    x,fs=filtered(signal,'ECG');peaks=detect_peaks(x,fs,'ECG');hr=hr_from_peaks(peaks,fs);rr,cv,rmssd,v=rr_features(peaks,fs)
    pnn50=float(np.mean(np.abs(np.diff(rr))>.05)) if len(rr)>1 else 0
    return np.array([hr if hr is not None else 0,cv,rmssd,pnn50,v,len(peaks),float(hr is None)])

def cnn_input(signals,modality='BVP',eda_scaler=None):
    if modality=='EDA':
        if signals['EDA'].unit!='microSiemens' or eda_scaler is None: raise ValueError('Training-only EDA scaler required')
        x=valid_array(signals['EDA'].values,True).reshape(-1)
        return ((x-eda_scaler[0])/(eda_scaler[1]+1e-6))[None,:].astype('float32')
    x,fs=filtered(signals[modality],modality);channels=[scale_window(x)]
    if 'ACC' in signals:
        mag=scale_window(acc_magnitude(signals['ACC']))
        # Interpolation uses only the completed window and endpoint holds.
        channels.append(np.interp(np.arange(len(x))/fs,np.arange(len(mag))/32,mag))
    return np.stack(channels).astype('float32')
