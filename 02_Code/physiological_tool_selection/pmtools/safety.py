"""Disjoint fit roles, finite-sample sets, lazy secondary tools and exact costs."""
from dataclasses import dataclass
import math
import hashlib
import numpy as np
from .models import person_weights

def probability(p):
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
    return float(p)

def person_quantile(values,subjects,fraction=.2):
    values=np.asarray(values,dtype=float);subjects=np.asarray(subjects)
    if not len(values) or len(values)!=len(subjects) or not np.isfinite(values).all() or not 0<fraction<=1:raise ValueError('Invalid quantile data')
    order=np.argsort(values,kind='stable');w=person_weights(subjects)[order]
    return float(values[order][min(np.searchsorted(np.cumsum(w),fraction-1e-12),len(order)-1)])

def quality_threshold(values,subjects,design,fraction=.2):
    design.require(subjects,'train');return person_quantile(values,subjects,fraction)

class Conformal:
    def __init__(self,alpha=.1,strategy='primary',budget=115,duration=10):
        if not 0<alpha<1:raise ValueError('Invalid alpha')
        self.alpha=alpha;self.key=(strategy,int(budget),int(duration));self.fitted=False
    def fit(self,p,y,subjects,design,policy_frozen=True):
        if len(subjects):design.require(subjects,'conformal')
        if not policy_frozen:raise ValueError('Policy must be frozen first')
        p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int)
        if len(p)!=len(y) or len(y)!=len(subjects) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not set(y)<= {0,1}:raise ValueError('Invalid scores')
        self.subjects=sorted(set(subjects));self.scores=np.sort(1-np.where(y==1,p,1-p))
        self.rank=math.ceil((len(p)+1)*(1-self.alpha))
        self.quantile=float('inf') if self.rank>len(p) else float(self.scores[self.rank-1]);self.fitted=True
        return self
    def prediction_set(self,p):
        if not self.fitted:raise ValueError('Conformal not fitted')
        p=probability(p);return tuple(k for k,v in [(0,1-p),(1,p)] if 1-v<=self.quantile)

def person_diagnostic_conformal(p,y,subjects,window_keys,design,alpha=.1):
    if not (len(p)==len(y)==len(subjects)==len(window_keys)):raise ValueError('Diagnostic lengths disagree')
    design.require(subjects,'conformal');selected={}
    for i,(subject,key) in enumerate(zip(subjects,window_keys)):
        rank=hashlib.sha256(('20260921|person_conformal|'+subject+'|'+repr(key)).encode()).hexdigest()
        if subject not in selected or rank<selected[subject][0]:selected[subject]=(rank,i)
    indices=[selected[s][1] for s in sorted(selected)]
    c=Conformal(alpha,strategy='one_window_per_person_diagnostic').fit(np.asarray(p)[indices],np.asarray(y)[indices],np.asarray(subjects)[indices],design)
    return c,indices

def unreliable(hr,reference,threshold=10):
    if reference is None or not np.isfinite(reference):raise ValueError('Independent valid reference required')
    return int(hr is None or not np.isfinite(hr) or abs(hr-reference)>threshold)

def corrected_error(reported_hr,reference):
    if reported_hr is None or not np.isfinite(reported_hr):return None
    if reference is None or not np.isfinite(reference):raise ValueError('Invalid reference')
    return float(abs(reported_hr-reference))

def fuse(primary,secondary,q):return (probability(secondary)+q*probability(primary))/(1+q)

def _bin(q,p):
    q=probability(q);p=probability(p)
    return min(3,int(q*4)),min(3,int(abs(2*p-1)*4))

@dataclass(frozen=True)
class PrimaryState:
    probability: float
    quality: float

class Planner:
    def __init__(self,secondary='ECG',penalty=.02):
        if secondary not in ('ECG','EDA','REFERENCE'):raise ValueError('Unknown tool')
        self.secondary=secondary;self.increment=(300 if secondary in ('ECG','REFERENCE') else 150)+5;self.penalty=penalty;self.fitted=False
    def fit(self,p,q,primary_loss,secondary_loss,subjects,design,latency_ratio):
        design.require(subjects,'policy')
        p=np.asarray(p);q=np.asarray(q);subjects=np.asarray(subjects);gain=np.asarray(primary_loss)-np.asarray(secondary_loss)
        if any(len(x)!=len(subjects) for x in (p,q,gain)) or not len(p) or not np.isfinite(gain).all():raise ValueError('Invalid policy observations')
        if not np.isfinite(latency_ratio) or latency_ratio<=0:raise ValueError('Invalid policy latency')
        if self.secondary=='REFERENCE' and latency_ratio!=1:raise ValueError('Oracle ratio must be conceptual 1')
        self.latency_ratio=float(latency_ratio);self.global_gain=float(person_weights(subjects)@gain);self.bins={};self.fit_people=sorted(set(subjects))
        keys=[_bin(a,b) if np.isfinite(a) and np.isfinite(b) else None for a,b in zip(q,p)]
        for key in set(keys)-{None}:
            use=np.array([k==key for k in keys]);m=len(set(subjects[use]))
            self.bins[key]=float(m/(m+5)*(person_weights(subjects[use])@gain[use])+5/(m+5)*self.global_gain) if m>=3 else self.global_gain
        self.fitted=True;return self
    def should_call(self,state,remaining):
        # Only these three variables; no secondary value, reference, label or outcome.
        if not self.fitted:raise ValueError('Unfitted planner')
        if type(remaining) is not int:raise ValueError('Budget must be integer hundredths')
        gain=self.bins.get(_bin(state.quality,state.probability),self.global_gain)
        return remaining>=self.increment and gain>self.penalty*(self.increment/100)*self.latency_ratio

@dataclass(frozen=True)
class SecondaryResult:
    probability: float | None
    quality: float
    technical_valid: bool=True
    reference_hr: float | None=None

@dataclass(frozen=True)
class Decision:
    prediction_set: tuple
    predicted: int | None
    abstain: bool
    cost: int
    secondary_called: bool
    probability: float | None
    reported_hr: float | None=None
    reason: str=''

def decide(primary,p_quality_threshold,s_quality_threshold,sets,budget,planner=None,secondary=None,strategy='adaptive',hr=None,is_hr=False):
    if type(budget) is not int or budget<0:raise ValueError('Integer nonnegative budget required')
    if strategy not in ('attempt','quality','fixed_fusion','adaptive'):raise ValueError('Unknown strategy')
    if primary is None:return Decision((),None,True,0,False,None,None,'primary_unavailable')
    p=probability(primary.probability);q=probability(primary.quality)
    if strategy=='attempt':
        if budget<100:return Decision((),None,True,0,False,None,None,'budget')
        failed=is_hr and (hr is None or not np.isfinite(hr))
        label=0 if is_hr else int(p>=.5)
        return Decision((label,),None if failed else label,failed,100,False,p,hr if is_hr and not failed else None,'hr_unavailable' if failed else '')
    if budget<115:return Decision((),None,True,0,False,None,None,'budget')
    calibrators=list(sets.values())
    if not calibrators or any(not c.fitted or c.key!=(strategy,budget,10) for c in calibrators):
        raise ValueError('Conformal must match this frozen strategy, budget and duration')
    if len({id(c) for c in calibrators})!=1:
        raise ValueError('Primary/fusion paths require the same strategy-execution conformal calibration')
    cost=115;called=False;corrected=None;qpass=q>=p_quality_threshold
    want=strategy=='fixed_fusion' or (strategy=='adaptive' and planner is not None and planner.should_call(primary,budget-cost))
    if want and planner is not None and budget-cost>=planner.increment:
        called=True;cost+=planner.increment
        if secondary is None:raise ValueError('Selected secondary callback missing')
        s=secondary()  # Deliberately lazy: this is the first access to secondary.
        if not s.technical_valid:return Decision((),None,True,cost,True,None,None,'secondary_unavailable')
        if planner.secondary=='REFERENCE':
            corrected=s.reference_hr
            if corrected is None or not np.isfinite(corrected):return Decision((),None,True,cost,True,None,None,'reference_unavailable')
            return Decision((0,),0,False,cost,True,0.,float(corrected),'offline_reference_simulation')
        if s.probability is None:return Decision((),None,True,cost,True,None,None,'secondary_unavailable')
        p=fuse(p,s.probability,q);qpass=probability(s.quality)>=s_quality_threshold
    key='fusion' if called else 'primary'
    if key not in sets:raise ValueError('Strategy-specific conformal required')
    prediction_set=sets[key].prediction_set(p);single=len(prediction_set)==1
    accepted=single and qpass and (not is_hr or prediction_set==(0,) and hr is not None and np.isfinite(hr))
    return Decision(prediction_set,prediction_set[0] if accepted else None,not accepted,cost,called,p,hr if is_hr and accepted else None,'accepted' if accepted else 'set_quality_or_hr_gate')
