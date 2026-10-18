"""Causal stream control; reference-independent resets; offline event scoring."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from .schema import Signal

class BOCPD:
    """Gaussian NIG run-length filter with a normalized 256-bin truncation.

    Run length zero starts a new segment at the current observation; its NIG
    prior is updated with that observation. Growth beyond max_run_length is
    pruned and the retained posterior renormalized, not lost or called zero.
    """
    def __init__(self,hazard=1/60,max_run_length=256):
        if not 0<hazard<1 or max_run_length<4:raise ValueError('Invalid BOCPD settings')
        self.hazard=hazard;self.maximum=max_run_length;self.reset()
    def reset(self):
        self.posterior=np.array([1.]);self.mu=np.array([0.]);self.kappa=np.array([1.]);self.alpha=np.array([1.]);self.beta=np.array([1.]);self.ticks=0;self.last_change=-6
    @staticmethod
    def _update(mu,kappa,alpha,beta,x):
        return (kappa*mu+x)/(kappa+1),kappa+1,alpha+.5,beta+kappa*(x-mu)**2/(2*(kappa+1))
    def update(self,probability):
        if not np.isfinite(probability) or not 0<=probability<=1:raise ValueError('Invalid light probability')
        p=np.clip(probability,.001,.999);x=float(np.log(p/(1-p)))
        scale=np.sqrt(self.beta*(self.kappa+1)/(self.alpha*self.kappa))
        logpred=student_t.logpdf(x,2*self.alpha,loc=self.mu,scale=scale)
        priorpred=student_t.logpdf(x,2,loc=0,scale=np.sqrt(2))
        logprior=np.log(np.maximum(self.posterior,np.finfo(float).tiny))
        growth=logprior+np.log1p(-self.hazard)+logpred
        cp=np.log(self.hazard)+priorpred
        logs=np.r_[cp,growth][:self.maximum+1];self.posterior=np.exp(logs-logsumexp(logs))
        arrays=self._update(np.r_[0.,self.mu],np.r_[1.,self.kappa],np.r_[1.,self.alpha],np.r_[1.,self.beta],x)
        self.mu,self.kappa,self.alpha,self.beta=[a[:len(self.posterior)] for a in arrays]
        self.ticks+=1;score=float(self.posterior[:4].sum())
        change=self.ticks>=6 and self.ticks-self.last_change>=6 and score>.15
        if change:self.last_change=self.ticks
        return score,change

@dataclass(frozen=True)
class ScheduledCall:
    subject: str
    record: str
    start: float
    end: float
    duration: int
    ecg: Signal
    cost: int
    change_score: float

class MonitorScheduler:
    def __init__(self,adaptive=True):self.adaptive=adaptive;self.bocpd=BOCPD();self.reset()
    def reset(self):
        self.buffer=[];self.previous=None;self.fast_remaining=0;self.total_cost=0;self.calls=0;self.bocpd.reset()
    def push(self,block,light_probability):
        # RuntimeBlock has no rhythm truth, event annotations or HR reference.
        previous=self.previous
        gap=previous is not None and (block.subject!=previous.subject or block.record!=previous.record or block.start!=previous.end)
        if gap or not block.technical_valid:
            self.buffer=[];self.fast_remaining=0;self.bocpd.reset()
        self.previous=block
        if not block.technical_valid:return None
        if block.end-block.start!=10:raise ValueError('Monitor tick must be 10 seconds')
        self.total_cost+=15;score,change=self.bocpd.update(light_probability)
        if self.adaptive and change and self.fast_remaining==0:
            self.fast_remaining=6;self.buffer=[]
        if self.fast_remaining:
            duration=10;signal=block.signals['ECG'];self.fast_remaining-=1;start=block.start
            self.buffer=[]
        else:
            self.buffer.append(block)
            if len(self.buffer)<3:return None
            duration=30;start=self.buffer[0].start;native=self.buffer[0].signals['ECG']
            if any(x.signals['ECG'].fs!=native.fs for x in self.buffer):raise ValueError('Mixed native rate')
            signal=Signal(np.concatenate([x.signals['ECG'].values for x in self.buffer]),native.fs,native.unit);self.buffer=[]
        self.total_cost+=300*(duration//10);self.calls+=1
        return ScheduledCall(block.subject,block.record,start,block.end,duration,signal,300*(duration//10),score)

def union_intervals(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((float(a),float(b)))
    return out

def intersect_intervals(a,b):
    a=union_intervals(a);b=union_intervals(b);i=j=0;out=[]
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<hi:out.append((lo,hi))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return out

def in_intervals(time,intervals):return any(a<=time<b for a,b in intervals)

def duration(intervals):return float(sum(b-a for a,b in union_intervals(intervals)))

def alarm_episodes(decisions,exposure_start,exposure_end):
    """decisions sorted (time,active), including gap/unavailable reset decisions.

    Caller includes reset at acquisition-gap start. Late decisions are not
    backdated or matched after exposure ends. Abstention is inactive.
    """
    ordered=sorted(decisions,key=lambda x:x[0]);onsets=[];spans=[];active=False;last=exposure_start
    for when,state in ordered:
        if when<exposure_start:raise ValueError('Backdated decision')
        if when>=exposure_end:break
        if active and when>last:spans.append((last,when))
        if state and not active:onsets.append(float(when))
        active=bool(state);last=when
    if active and last<exposure_end:spans.append((last,exposure_end))
    return onsets,union_intervals(spans)

def score_events(events,alarm_onsets,negative_intervals,valid_intervals,alarm_spans=(),horizon=120):
    """Single record/clock; aggregate records only after scoring each record."""
    if horizon<=0:raise ValueError('Invalid horizon')
    events=sorted(events,key=lambda x:x[0]);alarms=sorted(alarm_onsets);used=set();matches=[]
    for index,(start,end) in enumerate(events):
        if start>=end:raise ValueError('Invalid target event')
        eligible=[j for j,a in enumerate(alarms) if j not in used and start<=a<end and a<=start+horizon and in_intervals(a,valid_intervals)]
        if eligible:
            j=eligible[0];used.add(j);matches.append({'event':index,'onset':start,'alarm':alarms[j],'delay':alarms[j]-start,'detected':True})
        else:matches.append({'event':index,'onset':start,'alarm':None,'delay':float(horizon),'detected':False})
    exposure=intersect_intervals(negative_intervals,valid_intervals);hours=duration(exposure)/3600
    false=[a for a in alarms if in_intervals(a,exposure)]
    detected=[x['delay'] for x in matches if x['detected']]
    return {'matches':matches,'events':len(events),'detected':len(detected),'missed':len(events)-len(detected),'detection_rate':len(detected)/len(events) if events else None,'miss_penalized_delay':float(np.mean([x['delay'] for x in matches])) if matches else None,'detected_only_delay':float(np.mean(detected)) if detected else None,'false_alarms':len(false),'negative_hours':hours,'FAR_per_non_target_hour':len(false)/hours if hours else None,'negative_alarm_occupancy':duration(intersect_intervals(alarm_spans,exposure))/duration(exposure) if hours else None}

@dataclass(frozen=True)
class PolicyMonitorRecord:
    subject: str
    record: str
    strategy: str
    # Rows: end+latency, probability, quality_pass, scheduled known-negative.
    calls: tuple
    events: tuple
    negative_intervals: tuple
    valid_intervals: tuple
    start: float
    end: float
    resets: tuple=()

def calibrate_alarm(records,design,cap=.05,detection_target=.9):
    design.require([r.subject for r in records],'policy')
    if {r.strategy for r in records}!={'fixed','adaptive'}:raise ValueError('Both policy strategies required')
    if not 0<=cap<=1 or not 0<=detection_target<=1:raise ValueError('Invalid calibration targets')
    values=np.array([c[1] for r in records for c in r.calls if np.isfinite(c[1])])
    if not len(values):raise ValueError('No policy heavy probabilities')
    candidates=np.r_[np.unique(np.quantile(values,np.linspace(0,1,101),method='linear')),np.inf];summaries=[]
    for threshold in candidates:
        results={'fixed':[],'adaptive':[]};negative={k:[0,0] for k in results}
        for r in records:
            decisions=[(t,bool(qpass and p>=threshold)) for t,p,qpass,neg in r.calls]+[(t,False) for t in r.resets]
            onsets,spans=alarm_episodes(decisions,r.start,r.end)
            result=score_events(r.events,onsets,r.negative_intervals,r.valid_intervals,spans)
            results[r.strategy].append((r.subject,result))
            for t,p,qpass,neg in r.calls:
                if neg and t<r.end:
                    negative[r.strategy][1]+=1;negative[r.strategy][0]+=int(qpass and p>=threshold)
        # Same locked denominator; quality abstentions count as negative decisions.
        fractions={k:(v[0]/v[1] if v[1] else None) for k,v in negative.items()}
        if any(v is None for v in fractions.values()):raise ValueError('No policy non-target decisions')
        stat={}
        for strategy,rows in results.items():
            per={}
            for subject,result in rows:per.setdefault(subject,[]).append(result)
            detected=sum(x['detected'] for _,x in rows);event_count=sum(x['events'] for _,x in rows)
            if not event_count:raise ValueError('No policy qualifying events')
            delays=[]
            for subject,rs in per.items():
                ds=[m['delay'] for x in rs for m in x['matches']]
                if ds:delays.append(float(np.mean(ds)))
            exposure=sum(x['negative_hours'] for _,x in rows)
            if not exposure:raise ValueError('No policy non-target exposure')
            stat[strategy]={'detection':detected/event_count,'far':sum(x['false_alarms'] for _,x in rows)/exposure,'delay':float(np.mean(delays))}
        maxfrac=max(fractions.values());detection=min(v['detection'] for v in stat.values());far=max(v['far'] for v in stat.values());delay=max(v['delay'] for v in stat.values())
        summaries.append({'threshold':float(threshold),'cap_pass':maxfrac<=cap,'target_feasible':maxfrac<=cap and detection>=detection_target,'max_negative_positive_fraction':maxfrac,'minimum_detection':detection,'maximum_FAR':far,'maximum_person_delay':delay,'strategy_metrics':stat})
    feasible=[x for x in summaries if x['target_feasible']]
    if feasible:chosen=min(feasible,key=lambda x:(x['maximum_FAR'],x['maximum_person_delay'],-x['threshold']))
    else:chosen=min((x for x in summaries if x['cap_pass']),key=lambda x:(-x['minimum_detection'],x['maximum_FAR'],-x['threshold']))
    return {'selected':chosen,'candidate_count':len(summaries),'candidates':summaries,'fit_people':sorted(set(r.subject for r in records)),'conformal_not_used':True}

def final_alarm(prediction_set,quality_pass,probability,threshold):
    return prediction_set==(1,) and bool(quality_pass) and probability>=threshold
