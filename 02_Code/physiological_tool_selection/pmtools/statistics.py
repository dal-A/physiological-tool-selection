"""Person-level endpoints. Undefined risks are None, never fabricated zeros."""
import numpy as np
from scipy.stats import binomtest,norm,beta
from sklearn.metrics import roc_auc_score

def decision_loss(accepted,wrong):
    accepted=np.asarray(accepted,dtype=bool);wrong=np.asarray(wrong,dtype=bool)
    if accepted.shape!=wrong.shape:raise ValueError('Loss shape mismatch')
    return np.where(accepted,wrong.astype(float),.2)

def report_metrics(accepted,wrong,errors=None):
    a=np.asarray(accepted,dtype=bool);w=np.asarray(wrong,dtype=bool);n=len(a);retained=int(a.sum())
    if len(w)!=n:raise ValueError('Metric length mismatch')
    out={'eligible_windows':n,'reported_windows':retained,'report_fraction':retained/n if n else None,'selective_risk':float(w[a].mean()) if retained else None,'dangerous_acceptance_all_windows':float((a&w).mean()) if n else None,'no_reports':not retained,'mean_decision_loss':float(decision_loss(a,w).mean()) if n else None}
    out['selective_risk_undefined_reason']=None if retained else 'zero_reported_windows'
    if errors is not None:
        err=np.asarray(errors,dtype=float)
        if len(err)!=n:raise ValueError('Error length mismatch')
        if np.any(a & ~np.isfinite(err)):raise ValueError('Nonfinite accepted HR must abstain')
        out['retained_MAE']=float(err[a].mean()) if retained else None
    return out

def safe_auroc(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if len(y)!=len(p) or not np.isfinite(p).all():raise ValueError('Invalid AUROC observations')
    return float(roc_auc_score(y,p)) if len(set(y))==2 else None

def person_means(values,subjects,time_keys=None):
    values=np.asarray(values,dtype=float);subjects=np.asarray(subjects)
    if len(values)!=len(subjects):raise ValueError('Aggregation mismatch')
    out={}
    for s in sorted(set(subjects)):
        ids=np.flatnonzero(subjects==s)
        if time_keys is not None:
            if len(time_keys)!=len(values):raise ValueError('Time key mismatch')
            bins={}
            for i in ids:bins.setdefault(time_keys[i],[]).append(values[i])
            vals=np.array([np.mean(v) for v in bins.values()])
        else:vals=values[ids]
        out[s]=float(np.mean(vals)) if len(vals) and np.isfinite(vals).all() else None
    return out

def balanced_person_mean(values,labels=None):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():return None
    if labels is None:return float(a.mean()) if len(a) else None
    labels=np.asarray(labels)
    if len(labels)!=len(a) or set(labels)!={0,1}:raise ValueError('Both person classes required')
    return float(.5*np.mean(a[labels==0])+.5*np.mean(a[labels==1]))

def paired_bootstrap(left,right,labels=None,replicates=10000,seed=20260921):
    """Aligned person arrays, identical paired draws, source-specific invocation."""
    left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    if len(left)!=len(right) or not len(left) or replicates<1:raise ValueError('Invalid paired support')
    diff=left-right;n=len(diff);rng=np.random.default_rng(seed)
    if labels is None:
        draws=rng.integers(n,size=(replicates,n));sample=diff[draws].mean(axis=1);estimate=balanced_person_mean(diff)
    else:
        labels=np.asarray(labels)
        if len(labels)!=n or set(labels)!={0,1}:raise ValueError('Both person strata required')
        pieces=[]
        for k in (0,1):
            group=np.flatnonzero(labels==k);draws=rng.choice(group,size=(replicates,len(group)),replace=True);pieces.append(diff[draws].mean(axis=1))
        sample=.5*(pieces[0]+pieces[1]);estimate=balanced_person_mean(diff,labels)
    valid=sample[np.isfinite(sample)];enough=len(valid)>=.95*replicates
    return {'estimate':estimate,'CI95':np.quantile(valid,[.025,.975],method='linear').tolist() if enough else None,'valid_replicates':len(valid),'requested_replicates':replicates,'undefined_reason':None if enough else 'less_than_95_percent_valid','independent_people':n,'seed':seed}

def paired_sign_test(differences):
    d=np.asarray(differences,dtype=float)
    if not np.isfinite(d).all():raise ValueError('Undefined paired differences')
    nonzero=d[d!=0];n=len(nonzero)
    return {'p':float(binomtest(int((nonzero<0).sum()),n,.5).pvalue) if n else 1.,'effective_people':n,'ties':int((d==0).sum()),'null':'probability of improvement = 0.5, not mean contrast = 0'}

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid p-values')
    order=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]);result=np.empty_like(p);result[order]=np.minimum(1,adjusted);return result

def zero_event_bounds(n,alpha=.05):
    if n<1 or not 0<alpha<1:raise ValueError('Independent n required')
    z=norm.ppf(1-alpha/2);wilson=z*z/(n+z*z)
    return {'people':n,'Wilson_two_sided_upper':float(wilson),'Clopper_Pearson_one_sided_upper':float(1-alpha**(1/n)),'Clopper_Pearson_two_sided_upper':float(beta.ppf(1-alpha/2,1,n)),'not_zero_risk_proof':True}
