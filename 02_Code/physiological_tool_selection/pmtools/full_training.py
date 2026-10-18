"""Fresh full model fits; no smoke cap, cached checkpoint or evaluation optimization."""
import numpy as np
from .models import ClassicalModel, fit_cnn
from .persistence import save_model, load_model
from .signals import quality, estimate_hr, UnavailableSignal
from .safety import quality_threshold, unreliable
from .quick_classification import inputs
from .experiment_io import write_json, write_csv


def full_fit(windows, modality, family, design, directory, is_hr=False):
    arrays=[]; labels=[]; people=[]; rows=[]; qs=[]; qpeople=[]
    for i,w in enumerate(windows):
        runtime=w.runtime()
        q=quality(runtime.signals[modality],modality,runtime.signals.get('ACC') if modality=='BVP' else None)
        qs.append(q);qpeople.append(w.subject)
        row={'subject':w.subject,'record':w.record,'start':w.start,'duration':w.end-w.start,
             'label':None,'included_in_fit':False,'reason':None}
        try:
            if not runtime.technical_valid: raise UnavailableSignal('Source technical unavailable')
            target=unreliable(estimate_hr(runtime.signals['BVP']),w.reference_hr) if is_hr else w.label
            row['label']=target
            if target not in (0,1): raise UnavailableSignal('Unknown or mixed training reference')
            x=inputs(runtime,modality,family)
            arrays.append(x);labels.append(target);people.append(w.subject);row['included_in_fit']=True
        except UnavailableSignal as exc: row['reason']=str(exc)
        rows.append(row)
        if (i+1)%5000==0: print('TRAIN_PREPARED',modality,family,i+1,flush=True)
    if not arrays: raise ValueError('No usable training inputs')
    X=np.stack(arrays); del arrays
    y=np.asarray(labels,dtype=int)
    print('FIT_BEGIN',modality,family,'windows',len(y),'people',len(set(people)),
          'max_epochs',25 if family=='cnn' else None,flush=True)
    model=(ClassicalModel().fit(X,y,people,design) if family=='classical' else
           fit_cnn(X,y,people,design,max_epochs=25,patience=5,modality=modality))
    return persist_fit(model,X,rows,qs,qpeople,design,directory)


def persist_fit(model,X,rows,qs,qpeople,design,directory):
    save_model(model,directory); reloaded=load_model(directory)
    ids=np.unique(np.linspace(0,len(X)-1,min(32,len(X)),dtype=int))
    delta=float(np.max(np.abs(model.predict(X[ids])-reloaded.predict(X[ids]))))
    if delta>1e-7: raise AssertionError('Saved model prediction changed')
    write_csv(directory/'fit_windows.csv',rows)
    write_json(directory/'roundtrip.json',{'maximum_probability_difference':delta,
               'checked_windows':len(ids),'checked_training_indices':ids,'passed':True})
    write_json(directory/'fit_configuration.json',{'preset':'full','new_initialization':True,
               'max_epochs':25 if hasattr(model,'metadata') else None,'patience':5,
               'no_evaluation_or_calibration_for_optimization':True,
               'usable_fit_windows':len(X),'fit_people':sorted({r['subject'] for r in rows if r['included_in_fit']}),
               'candidate_windows':len(rows),'technical_or_unknown_exclusions':sum(not r['included_in_fit'] for r in rows)})
    print('FIT_SAVED',directory,flush=True)
    return reloaded,quality_threshold(qs,qpeople,design),rows
