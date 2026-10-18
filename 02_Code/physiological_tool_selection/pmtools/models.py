"""Training-person-only classical/CNN fitting with grouped epoch selection."""
from dataclasses import dataclass
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

SEED=20260921

def person_weights(subjects):
    subjects=np.asarray(subjects);unique,counts=np.unique(subjects,return_counts=True)
    if not len(unique): raise ValueError('Empty training set')
    lookup=dict(zip(unique,counts))
    return np.array([1/(len(unique)*lookup[s]) for s in subjects])

def balanced_weights(labels,subjects):
    y=np.asarray(labels);w=person_weights(subjects)
    if set(np.unique(y))!={0,1}: raise ValueError('Training requires both classes')
    mass=np.array([w[y==k].sum() for k in (0,1)])
    b=w/(2*mass[y.astype(int)])
    return b/b.sum()*len(y)

class ClassicalModel:
    def fit(self,X,y,subjects,design,feature_schema=None):
        design.require(subjects,'train')
        X=np.asarray(X,dtype=float); y=np.asarray(y,dtype=int)
        if len(X)!=len(y) or len(y)!=len(subjects) or not np.isfinite(X).all(): raise ValueError('Invalid train features')
        self.subjects=sorted(set(subjects))
        from .signals import BASE_FEATURE_NAMES,ACC_FEATURE_NAMES,EDA_FEATURE_NAMES,LIGHT_FEATURE_NAMES
        schemas={15:BASE_FEATURE_NAMES,21:BASE_FEATURE_NAMES+ACC_FEATURE_NAMES,6:EDA_FEATURE_NAMES,7:LIGHT_FEATURE_NAMES}
        self.feature_schema=list(feature_schema or schemas.get(X.shape[1],tuple('synthetic_'+str(i) for i in range(X.shape[1]))))
        if len(self.feature_schema)!=X.shape[1]:raise ValueError('Feature schema mismatch')
        self.scaler=StandardScaler().fit(X,sample_weight=person_weights(subjects))
        self.model=LogisticRegression(C=1,max_iter=1000,random_state=SEED).fit(self.scaler.transform(X),y,sample_weight=balanced_weights(y,subjects))
        return self
    def predict(self,X): return self.model.predict_proba(self.scaler.transform(np.asarray(X)))[:,1]

def eda_global_scaler(arrays,subjects,design):
    design.require(subjects,'train')
    if len(arrays)!=len(subjects) or any(not np.asarray(a).size or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid EDA training signals')
    means=np.array([np.mean(a) for a in arrays]);squares=np.array([np.mean(np.square(a)) for a in arrays]);w=person_weights(subjects)
    mean=float(w@means);variance=max(0.,float(w@squares)-mean*mean)
    return mean,float(np.sqrt(variance))

class TinyCNN(nn.Module):
    def __init__(self,channels=1):
        super().__init__();blocks=[];previous=channels
        for i,(out,k) in enumerate(zip((16,32,64),(9,7,5))):
            blocks.extend([nn.Conv1d(previous,out,k,padding=k//2),nn.BatchNorm1d(out),nn.ReLU()])
            if i<2:blocks.append(nn.MaxPool1d(2))
            previous=out
        self.blocks=nn.Sequential(*blocks);self.dropout=nn.Dropout(.2);self.head=nn.Linear(64,2)
    def forward(self,x):return self.head(self.dropout(self.blocks(x).mean(dim=-1)))

def hierarchical_draws(y,subjects,size,rng):
    y=np.asarray(y);subjects=np.asarray(subjects)
    if set(np.unique(y))!={0,1}:raise ValueError('Training requires both classes')
    candidates={k:{s:np.flatnonzero((y==k)&(subjects==s)) for s in np.unique(subjects[y==k])} for k in (0,1)}
    out=[]
    for _ in range(size):
        label=int(rng.integers(2)); person=rng.choice(sorted(candidates[label]));out.append(int(rng.choice(candidates[label][person])))
    return np.asarray(out,dtype=np.int64)

def _fit_epochs(X,y,subjects,epochs,seed,validation=None,patience=5):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model=TinyCNN(X.shape[1]).cpu();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;stale=0;history=[];draw_history=[]
    X=torch.tensor(X,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.long)
    for epoch in range(1,epochs+1):
        model.train();draws=hierarchical_draws(y,subjects,len(y),rng);draw_history.append(draws)
        for lo in range(0,len(draws),64):
            idx=draws[lo:lo+64];optimizer.zero_grad();loss=nn.functional.cross_entropy(model(X[idx]),yt[idx]);loss.backward();optimizer.step()
        if validation is not None:
            vx,vy,vs=validation;model.eval()
            with torch.no_grad():
                parts=[nn.functional.cross_entropy(model(torch.tensor(vx[j:j+64],dtype=torch.float32)),torch.tensor(vy[j:j+64],dtype=torch.long),reduction='none').numpy() for j in range(0,len(vy),64)]
            value=float(person_weights(vs)@np.concatenate(parts));history.append(value)
            if value<best:best=value;best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=patience:break
        else:best_epoch=epoch
    return model,best_epoch,history,draw_history

@dataclass
class CNNModel:
    model: TinyCNN
    metadata: dict
    draws: list
    def predict(self,X):
        self.model.eval();X=np.asarray(X,dtype='float32')
        if not len(X):return np.array([],dtype=float)
        if self.metadata.get('modality')=='EDA':
            mean,std=self.metadata['eda_scaler'];X=(X-mean)/(std+1e-6)
        with torch.no_grad():return np.concatenate([torch.softmax(self.model(torch.tensor(X[j:j+64])),dim=1)[:,1].numpy() for j in range(0,len(X),64)])

def fit_cnn(X,y,subjects,design,max_epochs=25,patience=5,modality='BVP'):
    """max_epochs<25 is permitted only for synthetic unit/smoke tests, never full."""
    design.require(subjects,'train'); X=np.asarray(X,dtype='float32');y=np.asarray(y,dtype=int);subjects=np.asarray(subjects)
    if X.ndim!=3 or len(X)!=len(y) or len(y)!=len(subjects) or not np.isfinite(X).all(): raise ValueError('Invalid CNN tensors')
    if not 1<=max_epochs<=25:raise ValueError('Epoch cap must be 1..25')
    if modality not in ('BVP','ECG','EDA'):raise ValueError('Unknown CNN modality')
    if modality=='EDA' and X.shape[1]!=1:raise ValueError('EDA requires raw microSiemens single channel')
    folds=np.array([design.fold(s) for s in subjects]);epochs=[];history=[];fold_people=[];all_draws=[];fold_scalers=[]
    for fold in range(3):
        val=folds==fold;train=~val
        if not val.any() or not train.any():raise ValueError('Empty grouped training fold')
        if set(subjects[train])&set(subjects[val]):raise ValueError('Person leakage')
        tx,vx=X[train],X[val]
        if modality=='EDA':
            # Raw EDA enters this API. Fit inside each training-person CV fold;
            # never use validation persons for the tonic-scale transformation.
            mean,std=eda_global_scaler(tx,subjects[train],design);tx=(tx-mean)/(std+1e-6);vx=(vx-mean)/(std+1e-6);fold_scalers.append([mean,std])
        model,best,h,draws=_fit_epochs(tx,y[train],subjects[train],max_epochs,SEED+fold,(vx,y[val],subjects[val]),patience)
        epochs.append(best);history.append(h);fold_people.append({'train':sorted(set(subjects[train])),'validation':sorted(set(subjects[val]))})
        # Original training indices, not private validation windows.
        all_draws.append([np.flatnonzero(train)[d] for d in draws])
    final_epochs=int(math.floor(float(np.median(epochs))+.5))
    scaler=eda_global_scaler(X,subjects,design) if modality=='EDA' else None
    finalX=(X-scaler[0])/(scaler[1]+1e-6) if scaler is not None else X
    model,_,_,draws=_fit_epochs(finalX,y,subjects,final_epochs,SEED+3)
    return CNNModel(model,{'seed':SEED,'max_epochs':max_epochs,'cv_best_epochs':epochs,'refit_epochs':final_epochs,'validation_loss':history,'cv_people':fold_people,'fit_people':sorted(set(subjects)),'CPU_threads':4,'modality':modality,'eda_scaler':scaler,'cv_eda_scalers':fold_scalers},all_draws+[draws])
