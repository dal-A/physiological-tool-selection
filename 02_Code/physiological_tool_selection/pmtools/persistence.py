"""Versioned scientific model checkpoint, no loading old experiment objects."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .protocol import PROTOCOL_SHA,ALLOCATION_SHA
from .models import CNNModel,TinyCNN,ClassicalModel

def save_model(model,directory):
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Never overwrite a model/results directory')
    directory.mkdir(parents=True)
    meta={'version':'P7_protocol_p6_v1','protocol_sha256':PROTOCOL_SHA,'allocations_sha256':ALLOCATION_SHA}
    if isinstance(model,CNNModel):
        meta.update({'family':'cnn','channels':model.model.blocks[0].in_channels,'training':model.metadata})
        torch.save(model.model.state_dict(),directory/'state.pt')
        arrays={f'fold{f}_epoch{e}':draw for f,fold in enumerate(model.draws) for e,draw in enumerate(fold)}
        np.savez_compressed(directory/'training_draws.npz',**arrays)
    elif isinstance(model,ClassicalModel):
        meta.update({'family':'classical','fit_people':model.subjects,'feature_schema':model.feature_schema});joblib.dump(model,directory/'model.joblib')
    else:raise TypeError('Unsupported model type')
    (directory/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def load_model(directory):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if meta['version']!='P7_protocol_p6_v1' or meta['protocol_sha256']!=PROTOCOL_SHA or meta['allocations_sha256']!=ALLOCATION_SHA:raise ValueError('Old/incompatible model refused')
    if meta['family']=='cnn':
        model=TinyCNN(meta['channels']);model.load_state_dict(torch.load(directory/'state.pt',map_location='cpu',weights_only=True));model.eval();return CNNModel(model,meta['training'],[])
    if meta['family']=='classical':return joblib.load(directory/'model.joblib') # Trusted local model files only.
    raise ValueError('Unknown family')
