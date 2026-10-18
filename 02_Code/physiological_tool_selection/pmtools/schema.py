"""Separate offline reference metadata from causal runtime input."""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

@dataclass(frozen=True)
class Signal:
    values: np.ndarray
    fs: float
    unit: str

    def validate(self, duration, channels=None):
        a=np.asarray(self.values)
        if not np.isfinite(self.fs) or self.fs <= 0: raise ValueError('Invalid sampling rate')
        expected=duration*self.fs
        if abs(expected-round(expected)) > 1e-7 or len(a)!=round(expected): raise ValueError('Signal length/time support mismatch')
        if a.ndim not in (1,2): raise ValueError('Invalid signal dimensions')
        if channels is not None and (1 if a.ndim==1 else a.shape[1])!=channels: raise ValueError('Channel mismatch')
        if not self.unit: raise ValueError('Declared unit required')
        return self

@dataclass(frozen=True)
class RuntimeBlock:
    subject: str
    record: str
    start: float
    end: float
    signals: dict
    technical_valid: bool

@dataclass(frozen=True)
class Window:
    subject: str
    source: str
    record: str
    role: str
    start: float
    end: float
    signals: dict
    label: int | None = None
    reference_hr: float | None = None
    activity: str | int | None = None
    wrist: str | None = None
    technical_valid: bool = True
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if self.end <= self.start: raise ValueError('Invalid half-open window')
        if self.label not in (None,-1,0,1): raise ValueError('Nonbinary target')
        for signal in self.signals.values(): signal.validate(self.end-self.start)
        return self

    def runtime(self):
        # No label, independent HR, activity/task, or reference intervals exposed.
        return RuntimeBlock(self.subject,self.record,self.start,self.end,self.signals,self.technical_valid)

def pure_wesad_label(raw):
    raw=np.asarray(raw).reshape(-1)
    if not len(raw) or not np.isfinite(raw).all() or len(np.unique(raw))!=1: return None
    v=int(raw[0])
    if v==2: return 1
    if v in (1,3,4): return 0
    return None

def activity_mode(values):
    values=np.asarray(values).reshape(-1)
    if not len(values): return None
    counts=Counter(int(x) for x in values)
    return min(counts,key=lambda x:(-counts[x],x))

def dalia_reference(labels, start, end, wholly_inside=False):
    labels=np.asarray(labels,dtype=float).reshape(-1)
    left=2*np.arange(len(labels)); centers=left+4
    use=(left>=start)&(left+8<=end) if wholly_inside else (centers>=start)&(centers<end)
    values=labels[use & np.isfinite(labels)]
    return float(np.median(values)) if len(values) else None

def check_dalia_count(label_count, duration):
    expected=max(0,int(np.floor((duration-8)/2))+1)
    if label_count!=expected: raise ValueError(f'DaLiA reference count mismatch: {label_count} != {expected}; do not optimize an offset')
    return expected

def contiguous_triples(windows):
    """Nonoverlapping 30s model-training contexts; labels used only offline."""
    buffer=[]
    for w in windows:
        compatible=not buffer or (w.subject==buffer[-1].subject and w.record==buffer[-1].record and w.start==buffer[-1].end and w.role==buffer[-1].role and w.label==buffer[-1].label)
        if not compatible or not w.technical_valid or w.label not in (0,1): buffer=[]
        if not w.technical_valid or w.label not in (0,1): continue
        buffer.append(w)
        if len(buffer)==3:
            yield Window(w.subject,w.source,w.record,w.role,buffer[0].start,w.end,{'ECG':Signal(np.concatenate([x.signals['ECG'].values for x in buffer]),w.signals['ECG'].fs,w.signals['ECG'].unit)},w.label).validate()
            buffer=[]
