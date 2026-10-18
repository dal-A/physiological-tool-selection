import numpy as np
import pytest
from pmtools.schema import Signal,Window

class ToyDesign:
    """Synthetic fixture, not a substitute for frozen real-person allocation."""
    def require(self,subjects,role):
        if not len(subjects) or any(not str(s).startswith(role+':') for s in subjects):raise ValueError('Role leakage')
    def fold(self,subject):self.require([subject],'train');return int(subject.split(':')[1])%3

@pytest.fixture
def toy_design():return ToyDesign()

@pytest.fixture
def bvp():
    t=np.arange(640)/64
    return Signal(np.sin(2*np.pi*1.2*t)+.1*np.sin(2*np.pi*2.4*t),64,'arbitrary_unit')

@pytest.fixture
def acc():return Signal(np.tile([1.,0.,0.],(320,1)),32,'source_unit')

def ecg_window(start=0,record='r',label=0,valid=True):
    t=np.arange(2000)/200;x=np.sin(2*np.pi*1.2*t)
    return Window('train:0','SHDB-AF',record,'train',start,start+10,{'ECG':Signal(x,200,'mV')},label=label,technical_valid=valid)
