"""P8 real-data pipeline smoke CLI. Full-cohort experiment is a separate gate."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pmtools.quick_classification import run_classification
from pmtools.quick_monitor import run_monitor

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--task',required=True,choices=['af_afl','hr_reliability','stress','continuous_monitoring'])
    parser.add_argument('--model',required=True,choices=['classical','cnn'])
    parser.add_argument('--preset',choices=['quick'],default='quick')
    args=parser.parse_args()
    if args.task=='continuous_monitoring':result=run_monitor(args.data_root,args.output,args.model)
    else:result=run_classification(args.data_root,args.output,args.task,args.model)
    print('QUICK_ARTIFACTS_SAVED',result,flush=True)

if __name__=='__main__':main()
