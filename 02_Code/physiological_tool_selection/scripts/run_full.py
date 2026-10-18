"""P9 frozen full-cohort experiment CLI; no quick overrides or warm start."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from threadpoolctl import threadpool_limits
import torch
from pmtools.full_classification import run_full_classification
from pmtools.full_monitor import run_full_monitor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--task',required=True,choices=['af_afl','hr_reliability','stress','continuous_monitoring'])
    parser.add_argument('--model',required=True,choices=['classical','cnn'])
    parser.add_argument('--preset',choices=['full'],default='full')
    args=parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        result=(run_full_monitor(args.data_root,args.output,args.model) if args.task=='continuous_monitoring' else
                run_full_classification(args.data_root,args.output,args.task,args.model))
    print('FULL_ARTIFACTS_SAVED_AWAITING_INDEPENDENT_ACCEPTANCE',result,flush=True)


if __name__=='__main__':main()
