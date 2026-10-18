"""Scientific data-interface validation, not quick/full model evaluation."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pmtools.adapters import DatasetAdapters

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--source',required=True,choices=['VITALDB','PPG-DaLiA','WESAD','PPGSPORT','GALAXYPPG','Stress-Predict','WearableDevice_V1','WearableDevice_V2','SHDB-AF'])
    parser.add_argument('--role',choices=['train','policy','conformal','evaluation','supplementary_evaluation'])
    parser.add_argument('--max-people',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output filename')
    if args.max_people is not None and args.max_people<1:raise ValueError('Positive limit required')
    adapter=DatasetAdapters(args.data_root);people=adapter.people(args.source,args.role)
    if args.max_people:people=people[:args.max_people]
    rows=[]
    for person in people:
        count=0;bad=0
        for w in adapter.iter_person(person['subject']):w.validate();count+=1;bad+=not w.technical_valid
        rows.append({'subject':person['subject'],'role':person['role'],'windows':count,'technical_unavailable':bad})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'protocol_id':adapter.design.protocol['protocol_id'],'data_interface_only':True,'performance_computed':False,'people':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Validated people:',len(rows),'windows:',sum(x['windows'] for x in rows))
if __name__=='__main__':main()
