"""Reproducible research provenance, distinct from internal task acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import pandas as pd
from run_statistics import check_inputs,read_json,SCIENCE
from pmtools.experiment_io import write_json,write_csv
from pmtools.protocol import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--p9-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();target=out/'result_manifest.json'
    if target.exists():raise FileExistsError('Never overwrite sealed result manifests')
    input_files=check_inputs(a.p9_root)
    captions=pd.read_csv(out/'figure_captions.csv').sort_values('figure').to_dict('records')
    assert len(captions)==20 and len({r['figure'] for r in captions})==20
    for row in captions:
        row['input_result_version']=a.p9_root.name;row['result_version']=out.name
        assert (out/'tables'/row['source_table']).is_file()
    write_csv(out/'figure_captions.csv',captions)
    # Retain scientifically relevant generation metadata; task progress and
    # acceptance are documented exclusively in the separate editorial folder.
    for filename in ('main_analysis_status.json','diagnostics_stage_status.json','sensitivity_stage_status.json'):
        data=read_json(out/filename)
        for key in ('P10_complete','P11_started','sensitivity_complete','independent_acceptance_pending',
                    'final_independent_numerical_and_visual_acceptance_pending'):
            data.pop(key,None)
        data['metadata_kind']='scientific_generation_snapshot_not_task_acceptance'
        write_json(out/filename,data)
    write_json(out/'reproducibility.json',dict(version=out.name,input_result_version=a.p9_root.name,
        generated_UTC=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(SCIENCE/'config/protocol.json'),
        allocation_sha256=sha256(SCIENCE/'config/person_allocations.json'),seed=20260921,cluster_replicates=10000,
        classification_sensitivity_runs=48,monitoring_sensitivity_runs=20,new_HR_target_or_alignment_models=6,
        scientific_figures=20,raw_signals_redistributed=False,source_domains_reported_separately=True,
        clinical_efficacy_established=False,
        original_P9_scientific_code_sha256=read_json(a.p9_root/'af_afl_classical_full/run_config.json')['code_files'],
        analysis_code_sha256={str(f.relative_to(Path(__file__).parent)):sha256(f) for f in sorted(Path(__file__).parent.rglob('*')) if f.is_file() and f.suffix in ('.py','.md')},
        analysis_config_is_initial_generation_snapshot=True,
        timings='unchanged original P9 isolated CPU timings; sensitivities replay recorded latencies'))
    files=[dict(file=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=sha256(f))
           for f in sorted(out.rglob('*')) if f.is_file() and f!=target]
    write_json(target,dict(version=out.name,input_result_version=a.p9_root.name,files=files,input_manifest=input_files))
    print('RESEARCH_PROVENANCE_MANIFEST_GENERATED',len(files),flush=True)


if __name__=='__main__':main()
