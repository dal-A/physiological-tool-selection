"""Expose policy feasibility and held-out results without changing thresholds."""
from pathlib import Path
import argparse,json
import pandas as pd
from run_statistics import read_json
from pmtools.experiment_io import write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    target=out/'tables/monitor_policy_and_evaluation_constraints.csv'
    if target.exists():raise FileExistsError(target)
    paths=sorted((out/'sensitivity_runs/continuous_monitoring').glob('*/*/settings_and_calibration.json'))
    assert len(paths)==20,'Wait for all monitoring sensitivities'
    rows=[]
    for path in paths:
        settings=read_json(path);selected=settings['policy']['selected']
        scores=pd.read_csv(path.parent/'person_metrics.csv')
        for strategy,g in scores.groupby('strategy'):
            rows.append(dict(model=path.parents[1].name,variant=path.parent.name,strategy=strategy,
                policy_negative_decision_cap=settings['alarm_cap'],
                selected_alarm_probability_threshold=selected['threshold'],
                policy_maximum_negative_positive_fraction=selected['max_negative_positive_fraction'],
                policy_cap_pass=selected['cap_pass'],
                policy_preconformal_90percent_detection_target_feasible=selected['target_feasible'],
                policy_preconformal_minimum_detection=selected['minimum_detection'],
                final_evaluation_person_macro_detection=g.detection_rate.mean(),
                final_evaluation_person_macro_FAR_per_negative_hour=g.FAR_per_non_target_hour.mean(),
                final_evaluation_person_macro_report_fraction=g.report_fraction.mean(),
                final_evaluation_person_macro_miss_penalized_delay=g.miss_penalized_delay.mean(),
                evaluation_people=g.subject.nunique(),event_contributing_people=int((g.events>0).sum()),
                horizon_seconds=settings['horizon_seconds'],minimum_event_duration_seconds=settings['minimum_event_duration_seconds'],
                exploratory=True,clinical_validity_established=False,
                policy_preconformal_target_not_final_test_detection_guarantee=True,
                policy_decision_cap_not_test_FAR_bound=True))
    write_csv(target,rows)

if __name__=='__main__':main()
