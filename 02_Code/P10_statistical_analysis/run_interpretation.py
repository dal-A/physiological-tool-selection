"""Explicit reporting guard and person-error uncertainty, no claim selection."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_statistics import plot_style,save_figure,SHORT,FAMILIES,STRATEGIES
from pmtools.experiment_io import write_csv,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    if (out/'claim_limits.json').exists():raise FileExistsError('Never silently overwrite completed analysis')
    e=pd.read_csv(out/'tables/endpoint_cluster_CI.csv');flags=[]
    for r in e[(e.task!='continuous_monitoring')&(e.metric=='report_fraction')].itertuples():
        flags.append(dict(task=r.task,model=r.model,source=r.source,strategy=r.strategy,budget_hundredths=r.budget_hundredths,
            independent_people=r.independent_people,report_fraction=r.estimate,minimum_report_fraction=.2,
            minimum_report_guard_pass=bool(r.estimate>=.2),
            empirical_set_coverage=float(e[(e.model==r.model)&(e.source==r.source)&(e.strategy==r.strategy)&(e.budget_hundredths==r.budget_hundredths)&(e.metric=='set_coverage')].estimate.iloc[0]),
            clinical_validity_established=False,secondary_HR_reference_is_offline_oracle=r.task=='hr_reliability',
            failure_reason=None if r.estimate>=.2 else 'report_fraction_below_frozen_20_percent_guard'))
    write_csv(out/'tables/reporting_guard_flags.csv',flags)
    failed=[r for r in flags if not r['minimum_report_guard_pass']]
    write_json(out/'claim_limits.json',dict(frozen_main_design_unchanged=True,report_guard_failures=failed,
        lower_loss_does_not_override_reporting_guard=True,
        policy_negative_decision_cap_not_evaluation_FAR_guarantee=True,
        empirical_coverage_not_conditional_or_patient_safety_guarantee=True,
        source_independent_HR_validation_counts={'PPGSPORT':29,'GALAXYPPG':23},
        pooled_sources_not_one_30_person_group=True,AF_AFL_not_pure_AF=True,
        HR_reference_correction_offline_only=True,clinical_validity_established=False))
    plot_style();bounds=pd.read_csv(out/'tables/person_error_event_bounds.csv')
    fig,axes=plt.subplots(2,4,figsize=(13.5,6))
    for i,model in enumerate(FAMILIES):
        for j,source in enumerate(('VITALDB','Stress-Predict','PPGSPORT','GALAXYPPG')):
            ax=axes[i,j];g=bounds[(bounds.model==model)&(bounds.source==source)&(bounds.endpoint=='at_least_one_wrong_accepted_window')].set_index('strategy').reindex(STRATEGIES)
            error=np.maximum(0,np.array([g.proportion-g.Wilson95_low,g.Wilson95_high-g.proportion]))
            ax.errorbar(range(4),g.proportion,yerr=error,fmt='o',capsize=3,color='#0072B2')
            ax.set_xticks(range(4),['Attempt','Gate','Fixed','Adaptive'],rotation=30);ax.set_ylim(-.02,1.04)
            ax.set_title(SHORT[source]+' / '+model)
            if j==0:ax.set_ylabel('People with >=1 accepted error')
    index=pd.read_csv(out/'figure_captions.csv').to_dict('records')
    save_figure(fig,out,'fig20_person_error_bounds','person_error_event_bounds.csv',
        'Proportion of independent evaluation people with at least one accepted wrong window at the main budget, with descriptive Wilson 95% intervals. This is not per-window selective risk; zero observed errors (including offline reference correction or all-abstain cases) are not proof of zero population risk. Exact one-sided upper bounds are in the table.',index)
    write_csv(out/'figure_captions.csv',index)


if __name__=='__main__':main()
