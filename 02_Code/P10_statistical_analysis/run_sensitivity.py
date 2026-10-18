"""P6 prespecified sensitivities; new version, never select a winning variant."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sensitivity_classification import run_classification_sensitivity
from sensitivity_monitoring import run_monitor_sensitivity
from run_statistics import check_inputs,plot_style,save_figure,LABELS,COLORS,SHORT,read_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--p9-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();scratch=a.scratch.resolve()
    if not (out/'main_analysis_status.json').is_file():raise ValueError('Main statistical stage must exist first')
    if (out/'sensitivity_runs').exists():raise FileExistsError('Never overwrite sensitivity outputs')
    scratch.mkdir(parents=True,exist_ok=True);(out/'sensitivity_inputs').mkdir();(out/'sensitivity_runs').mkdir()
    check_inputs(a.p9_root)
    from pmtools.experiment_io import write_json,write_csv
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run_classification_sensitivity(a.p9_root,a.data_root,out/'sensitivity_runs',scratch)
        run_monitor_sensitivity(a.p9_root,a.data_root,out/'sensitivity_runs',out/'sensitivity_inputs')
    plot_style();index=pd.read_csv(out/'figure_captions.csv').to_dict('records')
    classification=pd.read_csv(out/'tables/classification_sensitivity_CI.csv')
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for i,model in enumerate(('classical','cnn')):
        for j,source in enumerate(('VITALDB','Stress-Predict','PPGSPORT','GALAXYPPG')):
            ax=axes[i,j];strategy='quality' if source in ('PPGSPORT','GALAXYPPG') else 'adaptive'
            g=classification[(classification.model==model)&(classification.source==source)&
                             (classification.strategy==strategy)&(classification.metric=='mean_decision_loss')]
            ax.plot(range(len(g)),g.estimate,'o',color='#0072B2',ms=3)
            for k,(_,r) in enumerate(g.iterrows()):ax.plot([k,k],[r.CI95_low,r.CI95_high],color='#0072B2')
            names=[str(x).replace('quality_quantile','q').replace('planner_lambda','lambda').replace('HR_error_threshold_bpm','HR').replace('dalia_reference','reference') for x in g.variant]
            ax.set_xticks(range(len(g)),names,rotation=65,ha='right');ax.set_ylabel('Decision loss');ax.set_title(SHORT[source]+' / '+model)
    save_figure(fig,out,'fig15_classification_sensitivity','classification_sensitivity_CI.csv',
        'Prespecified one-factor sensitivity. Main settings remain fixed; each changed target is newly trained on development people and recalibrated on its own policy/conformal pools. No best-setting selection or primary-family re-testing.',index)
    mon=pd.read_csv(out/'tables/monitor_sensitivity_CI.csv');fig,axes=plt.subplots(2,3,figsize=(14,7))
    for i,model in enumerate(('classical','cnn')):
        for j,metric in enumerate(('miss_penalized_delay','detection_rate','FAR_per_non_target_hour')):
            ax=axes[i,j]
            for strategy in ('fixed','adaptive'):
                g=mon[(mon.model==model)&(mon.strategy==strategy)&(mon.metric==metric)]
                ax.plot(range(len(g)),g.estimate,'o-',color=COLORS[strategy],label=LABELS[strategy],ms=3)
            names=[str(x).replace('quality_quantile','q').replace('monitor_policy_cap','cap').replace('monitor_horizon_seconds','horizon').replace('monitor_event_duration_seconds','event') for x in g.variant]
            ax.set_xticks(range(len(g)),names,rotation=65,ha='right');ax.set_ylabel(metric.replace('_',' '));ax.set_title(model)
    axes[0,0].legend()
    save_figure(fig,out,'fig16_monitor_sensitivity','monitor_sensitivity_CI.csv',
        'Monitoring sensitivities keep the causal acquisition schedule and observed compute delays fixed. Changed caps refit only policy thresholds. Horizons change miss penalties; event-duration variants report their actual onset contributors separately. Do not compare seconds under different penalties as a selected improvement.',index)
    write_csv(out/'figure_captions.csv',index);check_inputs(a.p9_root)
    write_json(out/'sensitivity_stage_status.json',dict(prespecified_sensitivities_generated=True,
        completed_UTC=datetime.now(timezone.utc).isoformat(),P10_complete=False,
        final_independent_numerical_and_visual_acceptance_pending=True,P11_started=False))
    print('P10_SENSITIVITY_GENERATED_FINAL_ACCEPTANCE_PENDING',flush=True)

if __name__=='__main__':main()
