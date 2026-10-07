"""Regenerate daily weather comparison and fixed-date forecast plot from saved results."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[2]/'PVOD_Forecast/results/station05_sequence_v2'

def main():
    d=pd.read_csv(OUT/'daily_metrics.csv');rows=[]
    for (h,f),g in d.groupby(['horizon_minutes','fold']):
        pivot=g[g.model.isin(['HGB_local','HGB_local_surface','HGB_local_surface_pressure'])].pivot(index='day_utc',columns='model',values='RMSE_MW')
        rows.append({'horizon_minutes':h,'fold':f,'days':len(pivot),'surface_better_days':int((pivot.HGB_local_surface<pivot.HGB_local).sum()),'pressure_better_days':int((pivot.HGB_local_surface_pressure<pivot.HGB_local_surface).sum())})
    pd.DataFrame(rows).to_csv(OUT/'weather_daily_comparison.csv',index=False)
    c=pd.read_csv(OUT/'checkpoints.csv');chosen=[]
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    for ax,h in zip(axes,[15,60]):
        p=pd.read_csv(OUT/f'roll_3_{h}min_predictions.csv');p.target_utc=pd.to_datetime(p.target_utc,utc=True)
        p=p[(p.target_utc>=pd.Timestamp('2019-06-02',tz='UTC'))&(p.target_utc<pd.Timestamp('2019-06-05',tz='UTC'))]
        ax.plot(p.target_utc,p.actual_MW,label='Observed',color='#222222',linewidth=1.8)
        ax.plot(p.target_utc,p.HGB_local_surface_seed42,label='HGB + surface',color='#3575a8',linestyle='--',linewidth=1.3)
        for kind,color,style in [('LSTM','#bf8c44',':'),('GRU','#743f85','-.')]:
            match=c[c.run.str.startswith(f'roll_3_{h}min_{kind}_')].sort_values('validation_RMSE_MW').iloc[0]
            seed=int(match.run.rsplit('seed',1)[1])
            ax.plot(p.target_utc,p[f'{kind}_seed{seed}'],label=f'{kind}, validation-selected seed {seed}',color=color,linestyle=style,linewidth=1.3)
            chosen.append({'horizon_minutes':h,'kind':kind,'seed':seed,'validation_RMSE_MW':match.validation_RMSE_MW})
        ax.set_title(f'{h}-minute forecasts | fixed June 2-4, 2019 view (UTC)')
        ax.set_ylabel('Power (MW)');ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2)
    fig.savefig(OUT/'sequence_predictions.png',dpi=160);plt.close(fig)
    pd.DataFrame(chosen).to_csv(OUT/'plot_validation_selected_seeds.csv',index=False)
    print('Saved daily comparisons and forecast plot.')

if __name__=='__main__': main()
