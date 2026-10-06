"""Controlled GRU weather ablation with identical architecture, initialization and samples."""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import torch
import run_sequence_v2 as seq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=seq.ROOT
OUT=ROOT/'PVOD_Forecast/results/station05_weather_ablation_v3'
GROUPS=['local','local_surface','local_surface_pressure']

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,weather,source,old=seq.prepare()
    prior=ROOT/'PVOD_Forecast/results/station05_sequence_v2'
    v2manifest=json.loads((prior/'manifest.json').read_text())
    assert seq.sha(old/'station05_aligned_15min.csv')==v2manifest['aligned_sha256']
    names=list(weather)
    active={
        'local':[n for n in names if not n.startswith('era_')],
        'local_surface':[n for n in names if not n.startswith('era_') or n.count('_')==1],
        'local_surface_pressure':names,
    }
    masks_features={g:np.array([n in active[g] for n in names]) for g in GROUPS}
    seq.OUT=OUT
    seq.FEATURE_NAMES=names
    seq.CALENDAR_NAMES=['target_hour_sin','target_hour_cos','target_doy_sin','target_doy_cos']
    dump('manifest.json',{'sources':source['sources'],'aligned_file':str((old/'station05_aligned_15min.csv').relative_to(ROOT)),'aligned_sha256':seq.sha(old/'station05_aligned_15min.csv'),'script_sha256':seq.sha(Path(__file__)),'sequence_code_sha256':seq.sha(Path(seq.__file__)),'base_code_sha256':seq.sha(Path(seq.base.__file__)),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__})
    config=json.loads((prior/'config.json').read_text())
    config.update({'architecture':'GRU only; fixed 28 input slots, hidden32 + Dense32/ReLU/Dense1, residual over persistence','groups':active,'masking':'Disabled weather input slots zeroed BEFORE scaler fit; same 28-dimensional network and parameter count for every group','parameter_count':sum(p.numel() for p in seq.SequenceRegressor('GRU',len(names)).parameters()),'initialization':'same torch seed, same input dimension and initialization per paired group; disabled slot weights receive no data gradient','comparison_caveat':'identical model and sequence representation; only enabled weather inputs differ','full_group_reproducibility_reference':'v2 GRU on same folds/seeds/inputs'})
    dump('config.json',config)
    metrics=[];histories=[];splits=[];checkpoints=[];daily=[]
    for horizon in [1,4]:
        tab,y,target,_,windows,calendar=seq.dataset(data,weather,horizon)
        persistence=tab.power_lag_0.to_numpy();origin=tab.index
        for fold,vstart,estart,eend in seq.FOLDS:
            vs,es,ee=[pd.Timestamp(v,tz='UTC') for v in [vstart,estart,eend]]
            masks={'train':target<vs,'validation':(origin>=vs)&(target<es),'evaluation':(origin>=es)&(target<ee)}
            assert target[masks['train']].max()<origin[masks['validation']].min()
            assert target[masks['validation']].max()<origin[masks['evaluation']].min()
            for split,mask in masks.items():
                splits.append({'horizon_minutes':horizon*15,'fold':fold,'split':split,'n':int(mask.sum()),'first_origin':str(origin[mask].min()),'last_target':str(target[mask].max())})
            labels=np.select([masks['train'],masks['validation'],masks['evaluation']],['train','validation','evaluation'],default='unused')
            pd.DataFrame({'origin_utc':origin,'target_utc':target,'split':labels}).to_csv(OUT/f'{fold}_{horizon*15}min_split.csv',index=False)
            ev=masks['evaluation'];actual=y[ev];irrad=data.lmd_totalirrad.reindex(target[ev]).to_numpy()
            frame=pd.DataFrame({'origin_utc':origin[ev],'target_utc':target[ev],'actual_MW':actual,'target_irradiance_Wm2':irrad})
            def evaluate(model,seed,p):
                assert np.isfinite(p).all()
                frame[f'{model}_seed{seed}']=p
                for scope,mask in [('all',np.ones(len(p),dtype=bool)),('daylight',irrad>20)]:
                    metrics.append({'horizon_minutes':horizon*15,'fold':fold,'model':model,'seed':seed,'scope':scope,**seq.score(actual[mask],p[mask])})
                d=pd.DataFrame({'day':target[ev].strftime('%Y-%m-%d'),'error':p-actual})
                for day,part in d.groupby('day'):
                    daily.append({'horizon_minutes':horizon*15,'fold':fold,'model':model,'seed':seed,'day_utc':day,'n':len(part),'MAE_MW':float(abs(part.error).mean()),'RMSE_MW':float(np.sqrt(np.mean(part.error**2)))})
            evaluate('persistence',0,persistence[ev])
            for group in GROUPS:
                # Fixed dimensional slots hold exact zero for variables excluded from this treatment.
                masked=windows.copy();masked[:,:,~masks_features[group]]=0.0
                assert np.all(masked[:,:,~masks_features[group]]==0)
                for seed in seq.SEEDS:
                    tag=f'{fold}_{horizon*15}min_GRU_{group}_seed{seed}'
                    p,history,epoch,val=seq.train('GRU',seed,masked,calendar,y,persistence,masks,tag)
                    histories+=history
                    checkpoints.append({'run':tag,'horizon_minutes':horizon*15,'fold':fold,'group':group,'seed':seed,'best_epoch':epoch,'validation_RMSE_MW':val})
                    evaluate('GRU_'+group,seed,p)
                    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
                    pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
            frame.to_csv(OUT/f'{fold}_{horizon*15}min_predictions.csv',index=False)
            pd.DataFrame(splits).to_csv(OUT/'split_summary.csv',index=False)
            pd.DataFrame(checkpoints).to_csv(OUT/'checkpoints.csv',index=False)
            pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    m=pd.DataFrame(metrics)
    summary=m.groupby(['horizon_minutes','fold','model','scope']).agg(RMSE_mean_MW=('RMSE_MW','mean'),RMSE_seed_std_MW=('RMSE_MW','std'),MAE_mean_MW=('MAE_MW','mean'),seed_runs=('seed','size'),n=('n','first')).reset_index()
    summary.to_csv(OUT/'seed_summary.csv',index=False)
    paired=[]
    for (h,fold,seed,scope),part in m[m.model!='persistence'].groupby(['horizon_minutes','fold','seed','scope']):
        p=part.set_index('model')
        for label,a,b in [('add_surface','GRU_local','GRU_local_surface'),('add_pressure','GRU_local_surface','GRU_local_surface_pressure')]:
            before=p.loc[a,'RMSE_MW'];after=p.loc[b,'RMSE_MW']
            paired.append({'horizon_minutes':h,'fold':fold,'seed':seed,'scope':scope,'comparison':label,'before_RMSE_MW':before,'after_RMSE_MW':after,'delta_RMSE_MW':after-before,'improvement_pct':100*(1-after/before)})
    pd.DataFrame(paired).to_csv(OUT/'paired_deltas.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    colors=['#444444','#9faeb7','#3575a8','#743f85']
    models=['persistence']+['GRU_'+g for g in GROUPS]
    for ax,h in zip(axes,[15,60]):
        sub=summary.query('horizon_minutes==@h and scope=="all"')
        for k,model in enumerate(models):
            part=sub[sub.model==model].set_index('fold').reindex([f[0] for f in seq.FOLDS])
            ax.bar(np.arange(3)+(k-1.5)*.19,part.RMSE_mean_MW,.19,color=colors[k],label=model,yerr=part.RMSE_seed_std_MW.fillna(0),capsize=3)
        ax.set_xticks(range(3),['Apr25-May4','May11-May20','May30-Jun13'])
        ax.set_ylabel('RMSE (MW)');ax.set_title(f'{h}-minute GRU weather ablation');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=4,title='Same architecture; bars = mean +/- seed sample std (not confidence intervals)')
    fig.savefig(OUT/'weather_ablation_rmse.png',dpi=160);plt.close(fig)
    print(summary.query('scope=="all"').to_string(index=False),flush=True)
    print('Completed: '+str(OUT),flush=True)

if __name__=='__main__': main()
