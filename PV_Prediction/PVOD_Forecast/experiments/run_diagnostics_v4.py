"""ERA5/LMD agreement and train-threshold scene diagnostics; no model retraining."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_sequence_v2 as seq

ROOT=seq.ROOT
OUT=ROOT/'PVOD_Forecast/results/station05_diagnostics_v4'
V2=ROOT/'PVOD_Forecast/results/station05_sequence_v2'
V3=ROOT/'PVOD_Forecast/results/station05_weather_ablation_v3'

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def agreement(observed,era):
    d=pd.DataFrame({'observed':observed,'era':era}).dropna()
    diff=d.era-d.observed
    return {'n':len(d),'correlation':float(d.observed.corr(d.era)),'bias_ERA_minus_LMD':float(diff.mean()),'MAE':float(abs(diff).mean()),'RMSE':float(np.sqrt(np.mean(diff**2)))}

def weather_checks(data):
    hourly=pd.read_csv(ROOT/'PVOD_Forecast/results/station05_v1/era5_station05_hourly.csv',index_col=0)
    hourly.index=pd.to_datetime(hourly.index,utc=True)
    # Use native valid hour, not time-held 15-minute weather repeated four times.
    instant=data.reindex(hourly.index)
    hourly['era_windspeed']=np.hypot(hourly.era_u10,hourly.era_v10)
    pairs={
        'temperature_C':('lmd_temperature','era_t2m'),
        'pressure_hPa':('lmd_pressure','era_sp'),
        'windspeed_mps':('lmd_windspeed','era_windspeed'),
    }
    aligned=pd.DataFrame(index=hourly.index)
    for label,(lmd,era) in pairs.items():
        aligned[label+'_LMD']=instant[lmd];aligned[label+'_ERA']=hourly[era]
    # Approximate preceding-hour mean from 5 instantaneous samples (trapezoidal rule).
    radiation=data.lmd_totalirrad
    trap=(radiation.shift(4)/2+radiation.shift(3)+radiation.shift(2)+radiation.shift(1)+radiation/2)/4
    right=pd.concat([radiation.shift(i) for i in range(4)],axis=1).mean(axis=1,skipna=False)
    aligned['irradiance_Wm2_LMD']=trap.reindex(hourly.index)
    aligned['irradiance_Wm2_ERA']=hourly.era_ssrd
    aligned['irradiance_right4_Wm2_LMD']=right.reindex(hourly.index)
    aligned=aligned.loc[data.index.min():data.index.max()]
    aligned=aligned[aligned.index.strftime('%Y-%m-%d')!='2019-05-31']
    rows=[]
    for label in list(pairs)+['irradiance_Wm2']:
        rows.append({'variable':label,'scope':'all',**agreement(aligned[label+'_LMD'],aligned[label+'_ERA'])})
        if label=='irradiance_Wm2':
            day=aligned[aligned.irradiance_Wm2_LMD>20]
            rows.append({'variable':label,'scope':'daylight',**agreement(day[label+'_LMD'],day[label+'_ERA'])})
    rows.append({'variable':'irradiance_right4_Wm2','scope':'all',**agreement(aligned.irradiance_right4_Wm2_LMD,aligned.irradiance_Wm2_ERA)})
    aligned.index.name='valid_time_utc';aligned.to_csv(OUT/'weather_hourly_pairs.csv')
    pd.DataFrame(rows).to_csv(OUT/'weather_agreement.csv',index=False)
    cycle_parts=[]
    for label in list(pairs)+['irradiance_Wm2']:
        matched=aligned[[label+'_LMD',label+'_ERA']].dropna()
        cycle_parts.append(matched.groupby(matched.index.tz_convert('Asia/Shanghai').hour).mean())
    cycles=pd.concat(cycle_parts,axis=1);cycles.index.name='local_hour'
    cycles.to_csv(OUT/'weather_local_hour_cycle.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,label in zip(axes.ravel(),list(pairs)+['irradiance_Wm2']):
        # Matched finite pairs only for each series, so denominators agree.
        pair=aligned[[label+'_LMD',label+'_ERA']].dropna()
        hours=pair.index.tz_convert('Asia/Shanghai').hour
        means=pair.groupby(hours).mean()
        ax.plot(means.index,means[label+'_LMD'],color='#222222',label='LMD',marker='o',markersize=3)
        ax.plot(means.index,means[label+'_ERA'],color='#3575a8',label='ERA5',linestyle='--',marker='s',markersize=3)
        ax.set_title(label);ax.set_xlabel('Local hour (Asia/Shanghai)');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'weather_diurnal.png',dpi=160);plt.close(fig)
    return rows

def scene_checks(data):
    metrics=[];thresholds=[];registry=[]
    for h in [15,60]:
        for fold,vs,es,ee in seq.FOLDS:
            split=pd.read_csv(V3/f'{fold}_{h}min_split.csv')
            train=split[split.split=='train']
            targets=pd.to_datetime(train.target_utc,utc=True);origins=pd.to_datetime(train.origin_utc,utc=True)
            irrad=data.lmd_totalirrad.reindex(targets).to_numpy()
            power=data.power.reindex(targets).to_numpy();old=data.power.reindex(origins).to_numpy()
            day=irrad>20
            q1,q2=np.quantile(irrad[day],[1/3,2/3])
            ramp=float(np.quantile(abs(power[day]-old[day]),.9))
            thresholds.append({'horizon_minutes':h,'fold':fold,'training_n':len(train),'training_daylight_n':int(day.sum()),'irradiance_low_max_Wm2':float(q1),'irradiance_medium_max_Wm2':float(q2),'ramp_abs_change_MW_p90':ramp,'last_training_target_utc':str(targets.max())})
            p=pd.read_csv(V3/f'{fold}_{h}min_predictions.csv')
            q=pd.read_csv(V2/f'{fold}_{h}min_predictions.csv')
            assert p.target_utc.equals(q.target_utc) and np.allclose(p.actual_MW,q.actual_MW)
            for seed in seq.SEEDS:
                assert np.array_equal(p[f'GRU_local_surface_pressure_seed{seed}'],q[f'GRU_seed{seed}'])
            for c in q:
                if c.startswith('HGB_') or c.startswith('LSTM_'): p[c]=q[c]
            radiation=p.target_irradiance_Wm2.to_numpy();delta=(p.actual_MW-p.persistence_seed0).to_numpy()
            p['irradiance_scene']=np.select([radiation<=20,radiation<=q1,radiation<=q2],['night_lowlight','day_low','day_medium'],default='day_high')
            p['ramp_scene']=np.select([radiation<=20,delta>ramp,delta<-ramp],['night_lowlight','day_ramp_up','day_ramp_down'],default='day_stable')
            p['delta_power_MW']=delta
            columns=[c for c in p if '_seed' in c]
            for c in columns:
                name,seed=c.rsplit('_seed',1)
                for dimension in ['irradiance_scene','ramp_scene']:
                    for scene,part in p.groupby(dimension):
                        metrics.append({'horizon_minutes':h,'fold':fold,'model':name,'seed':int(seed),'dimension':dimension,'scene':scene,**seq.score(part.actual_MW.to_numpy(),part[c].to_numpy())})
            p.to_csv(OUT/f'{fold}_{h}min_scene_predictions.csv',index=False)
            registry.append({'horizon_minutes':h,'fold':fold,'n':len(p),'models':len(columns),'irradiance_counts':p.irradiance_scene.value_counts().to_dict(),'ramp_counts':p.ramp_scene.value_counts().to_dict()})
    m=pd.DataFrame(metrics);m.to_csv(OUT/'scene_metrics.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'training_scene_thresholds.csv',index=False)
    summary=m.groupby(['horizon_minutes','fold','model','dimension','scene']).agg(RMSE_mean_MW=('RMSE_MW','mean'),RMSE_seed_std_MW=('RMSE_MW','std'),MAE_mean_MW=('MAE_MW','mean'),n=('n','first'),seed_runs=('seed','size')).reset_index()
    summary.to_csv(OUT/'scene_seed_summary.csv',index=False)
    dump('scene_counts.json',registry)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    scenes=['day_stable','day_ramp_up','day_ramp_down'];models=['persistence','HGB_local_surface','GRU_local','GRU_local_surface','GRU_local_surface_pressure']
    colors=['#444444','#a1b8c5','#83b4c9','#3575a8','#743f85']
    for ax,h in zip(axes,[15,60]):
        group=summary[(summary.horizon_minutes==h)&(summary.fold=='roll_3')&(summary.dimension=='ramp_scene')]
        for k,model in enumerate(models):
            g=group[group.model==model].set_index('scene').reindex(scenes)
            ax.bar(np.arange(3)+(k-2)*.15,g.RMSE_mean_MW,.15,label=model,color=colors[k])
        ax.set_xticks(range(3),scenes);ax.set_title(f'{h}-minute | roll_3 daylight scenes');ax.set_ylabel('RMSE (MW)');ax.grid(axis='y',alpha=.2)
        counts=group[group.model=='persistence'].set_index('scene').n
        for j,s in enumerate(scenes): ax.text(j,ax.get_ylim()[1]*.98,f'n={counts[s]}',ha='center',va='top',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,title='Scenes use target values for diagnosis only; thresholds fitted on training')
    fig.savefig(OUT/'ramp_scene_rmse.png',dpi=160);plt.close(fig)
    return thresholds,registry

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,_,source,old=seq.prepare()
    assert seq.sha(old/'station05_aligned_15min.csv')==json.loads((V3/'manifest.json').read_text())['aligned_sha256']
    files=[old/'station05_aligned_15min.csv',old/'era5_station05_hourly.csv',V3/'manifest.json',V2/'manifest.json']
    files+=sorted(V2.glob('*_predictions.csv'))+sorted(V3.glob('*_predictions.csv'))+sorted(V3.glob('*_split.csv'))
    dump('manifest.json',{'original_sources':source['sources'],'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':seq.sha(p)} for p in files],'script_sha256':seq.sha(Path(__file__))})
    dump('method.json',{'weather':'ERA5 native UTC valid hour vs LMD measurement at same hour; wind = hypot(u10,v10)','radiation':'ERA5 preceding-hour mean vs trapezoidal integral of five LMD samples, assumes LMD labels are instantaneous; right-endpoint four-sample mean also reported','radiation_limit':'LMD interval label semantics not fully documented, approximate temporal comparability; no shifting to maximize correlation','weather_limits':'grid vs site, pressure/elevation and wind measurement height differences not corrected; no claim of sensor equivalence','scene_thresholds':'per fold and horizon; training daylight (>20W/m2) irradiance tertiles and absolute target-origin power change p90','scene_use':'future target measurements only for retrospective evaluation partition; never model inputs','evaluation':'previously inspected historical periods; no new independent holdout or model retraining'})
    weather=weather_checks(data);thresholds,counts=scene_checks(data)
    # Recompute saved scene metrics independently from persisted predictions.
    allmetrics=pd.read_csv(OUT/'scene_metrics.csv')
    for item in counts:
        h=item['horizon_minutes'];fold=item['fold'];p=pd.read_csv(OUT/f'{fold}_{h}min_scene_predictions.csv')
        assert sum(item['irradiance_counts'].values())==len(p)==sum(item['ramp_counts'].values())
        assert np.isfinite(p[[c for c in p if '_seed' in c]].to_numpy()).all()
        group=allmetrics[(allmetrics.horizon_minutes==h)&(allmetrics.fold==fold)]
        for r in group.itertuples(index=False):
            sub=p[p[r.dimension]==r.scene];error=sub[f'{r.model}_seed{r.seed}']-sub.actual_MW
            assert len(sub)==r.n
            assert abs(np.sqrt(np.mean(error**2))-r.RMSE_MW)<1e-9
            assert abs(np.mean(abs(error))-r.MAE_MW)<1e-9
    dump('validation_checks.json',{'groups':len(counts),'scene_metric_rows':len(allmetrics),'metrics_recomputed':True,'scene_partition_counts_reconciled':True,'full_GRU_v2_v3_predictions_equal':True,'all_predictions_finite':True,'thresholds_use_training_only':True})
    print(pd.DataFrame(weather).to_string(index=False),flush=True)
    print('Complete: '+str(OUT),flush=True)

if __name__=='__main__': main()
