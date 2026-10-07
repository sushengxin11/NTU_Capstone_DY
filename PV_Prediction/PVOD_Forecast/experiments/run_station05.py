"""Leakage-aware retrospective station05 forecast benchmark; run from any cwd."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import xarray as xr
import sklearn
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'PVOD_Forecast/results/station05_v1'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
LAT, LON = 38.2355, 114.1236
CAPACITY_MW = 35.0  # metadata Capacity=35000, interpreted as kW; see source note.
EXCLUDED = '2019-05-31'
STEP = pd.Timedelta(minutes=15)

def log(s):
    print(s, flush=True)

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def source_manifest():
    files = [ROOT/'data/PVODdatasets_v1.0/station05.csv', ROOT/'data/PVODdatasets_v1.0/metadata.csv'] + sorted((ROOT/'data/ERA5_data').rglob('*.nc'))
    return [{'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]

def weather():
    frames=[]
    for group in ['single_levels_instant', 'single_levels_radiation', 'pressure_levels']:
        pieces=[]
        for path in sorted((ROOT/'data/ERA5_data'/group).glob('*.nc')):
            with xr.open_dataset(path, engine='h5netcdf') as ds:
                point = ds.interp(latitude=LAT, longitude=LON).load()
                values={}
                for name, v in point.data_vars.items():
                    if 'pressure_level' in v.dims:
                        for level in point.pressure_level.values:
                            values[f'era_{name}_{int(level)}'] = v.sel(pressure_level=level).values
                    else:
                        values[f'era_{name}']=v.values
                pieces.append(pd.DataFrame(values, index=pd.to_datetime(point.valid_time.values, utc=True)))
        f=pd.concat(pieces).sort_index()
        assert not f.index.has_duplicates
        frames.append(f)
    result=pd.concat(frames,axis=1).sort_index()
    for col in result:
        if col in ['era_t2m','era_d2m'] or col.startswith('era_t_'): result[col]-=273.15
        if col == 'era_sp': result[col]/=100.0
        if col in ['era_ssrd','era_fdir']: result[col]/=3600.0
    result.index.name='timestamp_utc'
    result.to_csv(OUT/'era5_station05_hourly.csv')
    return result

def build():
    raw=pd.read_csv(ROOT/'data/PVODdatasets_v1.0/station05.csv')
    raw.index=pd.to_datetime(raw.pop('date_time'), utc=True)
    raw=raw.sort_index()
    assert not raw.index.has_duplicates
    full=pd.date_range(raw.index.min(),raw.index.max(),freq='15min')
    pv=raw.reindex(full)
    assert pv.notna().all().all(), 'Unexpected PV missing values; require explicit handling.'
    # No removal followed by shift: preserve complete regular grid with excluded rows as NaN.
    excluded = pv.index.strftime('%Y-%m-%d') == EXCLUDED
    pv.loc[excluded,:]=np.nan
    era=weather()
    # Latest hourly valid time <= origin. No linear time interpolation or backfill.
    aligned=era.reindex(pv.index,method='ffill',tolerance=pd.Timedelta(minutes=59))
    aligned.loc[excluded,:]=np.nan
    merged=pv.join(aligned)
    merged.index.name='timestamp_utc'
    merged.to_csv(OUT/'station05_aligned_15min.csv')
    return merged, int(excluded.sum())

def features(data,horizon):
    idx=data.index
    origin=pd.Series(idx,index=idx)
    target_time=idx+horizon*STEP
    target=data.power.shift(-horizon)
    x=pd.DataFrame(index=idx)
    # 16 history observations: origin and preceding 15 quarter-hours.
    for lag in range(16): x[f'power_lag_{lag}']=data.power.shift(lag)
    x['power_mean_4h']=data.power.rolling(16,min_periods=16).mean()
    x['power_std_4h']=data.power.rolling(16,min_periods=16).std()
    local=target_time.tz_convert('Asia/Shanghai')
    hour=local.hour+local.minute/60
    x['target_hour_sin']=np.sin(2*np.pi*hour/24)
    x['target_hour_cos']=np.cos(2*np.pi*hour/24)
    x['target_doy_sin']=np.sin(2*np.pi*local.dayofyear/365.25)
    x['target_doy_cos']=np.cos(2*np.pi*local.dayofyear/365.25)
    history=list(x)
    lmd=[c for c in data if c.startswith('lmd_')]
    for c in lmd:
        if c=='lmd_winddirection':
            x[c+'_sin']=np.sin(np.deg2rad(data[c])); x[c+'_cos']=np.cos(np.deg2rad(data[c]))
        else: x[c]=data[c]
    localcols=list(x)
    surface=[c for c in data if c.startswith('era_') and c.count('_')==1]
    pressure=[c for c in data if c.startswith('era_') and c.count('_')==2]
    for c in surface+pressure: x[c]=data[c]
    # Require every source row through target to be present; no sequence spans excluded day.
    continuous=data.power.notna().astype(int).rolling(16+horizon,min_periods=16+horizon).sum().shift(-horizon)==16+horizon
    valid=continuous & x.notna().all(axis=1) & target.notna()
    sets={'history':history,'local':localcols,'local_surface':localcols+surface,'local_surface_pressure':localcols+surface+pressure}
    return x.loc[valid],target.loc[valid],pd.DatetimeIndex(target_time[valid]),sets

def metrics(y,p):
    return {'n':len(y),'MAE_MW':float(mean_absolute_error(y,p)), 'RMSE_MW':float(np.sqrt(mean_squared_error(y,p))), 'R2':float(r2_score(y,p)), 'nRMSE_capacity_pct':float(100*np.sqrt(mean_squared_error(y,p))/CAPACITY_MW)}

def main():
    log('Preparing aligned datasets...')
    data, excluded=build()
    dump('manifest.json',{'sources':source_manifest(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'xarray':xr.__version__,'sklearn':sklearn.__version__,'seed':SEED})
    # Frozen calendar boundaries near 70/15/15 raw chronological split, not tuned to results.
    train_end=pd.Timestamp('2019-05-15 00:00:00',tz='UTC')
    val_end=pd.Timestamp('2019-05-30 00:00:00',tz='UTC')
    dump('config.json',{'station':'station05','lat':LAT,'lon':LON,'horizons_minutes':[15,60],'history_steps':16,'train_target_before':str(train_end),'validation_target_before':str(val_end),'test_target_from':str(val_end),'excluded_utc_dates':[EXCLUDED],'excluded_pv_rows':excluded,'power_unit':'MW','capacity_MW_assumption':CAPACITY_MW,'sp_unit':'hPa','temperature_unit':'degC','radiation_unit':'W/m2','weather_alignment':'bilinear space, previous valid hour held <=59min; no future temporal interpolation','input_availability':'retrospective zero-latency measurements and ERA5 valid-time assumption; ERA5 publication latency not simulated','nwp':'excluded: no issue time metadata','daytime_definition':'target LMD total irradiance >20 W/m2 (evaluation only)','model_selection':'validation RMSE; shared final test held out; single seed pilot'})
    rows=[]; splits=[]; predictions=[]; training=[]
    for horizon in [1,4]:
        x,y,t,sets=features(data,horizon)
        masks={'train':t<train_end,'validation':(t>=train_end)&(t<val_end),'test':t>=val_end}
        assert np.isfinite(x.to_numpy()).all()
        for split,mask in masks.items():
            splits.append({'horizon_minutes':horizon*15,'split':split,'n':int(mask.sum()),'first_target':str(t[mask].min()),'last_target':str(t[mask].max())})
        splitlabels=np.select([masks['train'],masks['validation']],['train','validation'],default='test')
        pd.DataFrame({'origin_utc':x.index,'target_utc':t,'split':splitlabels}).to_csv(OUT/f'split_{horizon*15}min.csv',index=False)
        # Seasonal baseline uses exact target-24h timestamp on regular grid.
        seasonal=data.power.reindex(t-pd.Timedelta(days=1)).to_numpy()
        predstore={}
        def evaluate(name,pred):
            assert np.isfinite(pred).all(), f'Nonfinite prediction: {name}'
            predstore[name]=pred
            target_irrad=data.lmd_totalirrad.reindex(t).to_numpy()
            for split in ['validation','test']:
                for scope in ['all','daylight']:
                    mask=masks[split].copy() & np.isfinite(pred)
                    if scope=='daylight': mask &= target_irrad>20
                    row={'horizon_minutes':horizon*15,'model':name,'split':split,'scope':scope,**metrics(y.to_numpy()[mask],pred[mask])}
                    rows.append(row)
        evaluate('persistence',x.power_lag_0.to_numpy())
        evaluate('previous_day_fallback',np.where(np.isfinite(seasonal),seasonal,x.power_lag_0.to_numpy()))
        # Tree ablations differ only by additional ERA5 inputs; common samples / settings.
        for subset,cols in sets.items():
            model=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=SEED)
            model.fit(x.loc[masks['train'],cols],y.loc[masks['train']])
            name='HGB_'+subset
            evaluate(name,np.maximum(0,model.predict(x[cols])))
            joblib.dump({'model':model,'features':cols},OUT/f'{horizon*15}min_{name}.joblib')
            log(f'{horizon*15}min {name} done')
        cols=sets['local_surface_pressure']
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=10))
        ridge.fit(x.loc[masks['train'],cols],y.loc[masks['train']])
        evaluate('Ridge_full',np.maximum(0,ridge.predict(x[cols])))
        joblib.dump({'model':ridge,'features':cols},OUT/f'{horizon*15}min_Ridge_full.joblib')
        # Explicit chronological validation, scalers fitted on train only.
        sx=StandardScaler().fit(x.loc[masks['train'],cols])
        sy=StandardScaler().fit(y.loc[masks['train']].to_numpy().reshape(-1,1))
        xx=sx.transform(x[cols]); yy=sy.transform(y.to_numpy().reshape(-1,1)).ravel()
        mlp=MLPRegressor(hidden_layer_sizes=(64,32),alpha=.01,batch_size=128,learning_rate_init=.001,max_iter=1,shuffle=False,random_state=SEED)
        best=float('inf'); stale=0; epochs=0
        for epoch in range(1,101):
            mlp.partial_fit(xx[masks['train']],yy[masks['train']])
            vp=np.maximum(0,sy.inverse_transform(mlp.predict(xx[masks['validation']]).reshape(-1,1)).ravel())
            score=float(np.sqrt(mean_squared_error(y.loc[masks['validation']],vp)))
            training.append({'horizon_minutes':horizon*15,'epoch':epoch,'train_scaled_loss':float(mlp.loss_),'validation_RMSE_MW':score})
            if score<best-1e-5:
                best=score; epochs=epoch; stale=0
                joblib.dump({'model':mlp,'x_scaler':sx,'y_scaler':sy,'features':cols,'epoch':epoch},OUT/f'{horizon*15}min_MLP_full.joblib')
            else: stale+=1
            if stale>=15: break
        saved=joblib.load(OUT/f'{horizon*15}min_MLP_full.joblib')
        evaluate('MLP_full',np.maximum(0,sy.inverse_transform(saved['model'].predict(xx).reshape(-1,1)).ravel()))
        log(f'{horizon*15}min MLP validation-selected epoch={epochs}, RMSE={best:.4f}MW')
        predframe=pd.DataFrame({'origin_utc':x.index,'target_utc':t,'split':splitlabels,'actual_MW':y.to_numpy(), 'target_irradiance_Wm2':data.lmd_totalirrad.reindex(t).to_numpy(),**predstore})
        predframe.loc[predframe.split!='train'].to_csv(OUT/f'predictions_{horizon*15}min.csv',index=False)
        predictions.append(predframe)
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(splits).to_csv(OUT/'split_summary.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'mlp_training.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,h in zip(axes,[15,60]):
        m=result.query('horizon_minutes==@h and split=="test" and scope=="all"')
        ax.barh(m.model,m.RMSE_MW,color='#3575a8');ax.set_xlabel('RMSE (MW)');ax.set_title(f'{h}-minute horizon | held-out test');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'test_rmse.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    choices={}
    for ax,h,p in zip(axes,[15,60],predictions):
        v=result.query('horizon_minutes==@h and split=="validation" and scope=="all"').sort_values('RMSE_MW')
        bestname=v.iloc[0].model;choices[str(h)]=bestname
        view=p[(p.split=='test') & (p.target_utc>=pd.Timestamp('2019-06-02',tz='UTC')) & (p.target_utc<pd.Timestamp('2019-06-05',tz='UTC'))]
        ax.plot(view.target_utc,view.actual_MW,label='Observed',color='black')
        ax.plot(view.target_utc,view[bestname],label=f'Validation-selected: {bestname}',color='#3575a8',linestyle='--')
        ax.set_title(f'{h}-minute forecast | June 2-4, 2019 (UTC)');ax.set_ylabel('Power (MW)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'test_predictions.png',dpi=160);plt.close(fig)
    dump('validation_selected_models.json',choices)
    log(result.query('split=="test" and scope=="all"').to_string(index=False))
    log('Completed: '+str(OUT))

if __name__=='__main__': main()
