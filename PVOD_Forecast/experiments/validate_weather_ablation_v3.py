"""Recompute saved metrics and independently reload masked-input GRU checkpoints."""
import json
import numpy as np
import pandas as pd
import torch
import joblib
import run_weather_ablation_v3 as exp

OUT=exp.OUT
config=json.loads((OUT/'config.json').read_text())
m=pd.read_csv(OUT/'metrics.csv')
data,weather,_,_=exp.seq.prepare()
checks=[]
for h in [15,60]:
    for fold,vs,es,ee in exp.seq.FOLDS:
        p=pd.read_csv(OUT/f'{fold}_{h}min_predictions.csv')
        origin=pd.to_datetime(p.origin_utc,utc=True);target=pd.to_datetime(p.target_utc,utc=True)
        assert ((target-origin)==pd.Timedelta(minutes=h)).all()
        assert (origin>=pd.Timestamp(es,tz='UTC')).all()
        start=pd.Timestamp('2019-05-31',tz='UTC');end=start+pd.Timedelta(days=1)
        assert not (((origin-pd.Timedelta(minutes=225))<end)&(target>=start)).any()
        splits=pd.read_csv(OUT/f'{fold}_{h}min_split.csv')
        for c in ['origin_utc','target_utc']: splits[c]=pd.to_datetime(splits[c],utc=True)
        assert splits.loc[splits.split=='train','target_utc'].max()<splits.loc[splits.split=='validation','origin_utc'].min()
        assert splits.loc[splits.split=='validation','target_utc'].max()<splits.loc[splits.split=='evaluation','origin_utc'].min()
        for col in [c for c in p if '_seed' in c]:
            assert np.isfinite(p[col]).all()
            model,seed=col.rsplit('_seed',1);seed=int(seed)
            for scope in ['all','daylight']:
                view=p if scope=='all' else p[p.target_irradiance_Wm2>20]
                row=m[(m.horizon_minutes==h)&(m.fold==fold)&(m.model==model)&(m.seed==seed)&(m.scope==scope)].iloc[0]
                error=view[col]-view.actual_MW
                assert row['n']==len(view)
                assert abs(np.sqrt(np.mean(error**2))-row.RMSE_MW)<1e-9
                assert abs(np.mean(abs(error))-row.MAE_MW)<1e-9
        ids=[0,len(p)//2,len(p)-1];wx=[];cal=[];persist=[]
        for i in ids:
            t=origin.iloc[i];tt=target.iloc[i]
            wx.append(weather.reindex(pd.date_range(t-pd.Timedelta(minutes=225),t,freq='15min')).to_numpy())
            local=tt.tz_convert('Asia/Shanghai');hour=local.hour+local.minute/60
            cal.append([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*local.dayofyear/365.25),np.cos(2*np.pi*local.dayofyear/365.25)])
            persist.append(data.loc[t,'power'])
            assert abs(data.loc[tt,'power']-p.actual_MW.iloc[i])<1e-10
        wx=np.stack(wx);cal=np.array(cal);maxdiff=0
        for group in exp.GROUPS:
            enabled=np.array([n in config['groups'][group] for n in weather])
            masked=wx.copy();masked[:,:,~enabled]=0
            for seed in exp.seq.SEEDS:
                tag=f'{fold}_{h}min_GRU_{group}_seed{seed}'
                ck=torch.load(OUT/(tag+'.pt'),weights_only=True,map_location='cpu')
                sc=joblib.load(OUT/(tag+'_scalers.joblib'))
                net=exp.seq.SequenceRegressor('GRU',ck['input_size']);net.load_state_dict(ck['state_dict'])
                assert sum(p.numel() for p in net.parameters())==config['parameter_count']
                assert np.all(sc['sequence_scaler'].mean_[~enabled]==0)
                assert np.all(sc['sequence_scaler'].scale_[~enabled]==1)
                xx=torch.from_numpy(sc['sequence_scaler'].transform(masked.reshape(-1,masked.shape[-1])).reshape(masked.shape).astype('float32'))
                assert torch.count_nonzero(xx[:,:,~enabled])==0
                cc=torch.from_numpy(sc['calendar_scaler'].transform(cal).astype('float32'))
                pred=np.maximum(0,np.array(persist)+sc['residual_scaler'].inverse_transform(exp.seq.predict(net,xx,cc).reshape(-1,1)).ravel())
                diff=float(np.max(abs(pred-p.loc[ids,f'GRU_{group}_seed{seed}'].to_numpy())))
                assert diff<1e-4;maxdiff=max(maxdiff,diff)
        previous=pd.read_csv(exp.ROOT/f'PVOD_Forecast/results/station05_sequence_v2/{fold}_{h}min_predictions.csv')
        assert p.target_utc.equals(previous.target_utc)
        reproduction=max(float(np.max(abs(p[f'GRU_local_surface_pressure_seed{s}']-previous[f'GRU_seed{s}']))) for s in exp.seq.SEEDS)
        assert reproduction<1e-6,reproduction
        checks.append({'horizon_minutes':h,'fold':fold,'n':len(p),'matched_samples':True,'metric_rows_recomputed':20,'gap_crossing_windows':0,'label_availability_checked':True,'fixed_parameter_count':config['parameter_count'],'checkpoint_reload_samples_per_model':3,'checkpoint_max_abs_difference_MW':maxdiff,'full_group_vs_v2_max_abs_difference_MW':reproduction})
assert len(m)==120
log=(OUT/'run.log').read_text()
assert 'Warning' not in log and 'Traceback' not in log
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
print('Passed: 120 metrics, 54 checkpoints, masking and exact full-group v2 reproduction.')
