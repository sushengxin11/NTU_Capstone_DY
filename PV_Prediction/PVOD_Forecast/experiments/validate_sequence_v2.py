"""Independent audit of persisted predictions, checkpoints, metrics and split boundaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import torch
import run_sequence_v2 as exp

OUT=exp.OUT
m=pd.read_csv(OUT/'metrics.csv')
data,weather,_,_=exp.prepare()
checks=[]
for h in [15,60]:
    for fold,vs,es,ee in exp.FOLDS:
        p=pd.read_csv(OUT/f'{fold}_{h}min_predictions.csv')
        origin=pd.to_datetime(p.origin_utc,utc=True);target=pd.to_datetime(p.target_utc,utc=True)
        assert ((target-origin)==pd.Timedelta(minutes=h)).all()
        assert (origin>=pd.Timestamp(es,tz='UTC')).all()
        assert (target<pd.Timestamp(ee,tz='UTC')).all()
        start=pd.Timestamp('2019-05-31',tz='UTC');end=start+pd.Timedelta(days=1)
        assert not (((origin-pd.Timedelta(minutes=225))<end)&(target>=start)).any()
        split=pd.read_csv(OUT/f'{fold}_{h}min_split.csv')
        for col in ['origin_utc','target_utc']: split[col]=pd.to_datetime(split[col],utc=True)
        assert split.loc[split.split=='train','target_utc'].max()<split.loc[split.split=='validation','origin_utc'].min()
        assert split.loc[split.split=='validation','target_utc'].max()<split.loc[split.split=='evaluation','origin_utc'].min()
        models=[c for c in p if '_seed' in c]
        assert len(models)==10
        assert np.isfinite(p[models].to_numpy()).all()
        for col in models:
            model,seed=col.rsplit('_seed',1);seed=int(seed)
            for scope in ['all','daylight']:
                view=p if scope=='all' else p[p.target_irradiance_Wm2>20]
                pred=view[col].to_numpy();y=view.actual_MW.to_numpy()
                row=m[(m.horizon_minutes==h)&(m.fold==fold)&(m.model==model)&(m.seed==seed)&(m.scope==scope)].iloc[0]
                assert len(view)==row['n']
                assert abs(np.sqrt(np.mean((pred-y)**2))-row.RMSE_MW)<1e-9
                assert abs(np.mean(abs(pred-y))-row.MAE_MW)<1e-9
        # Independently reconstruct three historical windows from aligned data and reload all RNN seeds.
        sample_ids=[0,len(p)//2,len(p)-1]
        wx=[];cal=[];persist=[]
        for i in sample_ids:
            t=origin.iloc[i];tt=target.iloc[i]
            history=pd.date_range(t-pd.Timedelta(minutes=225),t,freq='15min')
            wx.append(weather.reindex(history).to_numpy())
            local=tt.tz_convert('Asia/Shanghai');hour=local.hour+local.minute/60
            cal.append([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*local.dayofyear/365.25),np.cos(2*np.pi*local.dayofyear/365.25)])
            persist.append(data.loc[t,'power'])
            assert abs(data.loc[tt,'power']-p.actual_MW.iloc[i])<1e-10
        wx=np.stack(wx);cal=np.asarray(cal)
        maxdiff=0
        for kind in ['LSTM','GRU']:
            for seed in exp.SEEDS:
                tag=f'{fold}_{h}min_{kind}_seed{seed}'
                checkpoint=torch.load(OUT/(tag+'.pt'),weights_only=True,map_location='cpu')
                scalers=joblib.load(OUT/(tag+'_scalers.joblib'))
                net=exp.SequenceRegressor(kind,checkpoint['input_size']);net.load_state_dict(checkpoint['state_dict'])
                xx=torch.from_numpy(scalers['sequence_scaler'].transform(wx.reshape(-1,wx.shape[-1])).reshape(wx.shape).astype('float32'))
                cc=torch.from_numpy(scalers['calendar_scaler'].transform(cal).astype('float32'))
                pred=np.maximum(0,np.asarray(persist)+scalers['residual_scaler'].inverse_transform(exp.predict(net,xx,cc).reshape(-1,1)).ravel())
                difference=float(np.max(abs(pred-p.loc[sample_ids,f'{kind}_seed{seed}'].to_numpy())))
                assert difference<1e-4,(tag,difference)
                maxdiff=max(maxdiff,difference)
        checks.append({'horizon_minutes':h,'fold':fold,'n':len(p),'models':len(models),'finite_predictions':True,'gap_crossing_windows':0,'target_metrics_recomputed':True,'label_availability_boundaries_valid':True,'checkpoint_reload_sample_n':3,'checkpoint_max_abs_difference_MW':maxdiff})
assert len(m)==120
log=(OUT/'run.log').read_text()
assert 'Traceback' not in log
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
print('Validated six fold/horizon groups, 120 metric rows, 36 neural checkpoints (3 points each).')
