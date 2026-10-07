"""Recalculate saved metrics and verify inputs/model persistence without retraining."""
from pathlib import Path
import hashlib,json,joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/thermal_v1'
cfg=json.loads((R/'config.json').read_text())
for name,h in cfg['input_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
p=pd.read_csv(R/'predictions.csv')
a=np.load(R/'evaluation_arrays.npz')
pack=joblib.load(ROOT/'models/thermal_v1/thermal_models.joblib')
horizons=np.arange(10,61,10)
a_rc,b=pack['rc_params']
eq=p.origin_ambient_c.to_numpy()+a_rc*(p.origin_current_a.to_numpy()/100)**2/b
rc=eq[:,None]+(p.origin_temperature_c.to_numpy()-eq)[:,None]*np.exp(-b*horizons)
y=p[[f'true_{h}s_c' for h in horizons]].to_numpy()
assert np.allclose(a['y'],y,atol=1e-10,rtol=0)
train=p.split.eq('train').to_numpy()
assert np.allclose(pack['xscaler'].mean_,a['X'][train].mean(axis=0),atol=1e-10,rtol=0)
assert np.allclose(pack['yscaler'].mean_,(y[train]-rc[train]).mean(axis=0),atol=1e-10,rtol=0)
reloaded=rc+pack['yscaler'].inverse_transform(pack['mlp'].predict(pack['xscaler'].transform(a['X'])))
saved=p[[f'RC_plus_MLP_{h}s_c' for h in horizons]].to_numpy()
assert np.allclose(reloaded,saved,atol=1e-10,rtol=0)
maxerr=0
for table,keys in [('metrics_overall',['split']),('metrics_by_power',['split','power_band']),('metrics_by_session',['split','session_id']),('metrics_by_phase',['split','historical_phase'])]:
 for row in pd.read_csv(R/(table+'.csv')).to_dict('records'):
  mask=np.ones(len(p),dtype=bool)
  for key in keys:mask&=p[key].eq(row[key]).to_numpy()
  g=p[mask];target=g[[f'true_{h}s_c' for h in horizons]].to_numpy();pred=g[[f"{row['model']}_{h}s_c" for h in horizons]].to_numpy()
  err=pred-target;end=err[:,-1]
  estimates={'windows':len(g),'sessions':g.session_id.nunique(),'mae_60_c':np.abs(end).mean(),'rmse_60_c':np.sqrt(np.mean(end**2)),
     'trajectory_mae_c':np.abs(err).mean(),'trajectory_rmse_c':np.sqrt(np.mean(err**2)),
     'p95_absolute_60_c':np.quantile(np.abs(end),.95),'window_peak_mae_c':np.abs(pred.max(axis=1)-target.max(axis=1)).mean()}
  session_values=pd.DataFrame({'session':g.session_id.to_numpy(),'ae':np.abs(end),'se':end**2}).groupby('session')[['ae','se']].mean()
  estimates['session_macro_mae_60_c']=session_values.ae.mean();estimates['session_macro_rmse_60_c']=np.sqrt(session_values.se).mean()
  for k,v in estimates.items():
   difference=abs(v-row[k]);maxerr=max(maxerr,difference);assert difference<1e-10,(table,k,difference)
result={'input_hashes_match':True,'all_four_metric_tables_recomputed':True,'metric_max_abs_difference':maxerr,
 'saved_model_all_window_prediction_max_abs_difference':float(np.max(np.abs(reloaded-saved))),
 'scaler_training_means_recomputed':True,'retraining_performed':False}
(R/'independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
