"""Minimal measured battery-pack temperature experiment: persistence, effective RC, RC+MLP."""
from pathlib import Path
import copy, hashlib, json, platform
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/thermal_v1'
MODEL=ROOT/'models/thermal_v1'
HORIZONS=np.arange(10,61,10)
FEATURE_NAMES=['T0','T0_minus_ambient','T_change_20s','T_change_40s','T_change_60s',
 'I0_100A','I_mean_100A','I_std_100A','I_change_100A','P0_100kW','P_mean_100kW','P_std_100kW',
 'ambient0','ambient_change','soc0','soc_change','voltage0_400V']+[f'rc_delta_{h}s' for h in HORIZONS]

def rc_predict(t,ambient,current,params):
    a,b=params
    equilibrium=ambient+a*(current/100)**2/b
    return equilibrium[:,None]+(t-equilibrium)[:,None]*np.exp(-b*HORIZONS)

def history_features(history,params):
    h=history; now=h.iloc[-1]
    T=h.temperature_c.to_numpy(); I=h.battery_current_a.to_numpy()/100; P=h.dc_power_w.to_numpy()/1e5
    rc=rc_predict(np.array([T[-1]]),np.array([now.ambient_c]),np.array([now.battery_current_a]),params)[0]
    values=[T[-1],T[-1]-now.ambient_c,T[-1]-T[-21],T[-1]-T[-41],T[-1]-T[0],
            I[-1],I.mean(),I.std(),I[-1]-I[0],P[-1],P.mean(),P.std(),
            now.ambient_c,now.ambient_c-h.iloc[0].ambient_c,now.soc_percent,now.soc_percent-h.iloc[0].soc_percent,
            now.battery_voltage_v/400]+list(rc-T[-1])
    return np.array(values),rc

def load_windows():
    d=pd.read_csv(ROOT/'data/processed/bev_fast_clean.csv',parse_dates=['timestamp_utc'])
    splits=pd.read_csv(ROOT/'configs/provisional_split.csv')
    d=d.merge(splits[['session_id','split']],validate='many_to_one')
    rows=[];histories=[];targets=[]
    for sid,g in d.groupby('segment_id',sort=True):
        g=g.sort_values('timestamp_utc').reset_index(drop=True)
        assert np.all(np.diff(g.timestamp_utc.astype('int64').to_numpy())==10**9)
        for pos in range(60,len(g)-60,10):
            o=g.iloc[pos]
            rows.append({'session_id':o.session_id,'device_id':o.device_id,'segment_id':sid,'split':o.split,
                         'origin_utc':o.timestamp_utc,'target_utc':g.iloc[pos+60].timestamp_utc,
                         'origin_temperature_c':o.temperature_c,'origin_ambient_c':o.ambient_c,
                         'origin_current_a':o.battery_current_a,'origin_power_kw':o.dc_power_w/1000,
                         'history_delta_c':o.temperature_c-g.iloc[pos-60].temperature_c})
            histories.append(g.iloc[pos-60:pos+1].copy())
            targets.append(g.iloc[pos+HORIZONS].temperature_c.to_numpy())
    return pd.DataFrame(rows),histories,np.array(targets),d

def measure(meta,y,pred):
    err=pred-y; e=err[:,-1]
    per=[err[ix,-1] for ix in meta.groupby('session_id').indices.values()]
    return {'windows':len(y),'sessions':meta.session_id.nunique(), 'mae_60_c':np.abs(e).mean(),
            'rmse_60_c':np.sqrt(np.mean(e**2)), 'session_macro_mae_60_c':np.mean([np.abs(x).mean() for x in per]),
            'session_macro_rmse_60_c':np.mean([np.sqrt(np.mean(x**2)) for x in per]),
            'trajectory_mae_c':np.abs(err).mean(),'trajectory_rmse_c':np.sqrt(np.mean(err**2)),
            'p95_absolute_60_c':np.quantile(np.abs(e),.95),
            'window_peak_mae_c':np.mean(np.abs(pred.max(axis=1)-y.max(axis=1)))}

def predict_history(h,package):
    if len(h)!=61:raise ValueError('Exactly 61 consecutive 1-second history records required')
    times=pd.to_datetime(h.timestamp_utc,utc=True)
    if not np.all(np.diff(times.astype('int64'))==10**9):raise ValueError('History is not continuous at 1 second')
    x,rc=history_features(h,package['rc_params'])
    residual=package['yscaler'].inverse_transform(package['mlp'].predict(package['xscaler'].transform(x[None,:])))[0]
    return {'persistence':np.repeat(h.temperature_c.iloc[-1],6),'effective_RC':rc,'RC_plus_MLP':rc+residual}

def main():
    RESULT.mkdir(parents=True,exist_ok=True);MODEL.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/candidate_protocol.json').read_text())
    protocol.update(status='frozen_before_model_fit',seed=42,window_stride_s=10,output_horizons_s=HORIZONS.tolist(),
                    nn_candidates=[[16,16],[32,32]],nn_max_epochs=250,nn_patience=25,selection='validation session-macro RMSE at 60s',
                    rc_equation='dT/dt = a*(I_battery/100A)^2 - b*(T-Tambient)',rc_parameter_bounds=[[0,1e-8],[.01,.01]],
                    nn_feature_names=FEATURE_NAMES,measurement_latency_assumption='zero',raw_output_clipping=False)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/processed/bev_fast_clean.csv',ROOT/'configs/provisional_split.csv',Path(__file__)]}
    protocol['input_hashes']=hashes
    (RESULT/'config.json').write_text(json.dumps(protocol,indent=2))
    meta,histories,y,raw=load_windows();train=meta.split.eq('train').to_numpy();val=meta.split.eq('validation').to_numpy()
    assert set(meta.loc[train,'session_id']).isdisjoint(meta.loc[val,'session_id'])
    counts=meta.loc[train].session_id.value_counts();weights=meta.loc[train].session_id.map(lambda s:1/np.sqrt(counts[s])).to_numpy()
    def residual(params):
        p=rc_predict(meta.loc[train,'origin_temperature_c'].to_numpy(),meta.loc[train,'origin_ambient_c'].to_numpy(),meta.loc[train,'origin_current_a'].to_numpy(),params)
        return ((p-y[train])*weights[:,None]).ravel()
    fit=least_squares(residual,[.0005,.0001],bounds=([0,1e-8],[.01,.01]),max_nfev=500)
    params=fit.x
    if not fit.success:raise RuntimeError(fit.message)
    tau=1/params[1]
    (MODEL/'rc_parameters.json').write_text(json.dumps({'a_c_per_s_at_100a_squared':params[0],'b_per_s':params[1],'effective_tau_s':tau,'fit_train_sessions_only':True,'physical_R_C_not_identified':True,'optimizer_success':bool(fit.success)},indent=2))
    features=[];base=[]
    for h in histories:
        x,p=history_features(h,params);features.append(x);base.append(p)
    X=np.array(features);base=np.array(base)
    xs=StandardScaler().fit(X[train]);ys=StandardScaler().fit(y[train]-base[train])
    xt=xs.transform(X[train]);yt=ys.transform(y[train]-base[train]);xv=xs.transform(X[val])
    candidates=[];logs=[]
    for hidden in [(16,16),(32,32)]:
        nn=MLPRegressor(hidden_layer_sizes=hidden,activation='relu',solver='adam',alpha=.001,
                        batch_size=128,learning_rate_init=.001,random_state=42,shuffle=True)
        best=np.inf;best_nn=None;stale=0
        for epoch in range(1,251):
            nn.partial_fit(xt,yt)
            pv=base[val]+ys.inverse_transform(nn.predict(xv))
            score=measure(meta.loc[val],y[val],pv)['session_macro_rmse_60_c']
            logs.append({'architecture':str(hidden),'epoch':epoch,'train_loss_scaled':nn.loss_,'validation_macro_rmse_60_c':score})
            if score<best-1e-6:best=score;best_nn=copy.deepcopy(nn);best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=25:break
        candidates.append((best,hidden,best_epoch,best_nn))
        print('NN validation',hidden,'best epoch',best_epoch,'macro RMSE',best,flush=True)
    best,hidden,best_epoch,nn=min(candidates,key=lambda x:x[0])
    package={'rc_params':params,'mlp':nn,'xscaler':xs,'yscaler':ys,'feature_names':FEATURE_NAMES,'horizons_s':HORIZONS,'selected_architecture':hidden,'selected_epoch':best_epoch}
    joblib.dump(package,MODEL/'thermal_models.joblib')
    predictions={'persistence':np.repeat(meta.origin_temperature_c.to_numpy()[:,None],6,axis=1),
                 'effective_RC':base,'RC_plus_MLP':base+ys.inverse_transform(nn.predict(xs.transform(X)))}
    pd.DataFrame(logs).to_csv(RESULT/'training_history.csv',index=False)
    pd.DataFrame([{'architecture':str(h),'best_epoch':e,'validation_macro_rmse_60_c':s,'selected':h==hidden} for s,h,e,n in candidates]).to_csv(RESULT/'nn_selection.csv',index=False)
    bounds=protocol['power_bands_w'];meta['power_band']=pd.cut(meta.origin_power_kw*1000,[-np.inf,bounds['idle_upper'],bounds['low_upper'],bounds['mid_upper'],np.inf],labels=['idle_or_low_flow','low','mid','high'])
    meta['historical_phase']=np.where(meta.history_delta_c>.25,'warming',np.where(meta.history_delta_c<-.25,'cooling','stable'))
    tables={}
    for filename,columns in [('metrics_overall',['split']),('metrics_by_power',['split','power_band']),('metrics_by_session',['split','session_id']),('metrics_by_phase',['split','historical_phase'])]:
        results=[]
        for key,g in meta.groupby(columns,observed=True):
            key=key if isinstance(key,tuple) else (key,)
            ix=g.index.to_numpy()
            for name,p in predictions.items():results.append(dict(zip(columns,key),model=name,**measure(g,y[ix],p[ix])))
        tables[filename]=pd.DataFrame(results);tables[filename].to_csv(RESULT/(filename+'.csv'),index=False)
    pred=meta.copy()
    for k,h in enumerate(HORIZONS):
        pred[f'true_{h}s_c']=y[:,k]
        for name,p in predictions.items():pred[f'{name}_{h}s_c']=p[:,k]
    pred.to_csv(RESULT/'predictions.csv',index=False)
    np.savez_compressed(RESULT/'evaluation_arrays.npz',X=X,y=y)
    meta.to_csv(RESULT/'window_manifest.csv',index=False)
    # Reload and compare every prediction; check history-only entry point against saved test output.
    reloaded=joblib.load(MODEL/'thermal_models.joblib')
    re_pred=base+reloaded['yscaler'].inverse_transform(reloaded['mlp'].predict(reloaded['xscaler'].transform(X)))
    assert np.allclose(re_pred,predictions['RC_plus_MLP'],rtol=0,atol=1e-10)
    test_ix=np.flatnonzero(meta.split.eq('test'))
    demo_ix=test_ix[0];demo_h=histories[demo_ix]
    demo_h.to_csv(RESULT/'demo_history.csv',index=False)
    out=predict_history(demo_h,reloaded)
    for name in out:assert np.allclose(out[name],predictions[name][demo_ix],rtol=0,atol=1e-10)
    # Future-temperature changes leave features and forecasts unchanged, because inference reads history only.
    mutated=raw.copy();origin=meta.iloc[demo_ix]
    mask=(mutated.session_id==origin.session_id)&(mutated.timestamp_utc>origin.origin_utc)
    mutated.loc[mask,['temperature_c','dc_power_w','battery_current_a']]=999999
    h2=mutated[(mutated.segment_id==origin.segment_id)&(mutated.timestamp_utc>=demo_h.timestamp_utc.iloc[0])&(mutated.timestamp_utc<=origin.origin_utc)]
    out2=predict_history(h2,reloaded)
    assert all(np.array_equal(out[name],out2[name]) for name in out)
    demo=pd.DataFrame({'horizon_s':HORIZONS,**out});demo.to_csv(RESULT/'demo_predictions.csv',index=False)
    check_t=np.array([40.]);amb=np.array([20.]);zero=rc_predict(check_t,amb,np.array([0.]),params)[0]
    low=rc_predict(check_t,amb,np.array([100.]),params)[0];high=rc_predict(check_t,amb,np.array([200.]),params)[0]
    assert params[0]>=0 and params[1]>0 and np.all(np.diff(zero)<0) and np.all(high>=low)
    checks={'model_reload_max_abs_difference':float(np.max(np.abs(re_pred-predictions['RC_plus_MLP']))),
            'demo_matches_saved_prediction':True,'future_value_perturbation_invariance':True,
            'zero_heat_cools_toward_ambient':True,'higher_current_squared_not_lower_rc_temperature':True,
            'window_does_not_cross_missing_time':True,'train_validation_test_session_disjoint':meta.groupby('session_id').split.nunique().max()==1,
            'scalers_fit_train_only':int(xs.n_samples_seen_)==int(train.sum()),'no_output_clipping':True}
    checks={k:bool(v) if isinstance(v,(bool,np.bool_)) else v for k,v in checks.items()}
    (RESULT/'validation_checks.json').write_text(json.dumps(checks,indent=2))
    (RESULT/'run_summary.json').write_text(json.dumps({'python':platform.python_version(),'selected_architecture':hidden,'selected_epoch':best_epoch,'rc_a':params[0],'rc_b':params[1],'effective_tau_s':tau,'windows_by_split':meta.split.value_counts().to_dict(),'sessions_by_split':meta.groupby('split').session_id.nunique().to_dict(),'test_is_seen_devices_later_sessions':True,'runtime_scope':'public BMS max battery temperature; not charger module safety certification'},indent=2))
    make_figures(meta,y,predictions,tables,raw)
    print(tables['metrics_overall'].to_string(index=False),flush=True)

def make_figures(meta,y,preds,tables,raw):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'persistence':'#777777','effective_RC':'#2166ac','RC_plus_MLP':'#b35806'}
    test=tables['metrics_by_power'].query("split=='test'")
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    bands=[b for b in ['low','mid','high'] if b in set(test.power_band)]
    for j,name in enumerate(preds):
        g=test[test.model==name].set_index('power_band')
        ax.bar(np.arange(len(bands))+(j-1)*.24,[g.loc[b,'rmse_60_c'] for b in bands],width=.24,color=colors[name],label=name)
    ax.set_xticks(np.arange(len(bands)),bands);ax.set_ylim(bottom=0);ax.set_ylabel('60-second RMSE (°C)');ax.set_xlabel('Origin DC power band (training thresholds)');ax.legend(loc='upper left');ax.set_title('Test: temperature error by charging power')
    fig.savefig(RESULT/'test_power_comparison.png',dpi=180);plt.close(fig)
    sessions=sorted(meta.loc[meta.split.eq('test'),'session_id'].unique())
    # All test sessions shown; no visual best-case selection.
    fig,axes=plt.subplots(len(sessions),1,figsize=(10,2.5*len(sessions)),layout='constrained')
    for ax,sid in zip(axes,sessions):
        ix=meta.index[(meta.session_id==sid)&meta.split.eq('test')].to_numpy();m=meta.loc[ix]
        start=m.origin_utc.iloc[0];t=(m.target_utc-start).dt.total_seconds()/60
        ax.plot(t,y[ix,-1],color='black',lw=1.5,label='Observed BMS max temperature')
        for name,p in preds.items():ax.plot(t,p[ix,-1],lw=1,label=name,color=colors[name],linestyle='--' if name=='persistence' else '-')
        ax.set_title(sid,fontsize=10);ax.set_ylabel('Temperature (°C)');ax.set_xlabel('Minutes from first forecast origin')
    axes[0].legend(fontsize=8,ncol=2);fig.suptitle('All test sessions: rolling 60-second forecasts (each starts from observed origin T)',fontsize=12)
    fig.savefig(RESULT/'all_test_temperature_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for name,p in preds.items():
        ix=meta.split.eq('test').to_numpy();rmse=np.sqrt(np.mean((p[ix]-y[ix])**2,axis=0))
        ax.plot(HORIZONS,rmse,marker='o',color=colors[name],label=name)
    ax.set_xlabel('Forecast horizon (seconds)');ax.set_ylabel('Test RMSE (°C)');ax.set_ylim(bottom=0);ax.set_title('Test error across the 60-second prediction window');ax.legend()
    fig.savefig(RESULT/'test_horizon_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
