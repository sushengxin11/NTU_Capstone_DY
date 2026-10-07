"""Frozen one-hour ramp-weighted benchmark, isolated from existing experiments."""
from pathlib import Path
import json, copy, time, platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import run_sequence_v2 as seq

ROOT=seq.ROOT
OUT=ROOT/'PVOD_Forecast/results/station05_weighted_loss_v5'
ALPHAS=[1,2,4]
GUARDRAIL=1.02
SCOPES=['all','daylight','day_stable','day_ramp_up','day_ramp_down','day_ramp','night_lowlight']

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def inputs():
    data,weather,manifest,old=seq.prepare()
    tab,y,target,sets,windows,calendar=seq.dataset(data,weather,4)
    names=list(weather)
    active=np.array([not n.startswith('era_') or n.count('_')==1 for n in names])
    windows[:,:,~active]=0
    return data,tab,y,target,sets,windows,calendar,names,active,manifest,old

def fold_data(data,tab,y,target,fold):
    name,vs,es,ee=fold
    vs,es,ee=[pd.Timestamp(v,tz='UTC') for v in [vs,es,ee]]
    origin=tab.index
    masks={'train':target<vs,'validation':(origin>=vs)&(target<es),'evaluation':(origin>=es)&(target<ee)}
    assert target[masks['train']].max()<origin[masks['validation']].min()
    assert target[masks['validation']].max()<origin[masks['evaluation']].min()
    daylight=data.lmd_totalirrad.reindex(target).to_numpy()>20
    delta=y-tab.power_lag_0.to_numpy()
    tau=float(np.quantile(abs(delta[masks['train']&daylight]),.9))
    scene=np.select([~daylight,delta>tau,delta<-tau],['night_lowlight','day_ramp_up','day_ramp_down'],default='day_stable')
    ramp=daylight&(abs(delta)>tau)
    return masks,scene,ramp,tau

def scope_mask(scene,scope):
    if scope=='all': return np.ones(len(scene),dtype=bool)
    if scope=='daylight': return scene!='night_lowlight'
    if scope=='day_ramp': return np.isin(scene,['day_ramp_up','day_ramp_down'])
    return scene==scope

def train(kind,seed,windows,calendar,y,persistence,masks,tag,weights,alpha):
    seq.seed_all(seed)
    # Fit feature scaler on unique historical timestamps used by training windows;
    # flattening weights overlapping observations but uses no validation/test information.
    sx=StandardScaler().fit(windows[masks['train']].reshape(-1,windows.shape[-1]))
    sc=StandardScaler().fit(calendar[masks['train']])
    residual=y-persistence
    sy=StandardScaler().fit(residual[masks['train']].reshape(-1,1))
    xx=torch.from_numpy(sx.transform(windows.reshape(-1,windows.shape[-1])).reshape(windows.shape).astype('float32'))
    cc=torch.from_numpy(sc.transform(calendar).astype('float32'))
    yy=torch.from_numpy(sy.transform(residual.reshape(-1,1)).ravel().astype('float32'))
    model=seq.SequenceRegressor(kind,windows.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    lossfn=nn.MSELoss()
    ww=torch.from_numpy(weights.astype('float32'))
    ids=np.flatnonzero(masks['train'])
    best=float('inf');stale=0;beststate=None;history=[];bestepoch=0
    start=time.monotonic()
    for epoch in range(1,seq.MAX_EPOCHS+1):
        model.train();total=0
        # Fixed chronological batches, independent overlapping windows; hidden state reset per batch.
        for i in range(0,len(ids),256):
            batch=ids[i:i+256]
            optimizer.zero_grad(set_to_none=True)
            output=model(xx[batch],cc[batch])
            loss=lossfn(output,yy[batch]) if alpha==1 else (ww[batch]*(output-yy[batch])**2).sum()/ww[batch].sum()
            assert torch.isfinite(loss), f'Nonfinite loss: {tag}'
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=loss.item()*len(batch)
        vp=persistence[masks['validation']]+sy.inverse_transform(seq.predict(model,xx[masks['validation']],cc[masks['validation']]).reshape(-1,1)).ravel()
        val=seq.score(y[masks['validation']],np.maximum(0,vp))['RMSE_MW']
        history.append({'run':tag,'epoch':epoch,'train_batch_loss_mean':total/len(ids),'validation_RMSE_MW':val})
        if val<best-1e-5:
            best=val;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch;stale=0
        else: stale+=1
        if stale>=seq.PATIENCE: break
    model.load_state_dict(beststate)
    torch.save({'kind':kind,'state_dict':beststate,'input_size':windows.shape[-1],'hidden_size':32,'best_epoch':bestepoch,'seed':seed},OUT/(tag+'.pt'))
    joblib.dump({'sequence_scaler':sx,'calendar_scaler':sc,'residual_scaler':sy,'feature_names':FEATURE_NAMES,'calendar_names':CALENDAR_NAMES},OUT/(tag+'_scalers.joblib'))
    ep=np.maximum(0,persistence+sy.inverse_transform(seq.predict(model,xx,cc).reshape(-1,1)).ravel())
    assert np.isfinite(ep).all()
    print(f'{tag}: epoch={bestepoch} val_RMSE={best:.4f} elapsed={time.monotonic()-start:.1f}s',flush=True)
    return ep,history,bestepoch,best


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,tab,y,target,sets,windows,calendar,names,active,source,old=inputs()
    global FEATURE_NAMES,CALENDAR_NAMES
    FEATURE_NAMES=names
    CALENDAR_NAMES=['target_hour_sin','target_hour_cos','target_doy_sin','target_doy_cos']
    dump('config.json',{'alphas':ALPHAS,'guardrail_ratio':GUARDRAIL,'guardrail_scopes':['all','day_stable'],'folds':seq.FOLDS,'seeds':seq.SEEDS,'horizon_minutes':60,'history_steps':16,'features':names,'active_features':[n for n,a in zip(names,active) if a],'tree_features':sets['local_surface'],'max_epochs':35,'patience':7,'GRU_parameter_count':7169,'loss':'sum(weight * squared standardized residual error) / sum(weight), per batch; alpha1 exact original MSE','ramp':'target irradiance >20 and abs(target power - origin power) > training daylight p90','checkpoint_selection':'unweighted validation RMSE, unchanged','alpha_selection':'validation seed mean ramp RMSE subject to all/stable <=1.02*alpha1; fallback alpha1','expert_selection':'validation ramp RMSE only; route stage subject to improvement over baseline HGB','excluded_utc_dates':['2019-05-31'],'limitations':'Previously inspected historical evaluation; ERA5 valid-time retrospective inputs without release latency'})
    dump('manifest.json',{'sources':source['sources'],'aligned_sha256':seq.sha(old/'station05_aligned_15min.csv'),'script_sha256':seq.sha(Path(__file__)),'sequence_script_sha256':seq.sha(Path(seq.__file__)),'base_script_sha256':seq.sha(Path(seq.base.__file__)),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__})
    prior_thresholds=pd.read_csv(ROOT/'PVOD_Forecast/results/station05_diagnostics_v4/training_scene_thresholds.csv')
    rows=[];history=[];thresholds=[];checks=[];daily=[]
    for fold in seq.FOLDS:
        name=fold[0]
        masks,scene,ramp,tau=fold_data(data,tab,y,target,fold)
        reference=prior_thresholds.query('fold==@name and horizon_minutes==60').iloc[0]
        assert abs(tau-reference.ramp_abs_change_MW_p90)<1e-10
        thresholds.append({'fold':name,'tau_MW':tau,'train_n':int(masks['train'].sum()),'train_ramp_n':int((masks['train']&ramp).sum())})
        frames={}
        for split in ['validation','evaluation']:
            mask=masks[split]
            frames[split]=pd.DataFrame({'origin_utc':tab.index[mask],'target_utc':target[mask],'actual_MW':y[mask],'origin_power_MW':tab.power_lag_0.to_numpy()[mask],'scene':scene[mask],'ramp':ramp[mask]})
        pd.DataFrame({'origin_utc':tab.index,'target_utc':target,'split':np.select(list(masks.values()),list(masks),default='unused')}).to_csv(OUT/(name+'_split.csv'),index=False)
        def evaluate(model,seed,alpha,pred):
            for split,frame in frames.items():
                mask=masks[split];p=pred[mask];frame[f'{model}_a{alpha}_seed{seed}']=p
                assert np.isfinite(p).all()
                for scope in SCOPES:
                    sm=scope_mask(scene[mask],scope)
                    if sm.any(): rows.append({'fold':name,'split':split,'model':model,'alpha':alpha,'seed':seed,'scope':scope,**seq.score(y[mask][sm],p[sm])})
                for day,ids in pd.Series(np.arange(mask.sum()),index=target[mask].strftime('%Y-%m-%d')).groupby(level=0):
                    ix=ids.to_numpy();e=p[ix]-y[mask][ix]
                    daily.append({'fold':name,'split':split,'model':model,'alpha':alpha,'seed':seed,'day_utc':day,'n':len(ix),'MAE_MW':float(abs(e).mean()),'RMSE_MW':float(np.sqrt(np.mean(e**2)))})
        persistence=tab.power_lag_0.to_numpy()
        evaluate('persistence',0,1,persistence)
        for alpha in ALPHAS:
            weights=1+(alpha-1)*ramp.astype(float)
            tree=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42)
            tree.fit(tab.loc[masks['train'],sets['local_surface']],y[masks['train']],sample_weight=None if alpha==1 else weights[masks['train']])
            pred=np.maximum(0,tree.predict(tab[sets['local_surface']]))
            tag=f'{name}_HGB_a{alpha}_seed42'
            joblib.dump({'model':tree,'features':sets['local_surface']},OUT/(tag+'.joblib'))
            evaluate('HGB',42,alpha,pred)
            if alpha==1:
                ref=pd.read_csv(ROOT/f'PVOD_Forecast/results/station05_sequence_v2/{name}_60min_predictions.csv')
                diff=float(np.max(abs(pred[masks['evaluation']]-ref.HGB_local_surface_seed42.to_numpy())))
                assert diff<1e-10
                checks.append({'run':tag,'baseline_max_abs_diff':diff})
            for seed in seq.SEEDS:
                tag=f'{name}_GRU_a{alpha}_seed{seed}'
                pred,h,epoch,val=train('GRU',seed,windows,calendar,y,persistence,masks,tag,weights,alpha)
                history+=h
                evaluate('GRU',seed,alpha,pred)
                if alpha==1:
                    ref=pd.read_csv(ROOT/f'PVOD_Forecast/results/station05_weather_ablation_v3/{name}_60min_predictions.csv')
                    diff=float(np.max(abs(pred[masks['evaluation']]-ref[f'GRU_local_surface_seed{seed}'].to_numpy())))
                    assert diff<1e-10
                    checks.append({'run':tag,'baseline_max_abs_diff':diff})
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
                pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
        for split,frame in frames.items(): frame.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'training_thresholds.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    m=pd.DataFrame(rows)
    summary=m.groupby(['fold','split','model','alpha','scope']).agg(RMSE_MW=('RMSE_MW','mean'),RMSE_seed_std_MW=('RMSE_MW','std'),MAE_MW=('MAE_MW','mean'),n=('n','first'),seed_runs=('seed','size')).reset_index()
    summary.to_csv(OUT/'seed_summary.csv',index=False)
    selections=[];gates=[];paired=[]
    for name,*_ in seq.FOLDS:
        for model in ['HGB','GRU']:
            s=summary.query('fold==@name and model==@model and split=="validation"').pivot(index='alpha',columns='scope',values='RMSE_MW')
            feasible=s[(s['all']<=GUARDRAIL*s.loc[1,'all'])&(s.day_stable<=GUARDRAIL*s.loc[1,'day_stable'])]
            chosen=int(feasible.day_ramp.idxmin())
            expert=int(s.day_ramp.idxmin())
            selections.append({'fold':name,'model':model,'selected_alpha':chosen,'expert_alpha':expert,'validation_ramp_RMSE_MW':float(s.loc[chosen,'day_ramp']),'feasible_alphas':list(map(int,feasible.index))})
        candidates=summary.query('fold==@name and split=="validation" and model!="persistence" and scope=="day_ramp"').sort_values(['RMSE_MW','model','alpha'])
        winner=candidates.iloc[0]
        baseline=float(candidates.query('model=="HGB" and alpha==1').iloc[0].RMSE_MW)
        gates.append({'fold':name,'expert_model':winner.model,'expert_alpha':int(winner.alpha),'expert_validation_ramp_RMSE_MW':float(winner.RMSE_MW),'baseline_validation_ramp_RMSE_MW':baseline,'routing_enabled':bool(winner.RMSE_MW<baseline-1e-10)})
    for (name,split,model,seed,scope),g in m[m.model!='persistence'].groupby(['fold','split','model','seed','scope']):
        b=float(g[g.alpha==1].iloc[0].RMSE_MW)
        for _,r in g.iterrows(): paired.append({'fold':name,'split':split,'model':model,'seed':int(seed),'scope':scope,'alpha':int(r.alpha),'baseline_RMSE_MW':b,'RMSE_MW':float(r.RMSE_MW),'improvement_pct':100*(1-r.RMSE_MW/b)})
    pd.DataFrame(paired).to_csv(OUT/'paired_deltas.csv',index=False)
    dump('selection.json',selections);dump('routing_gate.json',gates);dump('baseline_reproduction.json',checks)
    print(json.dumps({'selection':selections,'routing_gate':gates},indent=2),flush=True)

if __name__=='__main__': main()
