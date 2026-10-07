"""Expanding-window LSTM/GRU benchmark; all outputs isolated from v1."""
from pathlib import Path
import json
import hashlib
import time
import copy
import random
import platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_station05 as base

ROOT=base.ROOT
OUT=ROOT/'PVOD_Forecast/results/station05_sequence_v2'
SEEDS=[17,42,2026]
MAX_EPOCHS=35
PATIENCE=7
FOLDS=[('roll_1','2019-04-15','2019-04-25','2019-05-05'),
       ('roll_2','2019-05-01','2019-05-11','2019-05-21'),
       ('roll_3','2019-05-15','2019-05-30','2019-06-14')]
torch.set_num_threads(2)


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class SequenceRegressor(nn.Module):
    def __init__(self,kind,n_features):
        super().__init__()
        cls=nn.LSTM if kind=='LSTM' else nn.GRU
        self.rnn=cls(n_features,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(36,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x,calendar):
        sequence,_=self.rnn(x)
        return self.head(torch.cat([sequence[:,-1],calendar],dim=1)).squeeze(-1)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def prepare():
    old=ROOT/'PVOD_Forecast/results/station05_v1'
    manifest=json.loads((old/'manifest.json').read_text())
    for item in manifest['sources']:
        source = Path(item['file'])
        # Resolve historical source paths after repository reorganization.
        if source.parts[0] in ('PVODdatasets_v1.0', 'ERA5_data'):
            source = Path('data') / source
        assert sha(ROOT/source)==item['sha256'], 'Source changed; regenerate alignment first: '+item['file']
    data=pd.read_csv(old/'station05_aligned_15min.csv',index_col=0)
    data.index=pd.to_datetime(data.index,utc=True)
    assert (data.index.to_series().diff().dropna()==pd.Timedelta(minutes=15)).all()
    features=['power']+[c for c in data if c.startswith('lmd_') or c.startswith('era_')]
    weather=data[features].copy()
    direction=weather.pop('lmd_winddirection')
    weather['lmd_winddirection_sin']=np.sin(np.deg2rad(direction))
    weather['lmd_winddirection_cos']=np.cos(np.deg2rad(direction))
    return data,weather,manifest,old

def dataset(data,weather,horizon):
    tab,y,target,sets=base.features(data,horizon)
    windows=np.stack([weather.shift(lag).reindex(tab.index).to_numpy() for lag in range(15,-1,-1)],axis=1)
    valid=np.isfinite(windows).all(axis=(1,2))
    tab=tab.loc[valid];y=y.loc[valid].to_numpy();target=target[valid];windows=windows[valid]
    calendar=tab[[c for c in tab if c.startswith('target_')]].to_numpy()
    return tab,y,target,sets,windows,calendar

def score(y,p): return base.metrics(y,p)

@torch.no_grad()
def predict(model,x,c):
    model.eval()
    return np.concatenate([model(x[i:i+512],c[i:i+512]).numpy() for i in range(0,len(x),512)])

def train(kind,seed,windows,calendar,y,persistence,masks,tag):
    seed_all(seed)
    # Fit feature scaler on unique historical timestamps used by training windows;
    # flattening weights overlapping observations but uses no validation/test information.
    sx=StandardScaler().fit(windows[masks['train']].reshape(-1,windows.shape[-1]))
    sc=StandardScaler().fit(calendar[masks['train']])
    residual=y-persistence
    sy=StandardScaler().fit(residual[masks['train']].reshape(-1,1))
    xx=torch.from_numpy(sx.transform(windows.reshape(-1,windows.shape[-1])).reshape(windows.shape).astype('float32'))
    cc=torch.from_numpy(sc.transform(calendar).astype('float32'))
    yy=torch.from_numpy(sy.transform(residual.reshape(-1,1)).ravel().astype('float32'))
    model=SequenceRegressor(kind,windows.shape[-1])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    lossfn=nn.MSELoss()
    ids=np.flatnonzero(masks['train'])
    best=float('inf');stale=0;beststate=None;history=[];bestepoch=0
    start=time.monotonic()
    for epoch in range(1,MAX_EPOCHS+1):
        model.train();total=0
        # Fixed chronological batches, independent overlapping windows; hidden state reset per batch.
        for i in range(0,len(ids),256):
            batch=ids[i:i+256]
            optimizer.zero_grad(set_to_none=True)
            loss=lossfn(model(xx[batch],cc[batch]),yy[batch])
            assert torch.isfinite(loss), f'Nonfinite loss: {tag}'
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=loss.item()*len(batch)
        vp=persistence[masks['validation']]+sy.inverse_transform(predict(model,xx[masks['validation']],cc[masks['validation']]).reshape(-1,1)).ravel()
        val=score(y[masks['validation']],np.maximum(0,vp))['RMSE_MW']
        history.append({'run':tag,'epoch':epoch,'train_scaled_MSE':total/len(ids),'validation_RMSE_MW':val})
        if val<best-1e-5:
            best=val;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch;stale=0
        else: stale+=1
        if stale>=PATIENCE: break
    model.load_state_dict(beststate)
    torch.save({'kind':kind,'state_dict':beststate,'input_size':windows.shape[-1],'hidden_size':32,'best_epoch':bestepoch,'seed':seed},OUT/(tag+'.pt'))
    joblib.dump({'sequence_scaler':sx,'calendar_scaler':sc,'residual_scaler':sy,'feature_names':FEATURE_NAMES,'calendar_names':CALENDAR_NAMES},OUT/(tag+'_scalers.joblib'))
    ep=np.maximum(0,persistence[masks['evaluation']]+sy.inverse_transform(predict(model,xx[masks['evaluation']],cc[masks['evaluation']]).reshape(-1,1)).ravel())
    assert np.isfinite(ep).all()
    print(f'{tag}: epoch={bestepoch} val_RMSE={best:.4f} elapsed={time.monotonic()-start:.1f}s',flush=True)
    return ep,history,bestepoch,best

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,weather,manifest,old=prepare()
    global FEATURE_NAMES,CALENDAR_NAMES
    FEATURE_NAMES=list(weather)
    CALENDAR_NAMES=['target_hour_sin','target_hour_cos','target_doy_sin','target_doy_cos']
    dump('manifest.json',{'original_sources':manifest['sources'],'aligned_file':str((old/'station05_aligned_15min.csv').relative_to(ROOT)),'aligned_sha256':sha(old/'station05_aligned_15min.csv'),'script_sha256':sha(Path(__file__)),'base_script_sha256':sha(Path(base.__file__)),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','threads':2})
    dump('config.json',{'seeds':SEEDS,'folds':FOLDS,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,'history_steps':16,'horizons_minutes':[15,60],'sequence_features':FEATURE_NAMES,'calendar_features':CALENDAR_NAMES,'architecture':'1 layer LSTM/GRU hidden=32 + Dense32/ReLU/Dense1; predict residual over persistence','optimizer':'Adam lr=0.001 weight_decay=0.0001 gradient_clip=1 batch=256','scaler':'train windows only; flattened overlapping timestamps weighted by occurrence','selection':'chronological validation RMSE checkpoint, no evaluation tuning','split_rule':'train target < validation start; validation origin >= validation start and target < evaluation start; evaluation origin >= evaluation start and target < evaluation end','excluded_utc_dates':['2019-05-31'],'era5_caveat':'retrospective valid-time inputs, publication latency not simulated','evaluation_caveat':'all evaluation periods lie in previously inspected historical data; roll_3 overlaps v1 test; exploratory, not fresh holdout','comparison_caveat':'RNN receives 16 observations of all weather fields; tree uses same power history but weather at origin only; algorithm and representation both differ'})
    rows=[];histories=[];splits=[];daily=[];checkpoints=[]
    for horizon in [1,4]:
        tab,y,target,sets,windows,calendar=dataset(data,weather,horizon)
        origin=tab.index
        persistence=tab.power_lag_0.to_numpy()
        for fold,valstart,evalstart,evalend in FOLDS:
            vs,es,ee=[pd.Timestamp(v,tz='UTC') for v in [valstart,evalstart,evalend]]
            masks={'train':target<vs,'validation':(origin>=vs)&(target<es),'evaluation':(origin>=es)&(target<ee)}
            assert target[masks['train']].max()<origin[masks['validation']].min()
            assert target[masks['validation']].max()<origin[masks['evaluation']].min()
            for split,mask in masks.items():
                splits.append({'horizon_minutes':horizon*15,'fold':fold,'split':split,'n':int(mask.sum()),'first_origin':str(origin[mask].min()),'last_target':str(target[mask].max())})
            labels=np.select([masks['train'],masks['validation'],masks['evaluation']],['train','validation','evaluation'],default='unused')
            pd.DataFrame({'origin_utc':origin,'target_utc':target,'split':labels}).to_csv(OUT/f'{fold}_{horizon*15}min_split.csv',index=False)
            ev=masks['evaluation'];actual=y[ev];irrad=data.lmd_totalirrad.reindex(target[ev]).to_numpy()
            frame=pd.DataFrame({'origin_utc':origin[ev],'target_utc':target[ev],'actual_MW':actual,'target_irradiance_Wm2':irrad})
            def evaluate(name,seed,p):
                assert np.isfinite(p).all()
                frame[f'{name}_seed{seed}']=p
                for scope,mask in [('all',np.ones(len(p),dtype=bool)),('daylight',irrad>20)]:
                    rows.append({'horizon_minutes':horizon*15,'fold':fold,'model':name,'seed':seed,'scope':scope,**score(actual[mask],p[mask])})
                d=pd.DataFrame({'day':target[ev].strftime('%Y-%m-%d'),'error':p-actual,'absolute_error':abs(p-actual)})
                for day,part in d.groupby('day'):
                    daily.append({'horizon_minutes':horizon*15,'fold':fold,'model':name,'seed':seed,'day_utc':day,'n':len(part),'MAE_MW':float(part.absolute_error.mean()),'RMSE_MW':float(np.sqrt(np.mean(part.error**2)))})
            evaluate('persistence',0,persistence[ev])
            for subset in ['local','local_surface','local_surface_pressure']:
                cols=sets[subset]
                tree=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42)
                tree.fit(tab.loc[masks['train'],cols],y[masks['train']])
                evaluate('HGB_'+subset,42,np.maximum(0,tree.predict(tab.loc[ev,cols])))
                joblib.dump({'model':tree,'features':cols},OUT/f'{fold}_{horizon*15}min_HGB_{subset}.joblib')
            for kind in ['LSTM','GRU']:
                for seed in SEEDS:
                    tag=f'{fold}_{horizon*15}min_{kind}_seed{seed}'
                    ep,history,epoch,val=train(kind,seed,windows,calendar,y,persistence,masks,tag)
                    histories+=history
                    checkpoints.append({'run':tag,'best_epoch':epoch,'validation_RMSE_MW':val})
                    evaluate(kind,seed,ep)
                    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
                    pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
            frame.to_csv(OUT/f'{fold}_{horizon*15}min_predictions.csv',index=False)
            pd.DataFrame(splits).to_csv(OUT/'split_summary.csv',index=False)
            pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
            pd.DataFrame(checkpoints).to_csv(OUT/'checkpoints.csv',index=False)
    m=pd.DataFrame(rows)
    agg=m.groupby(['horizon_minutes','fold','model','scope']).agg(RMSE_mean_MW=('RMSE_MW','mean'),RMSE_seed_std_MW=('RMSE_MW','std'),MAE_mean_MW=('MAE_MW','mean'),seed_runs=('seed','size'),n=('n','first')).reset_index()
    agg.to_csv(OUT/'seed_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    colors=['#444444','#9faeb7','#83b4c9','#3575a8','#bf8c44','#743f85']
    for ax,h in zip(axes,[15,60]):
        f=agg.query('horizon_minutes==@h and scope=="all"')
        models=list(f.model.unique());width=.12
        for k,model in enumerate(models):
            r=f[f.model==model].set_index('fold').reindex([x[0] for x in FOLDS])
            ax.bar(np.arange(3)+(k-2.5)*width,r.RMSE_mean_MW,width,label=model,color=colors[k],yerr=r.RMSE_seed_std_MW.fillna(0),capsize=2)
        ax.set_xticks(range(3),['Apr25-May4','May11-May20','May30-Jun13'])
        ax.set_ylabel('RMSE (MW)');ax.set_title(f'{h}-minute rolling evaluation');ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,title='RNN bars: mean +/- sample std over 3 seeds (not confidence intervals)')
    fig.savefig(OUT/'rolling_rmse.png',dpi=160);plt.close(fig)
    print(agg.query('scope=="all"').to_string(index=False),flush=True)
    print('Complete: '+str(OUT),flush=True)

if __name__=='__main__': main()
