"""Reload checkpoints and independently recompute improvement and routing results."""
import json
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.special import logit
from sklearn.metrics import mean_squared_error, mean_absolute_error, average_precision_score, brier_score_loss
import run_weighted_loss_v5 as w

ROUTE=w.ROOT/'PVOD_Forecast/results/station05_routing_v5'

def score(y,p): return float(np.sqrt(mean_squared_error(y,p))),float(mean_absolute_error(y,p))

def main():
    data,tab,y,target,sets,windows,calendar,names,active,*_=w.inputs()
    counts={'weighted_metric_rows':0,'routing_metric_rows':0,'GRU_reloaded':0,'HGB_reloaded':0,'classifiers_reloaded':0,'classification_metric_rows':0}
    maxdiff=0.0
    wm=pd.read_csv(w.OUT/'metrics.csv');rm=pd.read_csv(ROUTE/'metrics.csv')
    cm=pd.read_csv(ROUTE/'classification_metrics.csv')
    choices=json.loads((w.OUT/'selection.json').read_text());routes=json.loads((ROUTE/'selection.json').read_text())
    gates=json.loads((w.OUT/'routing_gate.json').read_text())
    for fold in w.seq.FOLDS:
        name=fold[0];masks,scenes,ramps,tau=w.fold_data(data,tab,y,target,fold)
        # Independently verify excluded day and full origin-target continuity.
        for split in ['train','validation','evaluation']:
            for lag in range(-4,16):
                dates=(tab.index[masks[split]]-pd.Timedelta(minutes=15*lag)).strftime('%Y-%m-%d')
                assert not (dates=='2019-05-31').any()
        daylight=data.lmd_totalirrad.reindex(target).to_numpy()>20
        delta=y-tab.power_lag_0.to_numpy()
        assert tau==float(np.percentile(abs(delta[masks['train']&daylight]),90))
        frames={s:pd.read_csv(w.OUT/f'{name}_{s}_predictions.csv') for s in ['validation','evaluation']}
        for split,frame in frames.items():
            assert np.array_equal(pd.to_datetime(frame.origin_utc,utc=True),tab.index[masks[split]])
            assert np.allclose(frame.actual_MW,y[masks[split]],atol=1e-12,rtol=0)
            assert np.array_equal(frame.ramp.to_numpy(),ramps[masks[split]])
            for _,row in wm[(wm.fold==name)&(wm.split==split)].iterrows():
                mask=w.scope_mask(frame.scene.to_numpy(),row.scope)
                p=frame[f'{row.model}_a{row.alpha}_seed{row.seed}'].to_numpy()
                r,a=score(frame.actual_MW.to_numpy()[mask],p[mask])
                assert int(mask.sum())==row.n and abs(r-row.RMSE_MW)<1e-10 and abs(a-row.MAE_MW)<1e-10
                counts['weighted_metric_rows']+=1
        persistence=tab.power_lag_0.to_numpy()
        for alpha in w.ALPHAS:
            bundle=joblib.load(w.OUT/f'{name}_HGB_a{alpha}_seed42.joblib')
            p=np.maximum(0,bundle['model'].predict(tab[bundle['features']]))
            for split,frame in frames.items():
                d=float(np.max(abs(p[masks[split]]-frame[f'HGB_a{alpha}_seed42'].to_numpy())))
                assert d<1e-10;maxdiff=max(maxdiff,d)
            counts['HGB_reloaded']+=1
            for seed in w.seq.SEEDS:
                tag=f'{name}_GRU_a{alpha}_seed{seed}'
                checkpoint=torch.load(w.OUT/(tag+'.pt'),map_location='cpu',weights_only=True)
                scalers=joblib.load(w.OUT/(tag+'_scalers.joblib'))
                assert np.array_equal(scalers['sequence_scaler'].mean_[~active],np.zeros((~active).sum()))
                assert np.array_equal(scalers['sequence_scaler'].scale_[~active],np.ones((~active).sum()))
                model=w.seq.SequenceRegressor('GRU',len(names));model.load_state_dict(checkpoint['state_dict'])
                assert sum(v.numel() for v in model.parameters())==7169
                sx,sc,sy=[scalers[k] for k in ['sequence_scaler','calendar_scaler','residual_scaler']]
                assert np.allclose(sx.mean_,windows[masks['train']].reshape(-1,len(names)).mean(0),atol=1e-10)
                for split,frame in frames.items():
                    ww=windows[masks[split]]
                    xx=torch.from_numpy(sx.transform(ww.reshape(-1,len(names))).reshape(ww.shape).astype('float32'))
                    cc=torch.from_numpy(sc.transform(calendar[masks[split]]).astype('float32'))
                    p=np.maximum(0,persistence[masks[split]]+sy.inverse_transform(w.seq.predict(model,xx,cc).reshape(-1,1)).ravel())
                    d=float(np.max(abs(p-frame[f'GRU_a{alpha}_seed{seed}'].to_numpy())))
                    # Batch shapes can cause float32 differences relative to all-row prediction.
                    assert d<2e-5;maxdiff=max(maxdiff,d)
                counts['GRU_reloaded']+=1
        val=wm[(wm.fold==name)&(wm.split=='validation')].groupby(['model','alpha','scope']).RMSE_MW.mean()
        for model in ['GRU','HGB']:
            chosen=next(c for c in choices if c['fold']==name and c['model']==model)
            feasible=[a for a in w.ALPHAS if all(val[model,a,s]<=1.02*val[model,1,s] for s in ['all','day_stable'])]
            assert chosen['selected_alpha']==min(feasible,key=lambda a:val[model,a,'day_ramp'])
        route=next(r for r in routes if r['fold']==name)
        gate=next(g for g in gates if g['fold']==name)
        assert route['routing_enabled']==gate['routing_enabled']
        if not gate['routing_enabled']: continue
        audit=pd.read_csv(ROUTE/'classifier_split_audit.csv').query('fold==@name').iloc[0]
        assert pd.Timestamp(audit.last_fit_target)<pd.Timestamp(audit.first_calibration_origin)
        assert pd.Timestamp(audit.last_calibration_target)<pd.Timestamp(audit.first_validation_origin)
        for split in ['validation','evaluation']:
            f=pd.read_csv(ROUTE/f'{name}_{split}_predictions.csv')
            for classifier in ['LR','HGB']:
                b=joblib.load(ROUTE/f'{name}_{classifier}_classifier.joblib')
                x=tab.loc[masks[split],b['features']]
                raw=b['model'].decision_function(x) if hasattr(b['model'],'decision_function') else logit(np.clip(b['model'].predict_proba(x)[:,1],1e-6,1-1e-6))
                p=b['calibrator'].predict_proba(raw.reshape(-1,1))[:,1]
                assert np.max(abs(p-f[f'{classifier}_ramp_probability'].to_numpy()))<1e-10
                row=cm.query('fold==@name and split==@split and classifier==@classifier').iloc[0]
                assert abs(row.PR_AUC_average_precision-average_precision_score(f.ramp,p))<1e-10
                assert abs(row.Brier-brier_score_loss(f.ramp,p))<1e-10
                counts['classification_metric_rows']+=1
            if split=='validation': counts['classifiers_reloaded']+=2
            for _,row in rm[(rm.fold==name)&(rm.split==split)].iterrows():
                column='baseline' if row.model=='baseline' else f'{row.model}_seed{row.seed}'
                sm=w.scope_mask(f.scene.to_numpy(),row.scope)
                r,a=score(f.actual_MW.to_numpy()[sm],f[column].to_numpy()[sm])
                assert int(sm.sum())==row.n and abs(r-row.RMSE_MW)<1e-10 and abs(a-row.MAE_MW)<1e-10
                counts['routing_metric_rows']+=1
                if row.model.startswith(('LR_','HGB_')):
                    classifier,mode,*eta=row.model.split('_')
                    prob=f[f'{classifier}_ramp_probability'].to_numpy()
                    mix=prob if mode=='soft' else (prob>=float(eta[0])).astype(float)
                    expect=(1-mix)*f.baseline.to_numpy()+mix*f[f'expert_seed{row.seed}'].to_numpy()
                    assert np.max(abs(expect-f[column].to_numpy()))<1e-10
                if row.model=='oracle':
                    expect=np.where(f.ramp,f[f'expert_seed{row.seed}'],f.baseline)
                    assert np.max(abs(expect-f[column].to_numpy()))<1e-10
                if row.model=='selected':
                    if route['selected']=='baseline': expect=f.baseline.to_numpy()
                    else:
                        prob=f[f"{route['classifier']}_ramp_probability"].to_numpy()
                        mix=prob if route['mode']=='soft' else (prob>=route['threshold']).astype(float)
                        expect=(1-mix)*f.baseline.to_numpy()+mix*f[f'expert_seed{row.seed}'].to_numpy()
                    assert np.max(abs(expect-f[column].to_numpy()))<1e-10
        candidates=pd.read_csv(ROUTE/'validation_candidates.csv').query('fold==@name')
        for _,candidate in candidates.iterrows():
            records=rm[(rm.fold==name)&(rm.split=='validation')&(rm.model==candidate.key)]
            for scope in w.SCOPES:
                assert abs(float(records[records.scope==scope].RMSE_MW.mean())-float(candidate[scope]))<1e-10
        base=rm.query('fold==@name and split=="validation" and model=="baseline"').set_index('scope')
        feasible=candidates[(candidates['all']<=1.02*base.loc['all','RMSE_MW'])&(candidates.day_stable<=1.02*base.loc['day_stable','RMSE_MW'])]
        best=feasible.sort_values(['day_ramp','key']).iloc[0] if len(feasible) else None
        expected='baseline' if best is None or best.day_ramp>=base.loc['day_ramp','RMSE_MW']-1e-10 else best.key
        assert route['selected']==expected
    assert counts['weighted_metric_rows']==len(wm) and counts['routing_metric_rows']==len(rm)
    result={'status':'passed',**counts,'max_reload_abs_difference_MW':maxdiff,'baseline_reproduction':json.loads((w.OUT/'baseline_reproduction.json').read_text()),'checks':['All validation/evaluation MAE and RMSE independently recomputed','All 36 regression checkpoints reloaded on full validation/evaluation','Four classifiers and calibration models reloaded','Training-only thresholds and scalers verified','Time boundaries and missing-date windows checked','Validation-only choices and 2% constraints independently reproduced','Hard, soft, oracle and selected predictions reconstructed']}
    (w.OUT/'validation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROUTE/'validation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
