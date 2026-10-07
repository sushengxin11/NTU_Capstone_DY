"""Calibrated ramp routing, gated only by outer-validation specialist advantage."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix, brier_score_loss, log_loss
import joblib
import run_weighted_loss_v5 as weighted

ROOT=weighted.ROOT
OUT=ROOT/'PVOD_Forecast/results/station05_routing_v5'
THRESHOLDS=np.round(np.arange(.10,.901,.05),2)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def raw_score(model,x):
    if hasattr(model,'decision_function'): return model.decision_function(x)
    return logit(np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6))

def regression_metrics(y,pred,scene):
    return {scope:weighted.seq.score(y[mask],pred[mask]) for scope in weighted.SCOPES if (mask:=weighted.scope_mask(scene,scope)).any()}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gates=json.loads((weighted.OUT/'routing_gate.json').read_text())
    data,tab,y,target,sets,*_=weighted.inputs()
    cols=sets['local_surface']
    dump('config.json',{'guardrail_ratio':1.02,'gate_sha256':weighted.seq.sha(weighted.OUT/'routing_gate.json'),'classifiers':['LR balanced C=1 max_iter=2000','HGB max_iter180 leaves15 lr0.05 l2=1 early_stopping=False seed42 balanced sample weights'],'fit_calibration':'chronological 80/20 of outer training; fit target strictly before first calibration origin; sigmoid LR C=1 on raw scores, unweighted calibration','thresholds':THRESHOLDS.tolist(),'classifier_features':cols,'selection':'min validation ramp RMSE under all/stable <=1.02 HGB baseline, candidate baseline included; seed means for GRU specialist','GRU_specialist':'three parallel routed models, averaged metrics; no best-seed selection or forecast ensembling','label':'same one-hour daylight net-change event as weighted v5','oracle':'future-label routing diagnostic only, not deployable or a theoretical upper bound','evaluation':'previously inspected historical data, exploratory; ERA5 release latency not simulated'})
    classification=[];calibration=[];metrics=[];val_candidates=[];selections=[];split_audit=[];error_groups=[];daily=[]
    for fold in weighted.seq.FOLDS:
        name=fold[0];gate=next(g for g in gates if g['fold']==name)
        masks,scene,ramp,tau=weighted.fold_data(data,tab,y,target,fold)
        frames={s:pd.read_csv(weighted.OUT/f'{name}_{s}_predictions.csv') for s in ['validation','evaluation']}
        if not gate['routing_enabled']:
            selections.append({'fold':name,'routing_enabled':False,'selected':'baseline','reason':'No validation specialist advantage over baseline HGB'})
            print(name+': routing skipped by validation gate',flush=True)
            continue
        expert_model=gate['expert_model'];alpha=gate['expert_alpha']
        seeds=weighted.seq.SEEDS if expert_model=='GRU' else [42]
        train_ids=np.flatnonzero(masks['train']);cut=int(.8*len(train_ids));cal_ids=train_ids[cut:]
        fit_ids=train_ids[:cut];fit_ids=fit_ids[target[fit_ids]<tab.index[cal_ids[0]]]
        assert target[fit_ids].max()<tab.index[cal_ids].min()
        assert len(np.unique(ramp[fit_ids]))==2 and len(np.unique(ramp[cal_ids]))==2
        split_audit.append({'fold':name,'fit_n':len(fit_ids),'calibration_n':len(cal_ids),'purged_n':cut-len(fit_ids),'last_fit_target':str(target[fit_ids].max()),'first_calibration_origin':str(tab.index[cal_ids[0]]),'last_calibration_target':str(target[cal_ids].max()),'first_validation_origin':str(tab.index[masks['validation']].min())})
        local_val=[];predictions={s:{} for s in frames}
        for classifier in ['LR','HGB']:
            if classifier=='LR':
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42))
                model.fit(tab.iloc[fit_ids][cols],ramp[fit_ids].astype(int))
            else:
                model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42)
                model.fit(tab.iloc[fit_ids][cols],ramp[fit_ids].astype(int),sample_weight=compute_sample_weight('balanced',ramp[fit_ids]))
            calibrator=LogisticRegression(C=1,max_iter=2000)
            calibrator.fit(raw_score(model,tab.iloc[cal_ids][cols]).reshape(-1,1),ramp[cal_ids].astype(int))
            joblib.dump({'model':model,'calibrator':calibrator,'features':cols},OUT/f'{name}_{classifier}_classifier.joblib')
            for split,frame in frames.items():
                prob=calibrator.predict_proba(raw_score(model,tab.loc[masks[split],cols]).reshape(-1,1))[:,1]
                frame[f'{classifier}_ramp_probability']=prob
                truth=frame.ramp.to_numpy(dtype=bool)
                hard=prob>=.5
                tn,fp,fn,tp=map(int,confusion_matrix(truth,hard,labels=[False,True]).ravel())
                classification.append({'fold':name,'split':split,'classifier':classifier,'threshold':.5,'n':len(truth),'event_n':int(truth.sum()),'event_frequency':float(truth.mean()),'PR_AUC_average_precision':float(average_precision_score(truth,prob)),'precision':float(precision_score(truth,hard,zero_division=0)),'recall':float(recall_score(truth,hard,zero_division=0)),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Brier':float(brier_score_loss(truth,prob)),'log_loss':float(log_loss(truth,prob,labels=[False,True]))})
                for lo in np.arange(0,1,.1):
                    bm=(prob>=lo)&(prob<lo+.1 if lo<.9 else prob<=1)
                    if bm.any(): calibration.append({'fold':name,'split':split,'classifier':classifier,'bin_lower':float(lo),'n':int(bm.sum()),'mean_probability':float(prob[bm].mean()),'event_frequency':float(truth[bm].mean())})
                base=frame.HGB_a1_seed42.to_numpy()
                modes=[('soft',None)]+[('hard',float(t)) for t in THRESHOLDS]
                if split=='evaluation':
                    # Hard-route threshold for each classifier is selected using validation only.
                    vc=pd.DataFrame(local_val)
                    candidates=vc[(vc.classifier==classifier)&(vc['mode']=='hard')]
                    reference=regression_metrics(frames['validation'].actual_MW.to_numpy(),frames['validation'].HGB_a1_seed42.to_numpy(),frames['validation'].scene.to_numpy())
                    valid=candidates[(candidates['all']<=1.02*reference['all']['RMSE_MW'])&(candidates.day_stable<=1.02*reference['day_stable']['RMSE_MW'])]
                    row=(valid if len(valid) else candidates).sort_values(['day_ramp','threshold']).iloc[0]
                    modes=[('soft',None),('hard',float(row.threshold))]
                for mode,eta in modes:
                    key=classifier+'_'+mode+('' if eta is None else f'_{eta:.2f}')
                    each=[]
                    for seed in seeds:
                        expert=frame[f'{expert_model}_a{alpha}_seed{seed}'].to_numpy()
                        mix=prob if mode=='soft' else (prob>=eta).astype(float)
                        pred=(1-mix)*base+mix*expert
                        each.append(regression_metrics(frame.actual_MW.to_numpy(),pred,frame.scene.to_numpy()))
                        predictions[split][key+f'_seed{seed}']=pred
                    if split=='validation':
                        local_val.append({'fold':name,'classifier':classifier,'mode':mode,'threshold':eta,'key':key,**{scope:float(np.mean([r[scope]['RMSE_MW'] for r in each])) for scope in weighted.SCOPES}})
        reference=regression_metrics(frames['validation'].actual_MW.to_numpy(),frames['validation'].HGB_a1_seed42.to_numpy(),frames['validation'].scene.to_numpy())
        vc=pd.DataFrame(local_val);valid=vc[(vc['all']<=1.02*reference['all']['RMSE_MW'])&(vc.day_stable<=1.02*reference['day_stable']['RMSE_MW'])]
        chosen=None
        if len(valid):
            candidate=valid.sort_values(['day_ramp','key']).iloc[0]
            if candidate.day_ramp<reference['day_ramp']['RMSE_MW']-1e-10: chosen=candidate
        selections.append({'fold':name,'routing_enabled':True,'expert_model':expert_model,'expert_alpha':alpha,'selected':'baseline' if chosen is None else chosen.key,'validation_ramp_RMSE_MW':reference['day_ramp']['RMSE_MW'] if chosen is None else float(chosen.day_ramp),'classifier':None if chosen is None else chosen.classifier,'mode':None if chosen is None else chosen['mode'],'threshold':None if chosen is None or pd.isna(chosen.threshold) else float(chosen.threshold),'feasible_routing_candidates':len(valid)})
        val_candidates+=local_val
        for split,frame in frames.items():
            frame['baseline']=frame.HGB_a1_seed42
            base=frame.baseline.to_numpy();truth=frame.ramp.to_numpy(dtype=bool)
            for seed in seeds:
                expert=frame[f'{expert_model}_a{alpha}_seed{seed}'].to_numpy()
                frame[f'expert_seed{seed}']=expert
                frame[f'oracle_seed{seed}']=np.where(truth,expert,base)
            for key,pred in predictions[split].items(): frame[key]=pred
            if chosen is None: frame['selected_seed42']=base
            else:
                # The selected threshold may differ from a classifier's best hard-only candidate.
                p=frame[f'{chosen.classifier}_ramp_probability'].to_numpy()
                mix=p if chosen['mode']=='soft' else (p>=float(chosen.threshold)).astype(float)
                for seed in seeds: frame[f'selected_seed{seed}']=(1-mix)*base+mix*frame[f'expert_seed{seed}'].to_numpy()
            report_cols=['baseline']+[c for c in frame if c.startswith(('expert_seed','oracle_seed','selected_seed'))]+list(predictions[split])
            for column in report_cols:
                pred=frame[column].to_numpy()
                if '_seed' in column: mode,seed=column.rsplit('_seed',1);seed=int(seed)
                else: mode=column;seed=42
                for scope,scores in regression_metrics(frame.actual_MW.to_numpy(),pred,frame.scene.to_numpy()).items(): metrics.append({'fold':name,'split':split,'model':mode,'seed':seed,'scope':scope,**scores})
                for day,part in frame.assign(error=pred-frame.actual_MW.to_numpy()).groupby(frame.target_utc.str[:10]):
                    daily.append({'fold':name,'split':split,'model':mode,'seed':seed,'day_utc':day,'n':len(part),'MAE_MW':float(abs(part.error).mean()),'RMSE_MW':float(np.sqrt(np.mean(part.error**2)))})
            for classifier in ['LR','HGB']:
                thresholds=[.5]
                if chosen is not None and chosen.classifier==classifier and chosen['mode']=='hard': thresholds.append(float(chosen.threshold))
                for eta in sorted(set(thresholds)):
                    predicted=frame[f'{classifier}_ramp_probability'].to_numpy()>=eta
                    for group,mask in [('TP',truth&predicted),('FP',~truth&predicted),('FN',truth&~predicted),('TN',~truth&~predicted)]:
                        for seed in seeds:
                            expert=frame[f'expert_seed{seed}'].to_numpy();p=np.where(predicted,expert,base)
                            e=p[mask]-frame.actual_MW.to_numpy()[mask];be=base[mask]-frame.actual_MW.to_numpy()[mask]
                            error_groups.append({'fold':name,'split':split,'classifier':classifier,'threshold':eta,'seed':seed,'group':group,'n':int(mask.sum()),'squared_error_sum':float(np.sum(e**2)),'baseline_squared_error_sum':float(np.sum(be**2)),'delta_squared_error_sum':float(np.sum(e**2-be**2))})
            frame.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
        print(name+': '+selections[-1]['selected'],flush=True)
    pd.DataFrame(classification).to_csv(OUT/'classification_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration_bins.csv',index=False)
    pd.DataFrame(val_candidates).to_csv(OUT/'validation_candidates.csv',index=False)
    pd.DataFrame(split_audit).to_csv(OUT/'classifier_split_audit.csv',index=False)
    pd.DataFrame(error_groups).to_csv(OUT/'classification_error_costs.csv',index=False)
    pd.DataFrame(daily).to_csv(OUT/'daily_metrics.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    if metrics:
        pd.DataFrame(metrics).groupby(['fold','split','model','scope']).agg(RMSE_MW=('RMSE_MW','mean'),RMSE_seed_std_MW=('RMSE_MW','std'),MAE_MW=('MAE_MW','mean'),n=('n','first'),seed_runs=('seed','size')).reset_index().to_csv(OUT/'seed_summary.csv',index=False)
    dump('selection.json',selections)
    dump('manifest.json',{'script_sha256':weighted.seq.sha(Path(__file__)),'weighted_manifest_sha256':weighted.seq.sha(weighted.OUT/'manifest.json'),'gate_sha256':weighted.seq.sha(weighted.OUT/'routing_gate.json')})
    print(json.dumps(selections,indent=2),flush=True)

if __name__=='__main__': main()
