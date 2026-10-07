"""Saved v5 HGB routing inference from origin-available feature rows only."""
import argparse,json
import numpy as np
import pandas as pd
import joblib
import run_weighted_loss_v5 as w
from scipy.special import logit

R=w.ROOT/'PVOD_Forecast/results/station05_routing_v5'

def predict(fold,features):
    choice=next(c for c in json.loads((R/'selection.json').read_text()) if c['fold']==fold)
    base=joblib.load(w.OUT/f'{fold}_HGB_a1_seed42.joblib')
    x=features[base['features']]
    if not np.isfinite(x.to_numpy(dtype=float)).all(): raise ValueError('Features must be finite numeric origin-available values')
    b=np.maximum(0,base['model'].predict(x))
    result=pd.DataFrame({'baseline_MW':b,'selected_MW':b},index=features.index)
    if choice['selected']=='baseline': return result
    if choice['expert_model']!='HGB': raise ValueError('This v5 saved-model interface supports the HGB specialists selected in this run')
    expert=joblib.load(w.OUT/f"{fold}_HGB_a{choice['expert_alpha']}_seed42.joblib")
    e=np.maximum(0,expert['model'].predict(features[expert['features']]))
    cls=joblib.load(R/f"{fold}_{choice['classifier']}_classifier.joblib")
    cx=features[cls['features']]
    raw=cls['model'].decision_function(cx) if hasattr(cls['model'],'decision_function') else logit(np.clip(cls['model'].predict_proba(cx)[:,1],1e-6,1-1e-6))
    p=cls['calibrator'].predict_proba(raw.reshape(-1,1))[:,1]
    mix=p if choice['mode']=='soft' else (p>=choice['threshold']).astype(float)
    result['expert_MW']=e;result['ramp_probability']=p;result['selected_MW']=(1-mix)*b+mix*e
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',choices=['roll_1','roll_2','roll_3'],required=True)
    parser.add_argument('--input',required=True,help='CSV containing the saved tree origin-available features, without target observations')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    features=pd.read_csv(args.input,index_col='origin_utc')
    predict(args.fold,features).to_csv(args.output)
