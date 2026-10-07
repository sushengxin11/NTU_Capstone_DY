"""Reproducible evidence tables and static research figures for v5."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, average_precision_score, brier_score_loss
import run_weighted_loss_v5 as w

R=w.ROOT/'PVOD_Forecast/results/station05_routing_v5'

def main():
    weighted=pd.read_csv(w.OUT/'seed_summary.csv')
    routing=pd.read_csv(R/'seed_summary.csv')
    choices=json.loads((R/'selection.json').read_text())
    comparisons=[];selected_cls=[];selected_costs=[]
    for choice in choices:
        name=choice['fold']
        for split in ['validation','evaluation']:
            if not choice['routing_enabled']:
                frame=pd.read_csv(w.OUT/f'{name}_{split}_predictions.csv')
                frame['baseline']=frame.HGB_a1_seed42;frame['selected_seed42']=frame.HGB_a1_seed42
            else: frame=pd.read_csv(R/f'{name}_{split}_predictions.csv')
            for scope in w.SCOPES:
                sm=w.scope_mask(frame.scene.to_numpy(),scope)
                base=w.seq.score(frame.actual_MW.to_numpy()[sm],frame.baseline.to_numpy()[sm])
                cols=[c for c in frame if c.startswith('selected_seed')]
                scores=[w.seq.score(frame.actual_MW.to_numpy()[sm],frame[c].to_numpy()[sm]) for c in cols]
                after=float(np.mean([s['RMSE_MW'] for s in scores]))
                comparisons.append({'fold':name,'split':split,'selected':choice['selected'],'scope':scope,'n':int(sm.sum()),'baseline_RMSE_MW':base['RMSE_MW'],'selected_RMSE_MW':after,'improvement_pct':100*(1-after/base['RMSE_MW']),'baseline_MAE_MW':base['MAE_MW'],'selected_MAE_MW':float(np.mean([s['MAE_MW'] for s in scores]))})
            if choice['routing_enabled'] and choice['selected']!='baseline':
                classifier=choice['classifier'];eta=choice['threshold'] if choice['mode']=='hard' else .5
                truth=frame.ramp.to_numpy(dtype=bool);p=frame[f'{classifier}_ramp_probability'].to_numpy();flag=p>=eta
                tn,fp,fn,tp=map(int,confusion_matrix(truth,flag,labels=[False,True]).ravel())
                selected_cls.append({'fold':name,'split':split,'classifier':classifier,'mode':choice['mode'],'threshold':eta,'threshold_role':'actual hard route' if choice['mode']=='hard' else '0.5 diagnostic; soft route has no threshold','n':len(truth),'event_n':int(truth.sum()),'event_frequency':float(truth.mean()),'PR_AUC_average_precision':float(average_precision_score(truth,p)),'precision':float(precision_score(truth,flag,zero_division=0)),'recall':float(recall_score(truth,flag,zero_division=0)),'TN':tn,'FP':fp,'FN':fn,'TP':tp,'Brier':float(brier_score_loss(truth,p))})
                e=frame.selected_seed42.to_numpy()-frame.actual_MW.to_numpy();be=frame.baseline.to_numpy()-frame.actual_MW.to_numpy()
                group_masks=[('TP',truth&flag),('FP',~truth&flag),('FN',truth&~flag),('TN',~truth&~flag)]
                assert sum(int(sm.sum()) for _,sm in group_masks)==len(frame)
                for group,sm in group_masks:
                    selected_costs.append({'fold':name,'split':split,'selected':choice['selected'],'group':group,'n':int(sm.sum()),'selected_SSE':float(np.sum(e[sm]**2)),'baseline_SSE':float(np.sum(be[sm]**2)),'delta_SSE':float(np.sum(e[sm]**2-be[sm]**2)),'group_definition':'threshold 0.5 diagnostic only for soft route' if choice['mode']=='soft' else 'actual hard-route confusion group'})
    comp=pd.DataFrame(comparisons);comp.to_csv(R/'selected_comparison.csv',index=False)
    pd.DataFrame(selected_cls).to_csv(R/'selected_classification_metrics.csv',index=False)
    pd.DataFrame(selected_costs).to_csv(R/'selected_error_costs.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    colors={'roll_1':'#697782','roll_2':'#3676A1','roll_3':'#B57935'}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    titles={'all':'All times','day_stable':'Daytime non-ramp','day_ramp':'Daytime ramp'}
    changes=[]
    for row,model in enumerate(['HGB','GRU']):
        for col,scope in enumerate(['all','day_stable','day_ramp']):
            ax=axes[row,col]
            for name,*_ in w.seq.FOLDS:
                p=weighted.query('split=="evaluation" and fold==@name and model==@model and scope==@scope').sort_values('alpha')
                change=100*(p.RMSE_MW/p[p.alpha==1].RMSE_MW.iloc[0]-1)
                ax.plot(p.alpha,change,marker=['o','s','^'][list(colors).index(name)],color=colors[name],label=name)
                for a,c in zip(p.alpha,change): changes.append({'model':model,'scope':scope,'fold':name,'alpha':int(a),'RMSE_change_pct':float(c)})
            ax.axhline(0,color='#333333',linewidth=.8)
            ax.set_xticks([1,2,4]);ax.set_xlabel('Ramp sample weight alpha');ax.set_ylabel('RMSE change vs alpha=1 (%)')
            ax.set_title(model+' - '+titles[scope]);ax.grid(axis='y',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,title='One-hour historical evaluation; lower is better; GRU = mean of 3 seed metrics')
    fig.savefig(w.OUT/'weighted_tradeoffs.png',dpi=160);plt.close(fig)
    pd.DataFrame(changes).to_csv(w.OUT/'weighted_chart_data.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    models=['baseline','expert','selected','oracle'];labels=['Baseline','Expert a=4','Selected route','Oracle*']
    palette=['#697782','#B57935','#3676A1','#C5CED5']
    for row,name in enumerate(['roll_2','roll_3']):
        for col,scope in enumerate(['all','day_stable','day_ramp']):
            part=routing.query('fold==@name and split=="evaluation" and scope==@scope').set_index('model')
            values=[part.loc[m,'RMSE_MW'] for m in models]
            ax=axes[row,col];ax.bar(range(4),values,color=palette,edgecolor='#333333',linewidth=.4)
            for k,v in enumerate(values): ax.text(k,v+.05,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
            ax.set_xticks(range(4),labels,rotation=18,ha='right');ax.set_ylabel('RMSE (MW)')
            ax.set_ylim(0,max(routing.query('split=="evaluation" and scope==@scope').RMSE_MW)*1.19)
            ax.set_title(name+' - '+titles[scope]+f' (n={int(part.loc["baseline","n"])})');ax.grid(axis='y',alpha=.18)
    fig.suptitle('Validation-selected one-hour routing on historical evaluation\n*Oracle uses future event labels: diagnostic only',fontsize=13)
    fig.savefig(R/'routing_comparison.png',dpi=160);plt.close(fig)
    # Export inference examples from origin-available features only.
    from predict_routing_v5 import predict
    _,tab,_,_,sets,*_=w.inputs()
    inference_checks=[]
    for choice in choices:
        name=choice['fold']
        saved=pd.read_csv(w.OUT/f'{name}_evaluation_predictions.csv')
        origins=pd.DatetimeIndex(pd.to_datetime(saved.origin_utc.iloc[:3],utc=True))
        features=tab.loc[origins,sets['local_surface']].copy()
        features.index.name='origin_utc'
        features.to_csv(w.OUT/f'{name}_demo_origin_features.csv')
        pred=predict(name,features)
        reference=pd.read_csv((R if choice['routing_enabled'] else w.OUT)/f'{name}_evaluation_predictions.csv')
        column='selected_seed42' if choice['routing_enabled'] else 'HGB_a1_seed42'
        diff=float(np.max(abs(pred.selected_MW.to_numpy()-reference[column].iloc[:3].to_numpy())))
        assert diff<1e-10
        pred.to_csv(w.OUT/f'{name}_demo_inference.csv')
        inference_checks.append({'fold':name,'rows':len(pred),'max_abs_diff_MW':diff,'future_target_columns_in_input':False})
    (w.OUT/'inference_checks.json').write_text(json.dumps(inference_checks,indent=2)+'\n')
    scripts=['run_improvements_v5.py','run_weighted_loss_v5.py','run_routing_v5.py','validate_improvements_v5.py','summarize_improvements_v5.py','predict_routing_v5.py']
    code_manifest={name:w.seq.sha(Path(__file__).parent/name) for name in scripts}
    (R/'pipeline_code_manifest.json').write_text(json.dumps(code_manifest,indent=2)+'\n')
    print(comp.query('split=="evaluation" and scope in ["all","day_stable","day_ramp"]').to_string(index=False))
    print(pd.DataFrame(selected_cls).to_string(index=False))

if __name__=='__main__': main()
