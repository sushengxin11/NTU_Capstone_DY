"""Fixed first test session demo, selected by identifier before performance comparison."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/thermal_v1'
p=pd.read_csv(R/'predictions.csv',parse_dates=['origin_utc','target_utc'])
sid=sorted(p[p.split=='test'].session_id.unique())[0]
g=p[(p.session_id==sid)&(p.split=='test')]
d=pd.read_csv(ROOT/'data/processed/bev_fast_clean.csv',parse_dates=['timestamp_utc'])
d=d[d.session_id==sid]
t0=d.timestamp_utc.min();x=(d.timestamp_utc-t0).dt.total_seconds()/60
fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
axes[0].plot(x,d.dc_power_w/1000,color='#2166ac');axes[0].set_ylabel('DC input (kW)');axes[0].set_title('Fixed demo: '+sid)
axes[1].plot(x,d.temperature_c,color='black',label='Observed BMS max temperature')
colors={'persistence':'#777777','effective_RC':'#2166ac','RC_plus_MLP':'#b35806'}
for name in colors:
 for k,segment in enumerate(g.groupby('segment_id',sort=True)):
  _,v=segment
  axes[1].plot((v.target_utc-t0).dt.total_seconds()/60,v[f'{name}_60s_c'],color=colors[name],label=name if k==0 else None,linestyle='--' if name=='persistence' else '-')
axes[1].set_ylabel('Temperature (°C)');axes[1].legend(fontsize=8,ncol=2);axes[1].set_title('Rolling 60-second forecast, reset only at each forecast origin',fontsize=10)
h=pd.read_csv(R/'demo_predictions.csv');first=g.iloc[0]
axes[2].plot([0]+h.horizon_s.tolist(),[first.origin_temperature_c]+[first[f'true_{n}s_c'] for n in h.horizon_s],color='black',marker='o',label='Observed')
for name in colors:axes[2].plot([0]+h.horizon_s.tolist(),[first.origin_temperature_c]+h[name].tolist(),color=colors[name],marker='.',label=name)
axes[2].set_ylabel('Temperature (°C)');axes[2].set_xlabel('Seconds after fixed forecast origin');axes[2].set_title('Single 60-second trajectory: no future measured input',fontsize=10)
axes[0].set_xlabel('Minutes from session start');axes[1].set_xlabel('Minutes from session start')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig(R/'fixed_demo.png',dpi=180)
