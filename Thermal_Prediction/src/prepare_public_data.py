"""Read publisher archives, audit thermal signals and write a minimal candidate dataset.
No interpolation, modeling or data-driven target filtering is performed.
"""
from pathlib import Path
import csv, hashlib, json, zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/preparation'
OUT.mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/raw/kuleuven_bev'
archive=RAW/'BEV_energy_dynamic_data_V2.zip'
meta=json.loads((RAW/'record_metadata.json').read_text())
filemeta=next(f['dataFile'] for f in meta['data']['latestVersion']['files'] if f['dataFile']['id']==272593)
assert hashlib.md5(archive.read_bytes()).hexdigest()==filemeta['checksum']['value']
z=zipfile.ZipFile(archive)
# Preserve only reference documentation outside archive; do not execute publisher code.
refs=RAW/'reference';refs.mkdir(exist_ok=True)
for name in z.namelist():
    if name.endswith(('CAN.dbc','List of variables.csv','pylibs.txt')):
        (refs/Path(name).name).write_bytes(z.read(name))
files=sorted(n for n in z.namelist() if '/fast charging sessions/' in n and n.endswith('.csv'))
profiles=[];frames=[];sources=[]
for name in files:
    b=z.read(name)
    d=pd.read_csv(__import__('io').BytesIO(b))
    sid=Path(name).stem
    time=pd.to_datetime(d.Timestamp,utc=True,errors='coerce')
    mapping={'temperature_c':'BMSmaxPackTemperature','ambient_c':'VCFRONT_tempAmbient',
             'dc_voltage_v':'FC_dcVoltage','dc_current_a':'FC_dcCurrent',
             'battery_voltage_v':'BattVoltage132','battery_current_a':'RawBattCurrent132',
             'soc_percent':'SOCave292','bms_dissipation_kw':'BMSdissipation312',
             'preconditioning_request':'UI_batteryPreconditioningRequest'}
    f=pd.DataFrame({'session_id':sid,'device_id':sid.split('_')[0],'timestamp_utc':time})
    for k,v in mapping.items():f[k]=pd.to_numeric(d[v],errors='coerce') if v in d else np.nan
    f['time_s']=(time-time.iloc[0]).dt.total_seconds()
    f['dc_power_w']=f.dc_voltage_v*f.dc_current_a
    dt=time.diff().dt.total_seconds().dropna()
    p={'session_id':sid,'device_id':sid.split('_')[0],'rows':len(d),'start_utc':str(time.min()),
       'end_utc':str(time.max()),'duration_s':float(f.time_s.max()),'timestamp_nulls':int(time.isna().sum()),
       'duplicate_timestamps':int(time[time.notna()].duplicated().sum()),'nonpositive_dt':int((dt<=0).sum()),
       'median_dt_s':float(dt.median()),'max_dt_s':float(dt.max()),'gaps_over_2s':int((dt>2).sum()),
       'dc_power_min_kw':float(f.dc_power_w.min()/1000),'dc_power_max_kw':float(f.dc_power_w.max()/1000),
       'dc_power_median_kw':float(f.dc_power_w.median()/1000),'temperature_min_c':float(f.temperature_c.min()),
       'temperature_max_c':float(f.temperature_c.max()),'temperature_distinct':int(f.temperature_c.nunique()),
       'ambient_min_c':float(f.ambient_c.min()),'ambient_max_c':float(f.ambient_c.max()),
       'soc_start':float(f.soc_percent.iloc[0]),'soc_end':float(f.soc_percent.iloc[-1])}
    for c in ['temperature_c','ambient_c','dc_voltage_v','dc_current_a','soc_percent','bms_dissipation_kw']:
        p[c+'_nonfinite']=int((~np.isfinite(f[c])).sum())
    p['negative_dc_power_rows']=int((f.dc_power_w<0).sum())
    p['missing_required']=sum(p[c+'_nonfinite'] for c in ['temperature_c','ambient_c','dc_voltage_v','dc_current_a'])
    p['temperature_outside_dbc_range']=int(((f.temperature_c < -25)|(f.temperature_c>100)).sum())
    profiles.append(p);frames.append(f)
    sources.append({'member':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'zip_crc_verified':True})
profile=pd.DataFrame(profiles)
profile.to_csv(OUT/'bev_fast_session_quality.csv',index=False)
allf=pd.concat(frames,ignore_index=True)
allf.to_csv(ROOT/'data/processed/bev_fast_candidate.csv',index=False)
# Remove invalid export rows only; do not fill target temperature.
required=['timestamp_utc','temperature_c','ambient_c','dc_voltage_v','dc_current_a']
mask=allf[required].notna().all(axis=1)
for c in required[1:]: mask &= np.isfinite(allf[c])
clean=allf.loc[mask].copy()
if clean.duplicated(['session_id','timestamp_utc']).any():
    raise RuntimeError('Conflicting valid timestamps require explicit resolution')
clean=clean.sort_values(['session_id','timestamp_utc'])
clean['segment_index']=clean.groupby('session_id')['timestamp_utc'].transform(lambda x: x.diff().dt.total_seconds().ne(1).cumsum())
clean['segment_id']=clean.session_id+'__segment_'+clean.segment_index.astype(str)
clean.to_csv(ROOT/'data/processed/bev_fast_clean.csv',index=False)
segments=clean.groupby(['session_id','segment_id']).agg(rows=('temperature_c','size'),start=('timestamp_utc','min'),end=('timestamp_utc','max'))
segments.to_csv(OUT/'continuous_segments.csv')
clean_checks={'valid_rows':len(clean),'removed_invalid_rows':len(allf)-len(clean),
              'valid_sessions':int(clean.session_id.nunique()),'valid_duplicate_timestamps':int(clean.duplicated(['session_id','timestamp_utc']).sum()),
              'target_interpolation':False,'continuous_segments':len(segments),
              'potential_60s_history_60s_target_windows_at_1s':int((segments.rows-120).clip(lower=0).sum()),
              'sessions_with_potential_windows':int(segments[segments.rows>120].reset_index().session_id.nunique())}
(OUT/'cleaning_checks.json').write_text(json.dumps(clean_checks,indent=2))

# Preliminary calendar-group split; no model/temperature performance consulted.
# Same vehicle and UTC date remain in the same split. Do not use until protocol reviewed.
splits=[]
for dev,g in profile.groupby('device_id'):
    dates=sorted({x[:10] for x in g.start_utc})
    n=len(dates); nt=max(1,int(n*0.6));nv=max(1,int(n*0.2))
    for row in g.to_dict('records'):
        ix=dates.index(row['start_utc'][:10]);split='train' if ix<nt else 'validation' if ix<nt+nv else 'test'
        splits.append({'session_id':row['session_id'],'device_id':dev,'date_utc':row['start_utc'][:10],'split':split})
pd.DataFrame(splits).to_csv(ROOT/'configs/provisional_split.csv',index=False)
joined=clean.merge(pd.DataFrame(splits)[['session_id','split']],validate='many_to_one')
training=joined[(joined.split=='train') & (joined.dc_power_w>1000)]
a,b=training.dc_power_w.quantile([1/3,2/3])
joined['power_band']=pd.cut(joined.dc_power_w,bins=[float('-inf'),1000,a,b,float('inf')],labels=['idle_or_low_flow','low','mid','high'])
joined.groupby(['split','power_band'],observed=True).agg(rows=('session_id','size'),sessions=('session_id','nunique')).reset_index().to_csv(OUT/'power_coverage.csv',index=False)
protocol={'status':'candidate_protocol_not_trained','data_route':'public_bev_pack_temperature','target':'BMSmaxPackTemperature',
          'horizon_s':60,'history_s':60,'future_power':'origin_hold_no_future_measured_power','ambient':'origin_hold',
          'power_bands_w':{'idle_upper':1000,'low_upper':float(a),'mid_upper':float(b)},
          'power_bands_fit_on':'positive training power > 1000 W','models':['temperature_persistence','effective_RC','RC_plus_small_MLP'],
          'scope_review':'pack temperature substitutes original module candidate; clarify in final report',
          'split':'provisional_split.csv'}
(ROOT/'configs/candidate_protocol.json').write_text(json.dumps(protocol,indent=2))

# Examine every downloaded TU Dortmund candidate header. Header declarations alone are insufficient.
tud=[]
for p in sorted((ROOT/'data/raw/tudortmund').glob('selected*/**/*.csv')):
    with p.open() as h: cols=next(csv.reader(h,delimiter=';'))
    targets=[c for c in cols if 'temperature ccs' in c.lower()]
    row={'file':str(p.relative_to(ROOT)),'target_columns':targets,'has_declared_target':bool(targets)}
    if targets:
        d=pd.read_csv(p,sep=';',skiprows=[1],low_memory=False)
        row['connector_current_abs_max_a']={c:float(pd.to_numeric(d[c],errors='coerce').abs().max()) for c in ['Current CCS','Current CHAdeMO'] if c in d}
        row['target_profile']={c:{'nonnull':int(pd.to_numeric(d[c],errors='coerce').notna().sum()),'unique':int(d[c].nunique()),'min':str(d[c].min()),'max':str(d[c].max())} for c in targets}
    tud.append(row)
(OUT/'tudortmund_target_audit.json').write_text(json.dumps(tud,indent=2))
summary={'status':'preparation_only_not_trained','source':'https://doi.org/10.48804/8KPDTW','license':'CC-BY-4.0',
         'archive_md5_verified':True,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
         'cleaning':clean_checks,'fast_sessions':len(profile),'devices':int(profile.device_id.nunique()),'rows':len(allf),
         'temperature_range_c':[float(allf.temperature_c.min()),float(allf.temperature_c.max())],
         'dc_power_range_kw':[float(allf.dc_power_w.min()/1000),float(allf.dc_power_w.max()/1000)],
         'ambient_range_c':[float(allf.ambient_c.min()),float(allf.ambient_c.max())],
         'required_nonfinite_entries':int(profile.missing_required.sum()),'duplicate_timestamps':int(profile.duplicate_timestamps.sum()),
         'gaps_over_2s':int(profile.gaps_over_2s.sum()),'negative_dc_power_rows':int(profile.negative_dc_power_rows.sum()),
         'provisional_split_counts':dict(pd.Series([r['split'] for r in splits]).value_counts().astype(int).items()),
         'tud_downloaded_members':len(tud),'tud_with_target_columns':sum(r['has_declared_target'] for r in tud),
         'slow_session_files':sum('/slow charging sessions/' in n and n.endswith('.csv') for n in z.namelist()),
         'scope':'BMS maximum battery pack temperature, not module/heatsink temperature',
         'limitations':['Cooling states not yet established','BMS dissipation signal semantics and active cooling need further review',
          'Low/mid/high bins not frozen; later based on training data only','No future real DC power permitted in online inference',
          'Processed publisher data; CAN decoding and interpolation not independently reproduced'], 'files':sources}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='files'},ensure_ascii=False,indent=2))
