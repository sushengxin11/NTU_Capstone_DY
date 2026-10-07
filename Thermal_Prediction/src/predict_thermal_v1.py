"""History-only inference using saved thermal_v1 models."""
from pathlib import Path
import argparse, joblib, pandas as pd
from run_thermal_v1 import predict_history, HORIZONS, ROOT
parser=argparse.ArgumentParser()
parser.add_argument('--history',type=Path,required=True,help='61 consecutive 1-second historical records')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
h=pd.read_csv(args.history,parse_dates=['timestamp_utc'])
package=joblib.load(ROOT/'models/thermal_v1/thermal_models.joblib')
forecast=predict_history(h,package)
args.output.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame({'horizon_s':HORIZONS,**forecast}).to_csv(args.output,index=False)
print('Saved 10–60 second temperature forecasts:',args.output)
