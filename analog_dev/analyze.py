from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parent;w=pd.read_csv(R/'weekly_mae.csv');rng=np.random.default_rng(608019);out={}
for kind,z in w.groupby('mask'):
 delta=(z.covariance_blend-z.empirical).to_numpy();draws=[]
 for _ in range(5000):
  starts=rng.integers(0,len(delta),13);ix=np.concatenate([(s+np.arange(2))%len(delta) for s in starts]);draws.append(delta[ix].mean())
 out[kind]={'mean_week_mae_delta':float(delta.mean()),'better_weeks':int((delta<0).sum()),'weeks':len(delta),'exploratory_circular_two_week_block_interval':np.quantile(draws,[.025,.975]).tolist()}
(R/'paired_week_diagnostics.json').write_text(json.dumps(out,indent=2));print(out)
