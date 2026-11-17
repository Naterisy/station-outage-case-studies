from pathlib import Path
import hashlib,json,platform
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
HERE=Path(__file__).resolve().parent
KS=[1,3,5,10,20,40,80]
def analogs(reference, values, observed, hidden, sd, B):
    # Ranking sees only observed coordinates; reference days are training-only.
    discrepancy=values[None,:]-reference[:,observed]
    order=np.argsort(np.mean((discrepancy/sd[observed])**2,axis=1),kind='stable')
    result={}
    for k in KS:
        a=reference[order[:k]].mean(axis=0)
        simple=a[hidden]
        offset=simple+np.mean(values-a[observed])
        corrected=simple+B@(values-a[observed])
        result[k]=(simple,offset,corrected)
    return result,order

def run():
    data=HERE.parent/'realdata';hist=HERE.parent/'historical_dev'
    with np.load(data/'dev/2021.npz') as z: past,ids=z['temperature_c'],z['station_ids']
    with np.load(data/'dev/2022.npz') as z: target,dates=z['temperature_c'],z['dates'];np.testing.assert_array_equal(ids,z['station_ids'])
    meta=json.loads((data/'manifest.json').read_text());elev=np.array([s['elevation_m'] for s in meta['stations']])
    t=np.arange(365)*2*np.pi/365;X=np.c_[np.ones(365),np.sin(t),np.cos(t),np.sin(2*t),np.cos(2*t)]
    season=np.empty_like(past)
    for j in range(len(ids)):
        ok=np.isfinite(past[:,j]);season[:,j]=X@np.linalg.lstsq(X[ok],past[ok,j],rcond=None)[0]
    resid=past-season;complete=np.isfinite(resid).all(1);reference=resid[complete];anom=target-season
    S=np.cov(reference,rowvar=False);sd=np.sqrt(np.diag(S))
    cases=json.loads((hist/'cases.json').read_text());previous=json.loads((hist/'selection.json').read_text())
    chosen={};rows=[]
    for kind,cs in cases.items():
        nugget=previous[kind+'/empirical_fixed']['parameters'][-1];cov=S+nugget*np.diag(sd**2)
        predictions={('empirical',0,0):[]}
        for family in ['analog','analog_offset','analog_covcorrected']:
            for k in KS: predictions[(family,k,1)]=[]
        for k in KS:
            for gamma in [.25,.5]:predictions[('covariance_blend',k,gamma)]=[]
        for day,o,h in cs:
            o=np.array(o);h=np.array(h);values=anom[day,o]
            B=solve(cov[np.ix_(o,o)],cov[np.ix_(o,h)],assume_a='pos').T
            baseline=B@values;predictions[('empirical',0,0)].append(baseline)
            aps,_=analogs(reference,values,o,h,sd,B)
            for k,(simple,offset,corrected) in aps.items():
                for fam,pred in [('analog',simple),('analog_offset',offset),('analog_covcorrected',corrected)]:predictions[(fam,k,1)].append(pred)
                for gamma in [.25,.5]:predictions[('covariance_blend',k,gamma)].append(baseline+gamma*(corrected-baseline))
        tune=[i for i,(d,_,_) in enumerate(cs) if d//7%2==0];diag=[i for i,(d,_,_) in enumerate(cs) if d//7%2==1]
        scores={key:float(np.mean(np.abs(np.concatenate([p[i]-anom[cs[i][0],cs[i][2]] for i in tune])))) for key,p in predictions.items()}
        for family in ['empirical','analog','analog_offset','analog_covcorrected','covariance_blend']:
            grid=[key for key in predictions if key[0]==family];best=min(grid,key=lambda key:scores[key])
            chosen[kind+'/'+family]={'k':best[1],'gamma':best[2],'tune_mae':scores[best],'grid':[{'k':key[1],'gamma':key[2],'tune_mae':scores[key]} for key in grid]}
            for i in diag:
                day,o,h=cs[i];err=predictions[best][i]-anom[day,h]
                for j,e in zip(h,err):
                    month=pd.Timestamp(dates[day]).month
                    rows.append((kind,family,day,day//7,str(ids[j]),float(e),bool(elev[j]>=1000),['winter','spring','summer','autumn'][(month%12)//3]))
    frame=pd.DataFrame(rows,columns=['mask','method','day','week','station','error','high','season']);frame['absolute']=frame.error.abs()
    summary=frame.groupby(['mask','method']).absolute.agg(n='size',mae='mean',p90=lambda x:x.quantile(.9));summary['high_mae']=frame[frame.high].groupby(['mask','method']).absolute.mean()
    summary.to_csv(HERE/'summary.csv');frame.to_csv(HERE/'diagnostic_errors.csv',index=False)
    frame.groupby(['mask','season','method']).absolute.mean().to_csv(HERE/'season.csv')
    frame.groupby(['mask','week','method']).absolute.mean().unstack('method').to_csv(HERE/'weekly_mae.csv')
    (HERE/'selection.json').write_text(json.dumps(chosen,indent=2));np.save(HERE/'reference_day_indices.npy',np.flatnonzero(complete))
    inputs=[data/'dev/2021.npz',data/'dev/2022.npz',data/'manifest.json',hist/'cases.json',hist/'selection.json']
    prov={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'reference_year':2021,'reference_days':len(reference),'test_access':'No 2023 opened or enumerated','inputs':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (HERE/'provenance.json').write_text(json.dumps(prov,indent=2));print(summary.to_string())
if __name__=='__main__':run()
