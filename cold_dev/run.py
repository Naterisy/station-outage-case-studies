from pathlib import Path
import json,hashlib,platform
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent

def fit_season(past,train,spatial):
    phase=np.arange(365)*2*np.pi/365
    time=np.c_[np.ones(365),np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)]
    design=(time[:,None,:,None]*spatial[None,:,None,:]).reshape(365,len(spatial),-1)
    # Restrict station dimension before extracting finite measurements or solving.
    y=past[:,train];X=design[:,train,:];ok=np.isfinite(y)
    beta=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0]
    return design@beta

def predict(cov,drift,o,h,values):
    v=cov[np.ix_(o,o)];cross=cov[np.ix_(h,o)];f=drift[o]
    weighted=solve(v,np.column_stack([values,f]),assume_a='pos')
    beta=solve(f.T@weighted[:,1:],f.T@weighted[:,0],assume_a='pos')
    return drift[h]@beta+cross@(weighted[:,0]-weighted[:,1:]@beta)

def models(coords,elev):
    xy=np.c_[coords[:,0]*111.195,coords[:,1]*111.195*np.cos(np.deg2rad(39.25))];d=cdist(xy,xy);v=abs(elev[:,None]-elev);I=np.eye(len(elev));grids={'geo':[],'graph':[],'drift':[((0,),I)]}
    for h in [90,220,500]:
      for height in [750,2000]:
        scaled=np.sqrt((d/h)**2+(v/height)**2)
        for noise in [.001,.02,.1]:grids['geo'].append(((h,height,noise),np.exp(-scaled)+noise*I))
        w=np.zeros_like(d);js=np.argsort(d,axis=1)[:,1:7];row=np.arange(len(d))[:,None];w[row,js]=np.exp(-scaled[row,js]);w=np.maximum(w,w.T);lap=np.diag(w.sum(1))-w
        for strength in [1,10,100]:grids['graph'].append(((h,height,strength),solve(I+strength*lap,I,assume_a='pos')+.01*I))
    return grids

def run():
    data=HERE.parent/'realdata'
    with np.load(data/'dev/2021.npz') as z: past,ids,coords=z['temperature_c'],z['station_ids'],z['coordinates']
    with np.load(data/'dev/2022.npz') as z: target=z['temperature_c'];np.testing.assert_array_equal(ids,z['station_ids'])
    meta=json.loads((data/'manifest.json').read_text());elev=np.array([s['elevation_m'] for s in meta['stations']]);n=len(ids)
    folds=np.empty(n,dtype=int)
    for high in [False,True]:
        ix=np.flatnonzero((elev>=1000)==high);ix=ix[np.argsort(ids[ix])]
        for k,j in enumerate(ix):folds[j]=k%5
    assert all(((folds==f)&(elev>=1000)).sum()==2 for f in range(5))
    (HERE/'folds.json').write_text(json.dumps([{'station':str(ids[j]),'fold':int(folds[j]),'high':bool(elev[j]>=1000)} for j in range(n)],indent=2))
    raw=np.c_[coords,elev];spatial=np.c_[np.ones(n),(raw-raw.mean(0))/raw.std(0)];drift=np.c_[np.ones(n),(elev-elev.mean())/elev.std()]
    tune_days=[d for d in range(365) if d//7%2==0];diag_days=[d for d in range(365) if d//7%2==1]
    rows=[];selected={}
    for outer in range(5):
        available=np.flatnonzero(folds!=outer);held=np.flatnonzero(folds==outer)
        grids_inner=models(coords[available],elev[available]);scores={name:np.zeros(len(grid)) for name,grid in grids_inner.items()};counts=0
        for inner in range(5):
            if inner==outer:continue
            train=np.flatnonzero((folds!=outer)&(folds!=inner));valid=np.flatnonzero(folds==inner)
            season=fit_season(past,train,spatial)
            train_local=np.flatnonzero(folds[available]!=inner);valid_local=np.flatnonzero(folds[available]==inner)
            for day in tune_days:
                o=train_local[np.isfinite(target[day,available[train_local]])];h=valid_local[np.isfinite(target[day,available[valid_local]])]
                values=target[day,available[o]]-season[day,available[o]];truth=target[day,available[h]]-season[day,available[h]];counts+=len(h)
                for name,grid in grids_inner.items():
                    for k,(_,cov) in enumerate(grid):scores[name][k]+=np.abs(predict(cov,drift[available],o,h,values)-truth).sum()
        best={name:int(np.argmin(score)) for name,score in scores.items()}
        selected[str(outer)]={name:{'parameters':grids_inner[name][best[name]][0],'inner_tune_mae':float(scores[name][best[name]]/counts),'n':counts,'grid_mae':(scores[name]/counts).tolist()} for name in best}
        final=models(coords,elev);season=fit_season(past,available,spatial)
        for day in diag_days:
            o=available[np.isfinite(target[day,available])];h=held[np.isfinite(target[day,held])];values=target[day,o]-season[day,o];truth=target[day,h]-season[day,h]
            for name in best:
                cov=final[name][best[name]][1];errors=predict(cov,drift,o,h,values)-truth
                for j,e in zip(h,errors):rows.append((outer,name,day,str(ids[j]),float(e),bool(elev[j]>=1000)))
        print('completed outer fold',outer,flush=True)
    frame=pd.DataFrame(rows,columns=['fold','method','day','station','error','high']);frame['absolute']=frame.error.abs();frame.to_csv(HERE/'errors.csv',index=False)
    for keys,file in [(['method'],'summary.csv'),(['fold','method'],'fold_summary.csv')]:
        s=frame.groupby(keys).absolute.agg(n='size',mae='mean',p90=lambda x:x.quantile(.9));s['high_mae']=frame[frame.high].groupby(keys).absolute.mean();s.to_csv(HERE/file);print(s.to_string())
    (HERE/'selection.json').write_text(json.dumps(selected,indent=2))
    prov={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'test_access':'No 2023 enumerated or opened','inputs':{str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'dev/2021.npz',data/'dev/2022.npz',data/'manifest.json']}}
    (HERE/'provenance.json').write_text(json.dumps(prov,indent=2))
if __name__=='__main__':run()
