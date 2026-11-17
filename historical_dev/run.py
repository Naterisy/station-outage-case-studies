"""Development-only geographic baseline and matched graph covariance ablation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent

def predictions(cov, design, observed, hidden, values):
    """BLUP with optional unknown fixed drift. No hidden responses consumed."""
    v = cov[np.ix_(observed, observed)]
    cross = cov[np.ix_(hidden, observed)]
    if design is None:
        return cross @ solve(v, values, assume_a="pos")
    f = design[observed]
    weighted = solve(v, np.column_stack((values, f)), assume_a="pos")
    beta = solve(f.T @ weighted[:, 1:], f.T @ weighted[:, 0], assume_a="pos")
    return design[hidden] @ beta + cross @ (weighted[:, 0] - weighted[:, 1:] @ beta)

def run():
    data = HERE.parent / "realdata"
    with np.load(data / "dev/2021.npz", allow_pickle=False) as z:
        past, coords, ids = (z[k] for k in ("temperature_c", "coordinates", "station_ids"))
    with np.load(data / "dev/2022.npz", allow_pickle=False) as z:
        target, dates = z["temperature_c"], z["dates"]
        np.testing.assert_array_equal(ids, z["station_ids"])
    meta = json.loads((data / "manifest.json").read_text())
    elev = np.array([x["elevation_m"] for x in meta["stations"]])
    n = len(ids)
    xy = np.c_[coords[:, 0]*111.195, coords[:, 1]*111.195*np.cos(np.deg2rad(39.25))]
    d = cdist(xy, xy)
    vert = abs(elev[:, None] - elev)
    phase = np.arange(365) * 2*np.pi/365
    seasonal_design = np.column_stack((np.ones(365), np.sin(phase), np.cos(phase),
                                       np.sin(2*phase), np.cos(2*phase)))
    seasonal = np.empty_like(past)
    for j in range(n):
        ok = np.isfinite(past[:, j])
        seasonal[:, j] = seasonal_design @ np.linalg.lstsq(
            seasonal_design[ok], past[ok,j], rcond=None)[0]
    anom = target - seasonal
    cases = {kind: [] for kind in ("random", "blocked")}
    for day in range(len(anom)):
        avail = np.flatnonzero(np.isfinite(anom[day]))
        k = round(.3 * len(avail))
        rng = np.random.default_rng(608019 + day)
        for center in rng.choice(avail, 4, replace=False):
            for kind, hidden in (("random", rng.choice(avail,k,replace=False)),
                                 ("blocked", avail[np.argsort(d[center,avail],kind="stable")[:k]])):
                h = np.sort(hidden)
                cases[kind].append((day, np.setdiff1d(avail,h), h))
    I = np.eye(n)
    residual = past-seasonal
    complete = np.isfinite(residual).all(axis=1)
    assert complete.sum() >= 200
    empirical = np.cov(residual[complete], rowvar=False, ddof=1)
    sd = np.sqrt(np.diag(empirical))
    assert np.linalg.eigvalsh(empirical).min() > 0
    previous = json.loads((HERE.parent/'kriging_dev/selection.json').read_text())
    provenance = {'complete_training_days':int(complete.sum()),'training_only_covariance':True,
                  'empirical_min_eigenvalue':float(np.linalg.eigvalsh(empirical).min()),
                  'warning':'Complete-case covariance can be biased by missingness; only one training year.',
                  'data_access':['realdata/dev/2021.npz','realdata/dev/2022.npz','realdata/manifest.json','kriging_dev/selection.json'],
                  'test_access':'No 2023 files enumerated or opened'}
    np.savez_compressed(HERE/'training_covariance.npz',covariance=empirical,complete_days=np.flatnonzero(complete))
    drift = np.column_stack((np.ones(n),(elev-elev.mean())/elev.std()))
    all_rows = []
    chosen = {}
    for kind, masks in cases.items():
        models = {}
        for trend in (False,True):
            suffix = 'drift' if trend else 'fixed'
            h,v,old_nugget = previous[f'{kind}/spatial_elev_{suffix}']['parameters']
            geographic = np.exp(-np.sqrt((d/h)**2+(vert/v)**2))
            target_cov = geographic * np.outer(sd,sd)
            models['original_geographic_'+suffix] = [((h,v,old_nugget),geographic+old_nugget*I,trend)]
            for family, alphas in [('heterogeneous_geographic',[0]),('empirical',[1]),('shrinkage',[0,.25,.5,.75,1])]:
                grid=[]
                for alpha in alphas:
                    for nugget in (.001,.02,.1,.5):
                        cov=alpha*empirical+(1-alpha)*target_cov+nugget*np.diag(sd**2)
                        grid.append(((h,v,alpha,nugget),cov,trend))
                models[family+'_'+suffix]=grid
        tune = [case for case in masks if (case[0]//7)%2 == 0]
        diag = [case for case in masks if (case[0]//7)%2 != 0]
        def errs(cov, trend, subset):
            for day,o,h in subset:
                yield predictions(cov,drift if trend else None,o,h,anom[day,o]) - anom[day,h]
        for name, grid in models.items():
            scores = [np.abs(np.concatenate(list(errs(c,tr,tune)))).mean()
                      for _,c,tr in grid]
            best = int(np.argmin(scores))
            params,c,tr = grid[best]
            chosen[f"{kind}/{name}"] = dict(parameters=params,tune_mae=scores[best],grid=[dict(parameters=g[0],tune_mae=float(score)) for g,score in zip(grid,scores)])
            for (day,o,h),e in zip(diag, errs(c,tr,diag)):
                all_rows.extend((kind,name,day,int(day//7),str(ids[j]),float(ej),elev[j]>=1000)
                                for j,ej in zip(h,e))
    frame = pd.DataFrame(all_rows, columns=["mask","method","day","week","station","error","high"])
    frame["absolute"] = frame.error.abs()
    summary = frame.groupby(["mask","method"]).absolute.agg(
        n="size", mae="mean", p90=lambda x: x.quantile(.9))
    high = frame[frame.high].groupby(["mask","method"]).absolute.mean()
    summary["high_mae"] = high
    print(summary.to_string())
    week = frame.groupby(["mask","week","method"]).absolute.mean().unstack("method")
    for suffix in ['fixed','drift']:
        wins = week['shrinkage_'+suffix] < week['original_geographic_'+suffix]
        print('Weeks shrinkage improves '+suffix, wins.groupby('mask').agg(['sum','size']))
    frame.to_csv(HERE/'diagnostic_errors.csv',index=False)
    import hashlib,platform,scipy
    provenance.update(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__)
    provenance['input_sha256']={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [data/'dev/2021.npz',data/'dev/2022.npz',data/'manifest.json',HERE.parent/'kriging_dev/selection.json']}
    serialized={k:[(day,o.tolist(),h.tolist()) for day,o,h in cs] for k,cs in cases.items()}
    (HERE/'cases.json').write_text(json.dumps(serialized))
    (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2))
    summary.to_csv(HERE/"summary.csv")
    week.to_csv(HERE/"weekly_mae.csv")
    (HERE/"selection.json").write_text(json.dumps(chosen,indent=2)+"\n")

if __name__ == "__main__":
    run()
