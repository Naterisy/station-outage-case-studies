# Historical covariance diagnostic: positive comparator, negative shrinkage result

All measurements below are development diagnostics, not independent test results. No 2023 file or output was enumerated or opened. `kriging_dev` is unchanged.

Reproduce here: `OPENBLAS_NUM_THREADS=1 python run.py && OPENBLAS_NUM_THREADS=1 python -m unittest -v test_run.py`.

## Construction and matching
Reuse the exact `kriging_dev` 365-day, four-mask-per-kind/date deterministic generator and alternating calendar-week tuning/diagnostic split. Fit each station's intercept and annual/semiannual harmonic mean on 2021 only. Form 36x36 empirical covariance S from **331 complete 2021 anomaly days**, using sample covariance with ddof=1; minimum eigenvalue 0.07547. Complete-case filtering preserves positive semidefiniteness but can bias the population represented by missingness.

Let D be training standard deviations; G is the established exponential geographic/elevation correlation at each prior baseline's tuning-selected length scales. Compare the unchanged original baseline, heteroskedastic geographic DGD, empirical S, and C = alpha S + (1-alpha) DGD + nugget D². Tune alpha in {0,.25,.5,.75,1} and nugget in {.001,.02,.1,.5} on the same 2022 tuning weeks, separately per mask and fixed-mean/daily-elevation-drift formulation. Historical covariance and D use **2021 only**, but hyperparameter selection uses 2022 tuning weeks. No 2022 response enters S. This is a standard historical kriging/covariance comparator, not a new method. Same-day observed anomalies are the only response inputs to conditional predictions.

The fixed geographic lengths are inherited, not reoptimized jointly with covariance shrinkage. Therefore failure of this shrinkage grid does not rule out every geographic hybrid. Original baselines reproduce previous summary, high-elevation, p90 and prediction counts to numerical precision (unit test).

## Diagnostic MAE (°C)
| Model | Blocked overall | Blocked >=1000m | Random overall | Random >=1000m |
|---|---:|---:|---:|---:|
| Original geographic, fixed mean | 1.340 | 2.029 | 1.186 | 1.658 |
| Heteroskedastic geographic, fixed | 1.329 | 1.973 | 1.148 | 1.570 |
| Historical empirical, fixed | **0.996** | **1.340** | **0.827** | **1.046** |
| Original geographic, daily drift | 1.352 | 2.043 | 1.178 | 1.604 |
| Historical empirical, daily drift | 1.023 | 1.373 | 0.845 | 1.067 |

7,904 repeated site/day/mask predictions per regime/method, not independent observations. Fixed empirical p90 absolute error is 2.190C blocked and 1.754C random. Empirical outperforms its original geographic comparator on each of the26 diagnostic weeks for both regimes and both drift variants; these counts are descriptive, not significance tests.

**Negative result:** every shrinkage selection lands at alpha=1 and nugget=.02. Thus the proposed geographic shrinkage contributes nothing at the selected settings; shrinkage predictions equal pure historical covariance. Free daily drift is also worse than fixed mean for the historical model here. Improvement is not a graph-regularizer novelty claim: it shows that the earlier geometry-only comparator omitted useful ordinary interstation history.

## Tests and limits
Four tests verify direct conditional prediction, exact 2021-only empirical covariance reconstruction/positive definiteness, unchanged original baseline numerical results/counts, and mask/split disjointness. Input hashes, versions and exact masks are stored; training covariance is inspectable. `selection.json` retains all tuning scores.

Only one training year and one repeatedly explored development year; same fixed stations, no new-station generalization. Random/blocked masks and weather are correlated; no inferential p-values. Targets are observed daily means, not noise-free signals or daily maximum heat. Historical station covariance cannot directly predict an unseen station without an additional mapping model. More complete-case and seasonal-covariance checks may help, but do not turn this straightforward baseline win into an invention. Highest-value next gate is training-history length/stability or held-out-station stress testing before any new local-contrast method or sealed-test access.
