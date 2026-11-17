# Historical analog versus empirical covariance: no substantial incremental gain

Private development experiment only. Original historical_dev/kriging_dev untouched. No 2023 data, filenames, directory inventory, or outputs opened. Reproduce: `OPENBLAS_NUM_THREADS=1 python run.py && OPENBLAS_NUM_THREADS=1 python -m unittest -v test_run.py && python analyze.py`.

## Design
Exact frozen masks and alternating tuning/diagnostic weeks loaded from historical_dev/cases.json. The 2021-only site harmonic seasonal means and empirical covariance are reconstructed. Candidate analogs are exactly the same 331 complete 2021 anomaly days used for covariance; no 2022 candidate days. For each query/mask, rank references by mean squared anomaly discrepancy across **the same observed station coordinates only**, standardized by each station's 2021 residual standard deviation. Take uniform average of k in {1,3,5,10,20,40,80} nearest days. Seasonal mean is common to all methods.

Methods: simple analog hidden anomalies; analog plus scalar mean observed mismatch; analog plus empirical conditional correction; and partial blend of corrected analog with empirical baseline. If B is the empirical conditional-prediction matrix and a is analog mean, corrected analog = a_H+B(y_O-a_O). Blend = By_O+gamma(a_H-Ba_O), gamma in {.25,.5}. Empirical baseline is retained separately (gamma0), full correction separately (gamma1). All k/gamma choices use 2022 tuning weeks only, separately per mask; empirical nugget .02 inherited from existing same-split tuning. Ranking never sees hidden query responses. Reference hidden coordinates are legitimate 2021 training data, not hidden 2022 truth.

## Diagnostic MAE, degrees Celsius
| Method | Blocked | Blocked >=1000m | Random | Random >=1000m |
|---|---:|---:|---:|---:|
| Empirical covariance | 0.996463 | 1.340299 | 0.826940 | 1.046453 |
| Simple analog | 1.392342 | 1.925587 | 1.225697 | 1.612970 |
| Analog plus offset | 1.299204 | 1.787656 | 1.134506 | 1.478472 |
| Analog plus covariance correction | 1.007505 | 1.375779 | 0.826803 | 1.043865 |
| Partial covariance blend | 0.993220 | 1.343684 | 0.823142 | 1.041747 |

Same7,904 repeated predictions per regime/method. The blend selects k5/gamma.25 for both masks. Uncorrected and offset analogs select k5; full covariance correction selects k20 blocked and k80 random (grid boundary). Full correction worsens blocked MAE and high-elevation error, and is essentially tied on random masks. The blend's tiny aggregate reduction (~0.003–0.004C) is not a substantial practical gain; blocked high-elevation error worsens by0.0034C.

Blend improves16/26 diagnostic weeks in each mask regime. Two-consecutive-diagnostic-week circular block bootstrap (5000 draws) has paired mean-week MAE difference intervals blocked [-0.00845,+0.00240]C and random [-0.00729,-0.00014]C. Exploratory conditional summaries only, not an independent significance claim; development year repeatedly inspected, masks/weather correlated, one seed/training year. Blocked blend loses in autumn, improves other seasons modestly; random blend slightly loses in spring. Largest seasonal mean gain is ~0.009C in winter. There is no persuasive justification here for replacing the empirical covariance baseline or presenting analog correction as a new method.

## Prior art and limits
Official WMO proceedings source was accessed and confirms Gregor Vertačnik, “A method for daily temperature data interpolation and quality control based on the selected past events,” proceedings printed pp53–59, 2008 seminar, WCDMP-No.76 / WMO-TD1576: https://mtb.met.hu/downloads.php?fn=/metadmin/attach/2013/02/homogenization-6th-2008.pdf . This is an independent comparator implementation, NOT a faithful reproduction of that method. Analog selection is old; no novelty claim.

Only331 complete training days limits analog diversity and biases references toward complete-record days. No seasonal restriction or metric search beyond station-standardized squared discrepancy; negative results apply to these tested baselines, not every analog approach. Physical daily mean observations are references, not noise-free truth. Historical fixed-station methods need additional machinery for genuinely unseen stations.

Four tests pass: ranking invariant to changing hidden reference coordinates; correction identity; empirical comparator reproduces historical summary/counts to numerical precision; reference indices exactly match complete 2021 days. All tuning scores, input hashes, versions, diagnostic errors, season/weekly summaries and code checksums retained. Recommended stop on this analog branch unless another clearly motivated independent failure setting warrants testing; do not keep mining the same 2022 diagnostics for a tiny win.
