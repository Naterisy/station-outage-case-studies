# Cold-station falsification: NO-GO for this graph-resolvent candidate

Predeclared in PROTOCOL.md before evaluation: require >=5% pooled MAE improvement, improvements in >=4/5 station folds, and no high-elevation degradation. The graph fails all three conditions. This is a negative result for this concrete candidate/grid, not a statement about every graph method.

| 2022 outer diagnostic metric | Geographic/elevation exponential | Graph resolvent | Shared drift-only |
|---|---:|---:|---:|
| MAE, C | **1.6551** | 1.8845 | 2.2277 |
| >=1000m MAE, C | **2.0996** | 2.3718 | 2.7509 |
| p90 absolute error, C | **3.5644** | 4.0637 | 4.9978 |

6,519 target station-day references per method, not independent observations. Graph is about13.9% worse in pooled MAE and loses overall MAE in **all5 outer folds**. Geographic/graph fold MAEs: f0 1.990/2.170; f1 1.340/1.800; f2 1.415/1.768; f3 1.840/1.856; f4 1.644/1.789. Graph's high-elevation MAE is slightly better in fold3 only (2.824 versus2.880), insufficient to offset degradation elsewhere.

## Why this is cold-station fitting
Five elevation-stratified folds use static metadata only, with two >=1000m sites in every fold. Each outer station's entire2021 and2022 temperature history is excluded from all fitting/selection. In each inner fold, climatology is refit excluding both outer and inner held-out stations; inner2022 even-week references select covariance parameters. Only after this selection is climatology refit using all non-outer stations'2021 data. Outer odd-week2022 temperatures are used once for reporting errors, never to choose parameters. All non-target same-day observations are available at inference. Target static location/elevation is allowed.

Both priors use identical regression climatology, same observed stations, same-day drift covariates and GLS prediction formula. Drift coefficients depend on covariance, as standard universal kriging requires; this is matched trend structure, not forcibly identical coefficients. Each method has18 parameter candidates, tuned separately per outer fold. Graph inner nodes exclude outer stations; final graph incorporates static outer target nodes, a legitimate new-location prediction input. Inner graph geometry therefore differs from final geometry; this is part of the chosen inductive workflow, not evidence for a universally optimal graph construction.

Important boundary: the existing36-station benchmark was retrospectively selected using2021 coverage. The experiment prohibits target **temperature values** in fitting/tuning, but does not erase that earlier availability-based panel selection. It demonstrates cold-station prediction within an existing panel, not a prospectively collected never-operated sensor experiment. Only one reused development year and one deterministic fold assignment; covariance grid optima can lie on boundaries. No claims of significance, novelty or publication readiness.2023 was neither enumerated nor opened.

## Reproduction and verification
`OPENBLAS_NUM_THREADS=1 python run.py && OPENBLAS_NUM_THREADS=1 python -m unittest -v test_run.py`

Four tests pass: held-out training-temperature poison invariance; inner prediction inputs exclude outer temperatures; covariance positive definiteness; unique stratified fold membership with two high-elevation sites each. Inputs/checksums, package versions, fold membership, full nested grid scores, raw errors and fold summaries are retained. Initial array-broadcast setup error was fixed before any evaluation; its log is retained as initial_shape_error.log.

Recommendation: stop this graph-resolvent candidate. The concrete obstacle is not merely lack of target history: the conventional location/elevation covariance still performs better under strict station exclusion. Do not reopen sealed2023 or expand variants just to rescue the premise.
