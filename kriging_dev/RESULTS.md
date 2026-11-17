# Development results only

`OPENBLAS_NUM_THREADS=1 python run.py` with only 2021/2022 data. Unit
verification: `python -m unittest -v test_run.py` (three tests). Parameters
and diagnostic counts are machine-readable in the adjacent outputs.

| 2022 diagnostic MAE (°C) | Blocked | Random |
| --- | ---: | ---: |
| Daily elevation drift only | 1.562 | 1.480 |
| KNN graph covariance, fixed mean | 1.493 | 1.220 |
| KNN graph covariance, daily elevation drift | 1.401 | 1.202 |
| Geographic/elevation covariance, fixed mean | 1.340 | 1.186 |
| Geographic/elevation covariance, daily elevation drift | 1.352 | 1.178 |

Graph and geographic models in the matched-drift comparison see the same
covariates. The graph model loses on average and on 19/26 blocked and 15/26
random diagnostic weeks. On blocked masks, the best mean model's p90 absolute
error is 2.985 °C and its high-elevation MAE is 2.029 °C. These are operational
failure locations to investigate, not evidence of a new algorithm. Counts
are repeated predictions on correlated stations/dates/masks, not independent
sample counts. Methods tune separately by mask. Grid winners can sit on a
boundary; listed results do not establish dominance over all graph models or
all residual models. The data are daily mean, not Tmax; we do not report an
extreme heat-wave index. The 2023 test set was neither read nor scored.

The two matched means are climatology plus an optional free same-day
intercept and elevation coefficient using observed residuals only. The
conditional nugget is included on observed covariance and not cross between
distinct nodes. The graph prior is a resolvent of a six-nearest-neighbor
weighted Laplacian with diagonal stabilization. It is a conventional control,
not a claim of originality.
