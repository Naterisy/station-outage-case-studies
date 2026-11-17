# Development-only falsification: does changing the graph help after controlling for same-day drift?

Variable is observed daily mean °C (not Tmax; no heat-wave claim). Spatial
same-day interpolation only. No 2023/test data. This is a stronger comparator
inspired by https://gmd.copernicus.org/articles/19/8895/2026/ and
https://gmd.copernicus.org/articles/18/3003/2025/, not their reproduction
(no coast/terrain grids, daily variogram fitting or Gaussian-score transform).
All methods get identical 2021 seasonal information, site geometry/elevation
and same-day observed values. Compare:
1. Fixed-mean exponential spatial/elevation GP with seasonality.
2. Universal kriging with precisely the same covariance family and
   intercept/elevation daily drift on station anomalies; include drift-only.
3. A graph residual model using the same meter-scale coordinates and
   configurable elevation. Compare with/without the same daily drift under
   the same GLS formula so graph-vs-geostatistics does not confound drift.

Keep four masks per kind/date. Number hidden is round(30% available).
Blocked hides stations nearest each sampled center geographically.
Use even calendar weeks for tuning and odd for diagnostics (not independent;
already used the same year). Tune each model on its own grid by masked MAE
separately by kind. Report descriptive per-week paired direction, high
elevation, and absolute-error tails; do not call week counts significance.
Keep both negative and positive results. No new-method or publication
decision on this development experiment alone.
