# When does a geographic graph help fill station outages?

Four CPU-only, reproducible case studies on [NOAA Global Summary of the Day (GSOD)](https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day).
They vary **what the estimator is allowed to know**, not just which numbers get hidden.
The included results are useful negative comparisons, not a new smoother
or a comprehensive benchmark.

| 2022 diagnostic setting | Compared methods | MAE (°C, lower is better) |
| --- | --- | ---: |
| Geographic block outage, known stations | Graph covariance with daily drift / geographic-elevation covariance with the same drift | 1.401 / **1.352** |
| Geographic block outage, known stations | Fixed-mean geographic-elevation covariance / 2021 empirical covariance | 1.340 / **0.996** |
| Geographic block outage, known stations | Historical empirical covariance / partial blend with similar historical days | 0.996 / **0.993** |
| Entire stations excluded from fitting and tuning | Graph resolvent / geographic-elevation covariance | 1.885 / **1.655** |

The analog gain is tiny and reverses in the blocked high-elevation subgroup.
The graph-resolvent comparison loses in all five cold-station folds. This is
evidence about these models and this panel, not a general ranking of graph
algorithms.

## Run it

On Python 3.12 with Bash, from the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh
```

After dependencies are installed, the default reproduction is offline.
It verifies and copies two bundled GSOD-derived arrays (about 58 KB combined).
On a single BLAS thread in the reference Linux environment, the four
experiments took about 45 seconds using already prepared data.

Optional `./reproduce.sh --refresh-online` requests exactly the 72 public
NOAA station-year CSV files, checks SHA-256 and rejects source revisions.
NOAA can return a temporary error; the offline snapshot is the default for
a reliable reproduction. Use `./reproduce.sh --cached-raw /path/to/raw` if
you already have the pinned original bytes at `raw/YEAR/STATION.csv`.
The exact panel, source links and hashes are in `realdata/`. The experiment
pipeline writes diagnostic predictions, scores and tuning records and
checks the four numerical result tables against `expected/`.

## The two information contracts

- **Known stations, same-day gap filling:** all 36 stations may supply historical
  recordings. On each 2022 date, a model gets same-day temperatures only from
  the observed subset. Random and contiguous geographic masks use the same
  hidden count; hyperparameters use even weeks and the reported diagnostic
  results use odd weeks. Each date has four masks per regime.
- **Cold stations, nested selection:** target locations and elevations are known;
  target temperatures are unavailable across *all* training and tuning days for
  that outer fold. Inner station folds tune models on other stations. Target
  station history is ineligible for empirical covariance or historical analog
  at those targets, so neither method appears on this comparison. The two
  displayed error scales are not comparable rankings across tasks.

These are 36 northern California stations selected using 2021 availability.
The references are reported daily mean temperatures, not clean latent truth.
The same 2022 diagnostic year was explored repeatedly and is explicitly
development evidence, not an independent holdout. There are no 2023 inputs
or results in this project. Adjacent dates and repeated masks are dependent;
prediction counts are not independent sample sizes. The compact covariance
grids and one station panel limit external validity. This is offline same-day
gap filling, not a weather forecast or a real-time deployment guarantee.

## Related resources and data

- [GSPBench](https://pypi.org/project/gspbench/) already includes NOAA GHCN
  weather graphs and standard interpolation baselines.
- [STA-GANN](https://github.com/blisky-li/STAGANN) already evaluates
  spatiotemporal kriging at unseen sensors.
- [Doherty et al.](https://gmd.copernicus.org/articles/18/3003/2025/) study
  detrending and spatial kriging for daily maximum temperature in California.
  The current case study uses mean temperature, different data, and no claim
  of reproducing their pipeline.
- Data source: NOAA/NCEI GSOD. [Public-use terms and attribution](https://registry.opendata.aws/noaa-gsod/).
  Only derived US station arrays and provenance are bundled, no raw agency CSVs.
  Data are subject to upstream revisions;
  authorship of observations remains with their sources.

The project code is under the [MIT License](LICENSE); the NOAA source data
retain their own public-use terms.
