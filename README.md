# Model validation tool

A Streamlit app for comparing calibration strategies of a hydrological model at
several stream gauges.

- **Map:** each gauge is a dot, colored by a performance metric (NSE, CPM scores
  or volume ratio) of the best case found by a chosen strategy.
- **Hydrograph:** click a gauge to see observed discharge (black dots) and the
  best-case simulation of each calibration strategy (colored lines). The line
  style shows the criterion used to pick the best case.
- **Flow duration curve / Q-Q plot:** switch between the two next to the
  hydrograph.
- **Performance summary:** a table of the metrics for each strategy at the
  selected gauge, and a comparison across all gauges.

Observed discharge is shown as measured (m³/s). Each simulated series is
rescaled to the observed mean (`sim / mean(sim) * mean(obs)`), so the plots
compare shape and timing. Volume bias appears in the table's volume ratio column.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

## Example data

The `example_data/` folder is **example data**. The app loads it by default so
it runs out of the box. It contains five calibration strategies (NSGA-II, Sobol,
TuRBO, DDS and MOGP Pareto) for five USGS gauges in the Indian River Lagoon
basin, Florida, from 2017 to 2019:

| Gauge | USGS ID | Time step |
|---|---|---|
| CraneCreek | 02249500 | hourly |
| Turkey | 02250030 | daily |
| EauGallie | 02249007 | hourly |
| NorthProngSaintSeb | 02251500 | hourly |
| SouthProngSaintSeb | 02251000 | hourly |

To use your own results, put files with the same layout in a folder and point
the app at it:

```
COMPARE_DATA_DIR=/path/to/compare_data streamlit run app.py
```

(PowerShell: `$env:COMPARE_DATA_DIR = "C:\path\to\compare_data"; streamlit run app.py`)

### Data layout

| File | Contents |
|---|---|
| `meta.json` | Strategies (label, color), gauges, metrics and their display names, best-case criteria and bases |
| `evaluations.parquet` | One row per evaluation: `strategy`, `id`, overall metrics, and per-gauge metrics as `<metric>@<gauge>` |
| `picks.parquet` | Best case per `strategy`, `gauge`, `criterion` and `basis` (`mean` of gauges or this `gauge`), as `eval_id` |
| `series_<gauge>.parquet` | Time-indexed (`time_utc`) discharge: an `observed` column and one `<strategy>\|<eval_id>` column per simulation |
| `runtimes.csv` *(optional)* | One row per `strategy`; its other columns (e.g. run time) are added to the summary table |

Gauge locations and drainage areas are in `gauges.csv` (one row per gauge, with
`gauge`, `site_no`, `site_name`, `lat`, `lon`, `usgs_area_km2`).
