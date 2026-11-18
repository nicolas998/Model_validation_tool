"""Compare calibration strategies at the IRL gauges.

Run with:  streamlit run app.py
The compare data folder defaults to DEFAULT_DATA_DIR and can be overridden
with the COMPARE_DATA_DIR environment variable.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = r"C:\Users\nicol\Projects\2026_IRL_project\compare_data"
DATA_DIR = Path(os.environ.get("COMPARE_DATA_DIR", DEFAULT_DATA_DIR))

EXTRA_METRICS = {"volume_ratio": "Volume ratio"}  # ideal value 1, not "higher is better"
DASHES = ["solid", "dash", "dot", "dashdot"]
DEFAULT_COLOR = "#ff7f0e"
PROBS = np.linspace(0.0005, 0.9995, 600)  # quantile grid for the FDC and Q-Q plots
PLOT_HEIGHT = 460
BEST_STYLE = "font-weight: bold; color: #0a7d32"
GAUGE_TRACE = 1  # index of the clickable gauge trace in the map figure

st.set_page_config(page_title="Calibration comparison", layout="wide")


# --------------------------------------------------------------------------- data


@st.cache_data
def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "meta.json").read_text())


@st.cache_data
def load_evaluations(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "evaluations.parquet").set_index(["strategy", "id"])


@st.cache_data
def load_picks(data_dir: Path) -> pd.Series:
    picks = pd.read_parquet(data_dir / "picks.parquet")
    return picks.set_index(["strategy", "gauge", "criterion", "basis"])["eval_id"]


GAUGES_CSV = APP_DIR / "gauges.csv"


@st.cache_data
def load_gauges(mtime: float) -> pd.DataFrame:  # mtime: reload when gauges.csv is edited
    return pd.read_csv(GAUGES_CSV, dtype={"site_no": str}).set_index("gauge")


@st.cache_data
def load_series(data_dir: Path, gauge: str) -> tuple[pd.DataFrame, float]:
    """Observed discharge as is (m³/s); each simulation rescaled to the observed mean.

    sim_scaled = sim / mean(sim) * mean(obs), so every curve is in m³/s with the observed volume
    and the plots compare shape and timing only (volume bias is reported in the table).
    """
    df = pd.read_parquet(data_dir / f"series_{gauge}.parquet")
    obs_mean = float(df["observed"].mean())
    sims = df.columns.drop("observed")
    df[sims] = df[sims] / df[sims].mean() * obs_mean
    return df, obs_mean


@st.cache_data
def load_runtimes(data_dir: Path) -> pd.DataFrame | None:
    """Optional runtimes.csv (strategy + any timing columns), merged into the table."""
    path = data_dir / "runtimes.csv"
    return pd.read_csv(path).set_index("strategy") if path.exists() else None


meta = load_meta(DATA_DIR)
evals = load_evaluations(DATA_DIR)
picks = load_picks(DATA_DIR)
gauges = load_gauges(GAUGES_CSV.stat().st_mtime).reindex(meta["gauges"])
runtimes = load_runtimes(DATA_DIR)

STRATEGIES = [s["label"] for s in meta["strategies"]]
COLORS = {s["label"]: s["color"] for s in meta["strategies"]}
METRICS = meta["metrics"] + list(EXTRA_METRICS)
METRIC_NAMES = {**meta["metric_names"], **EXTRA_METRICS}
N_EVALS = evals.groupby(level="strategy").size()


def criterion_name(c: str) -> str:
    return "Sum of metrics" if c == "Sum" else METRIC_NAMES.get(c, c)


def pick(strategy: str, gauge: str, criterion: str, basis: str) -> int | None:
    eid = picks.get((strategy, gauge, criterion, basis))
    return None if eid is None else int(eid)


def metric_at(strategy: str, eval_id: int, metric: str, gauge: str) -> float:
    return float(evals.at[(strategy, eval_id), f"{metric}@{gauge}"])


def quantiles(s: pd.Series) -> np.ndarray:
    v = s.dropna().to_numpy()
    return np.quantile(v, PROBS) if len(v) else np.full(PROBS.shape, np.nan)


# --------------------------------------------------------------------------- state

if "gauge" not in st.session_state:
    st.session_state.gauge = gauges.index[0]

# Apply a click on the map before the gauge selector is drawn, so both stay in sync.
# Only a *new* selection overrides the selector.
_map_state = st.session_state.get("map")
_points = _map_state["selection"]["points"] if _map_state else []
_sel = tuple((p.get("curve_number"), p.get("point_index")) for p in _points)
if _sel and _sel != st.session_state.get("_last_map_sel"):
    st.session_state._last_map_sel = _sel
    curve, idx = _sel[0]
    if curve == GAUGE_TRACE and idx is not None:
        st.session_state.gauge = gauges.index[idx]

# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.selectbox("Gauge", list(gauges.index), key="gauge", help="Or click a gauge on the map.")

    st.header("Best case")
    criteria = st.multiselect(
        "Best case by",
        meta["criteria"],
        default=["Sum"],
        format_func=criterion_name,
        help="One curve per strategy and criterion (line style = criterion).",
    )
    basis = st.radio(
        "Ranked on",
        list(meta["bases"]),
        format_func=meta["bases"].get,
        help="Best parameter set for the mean of all gauges, or for this gauge alone.",
    )
    strategies = st.multiselect("Strategies", STRATEGIES, default=STRATEGIES)
    show_default = st.checkbox("Show default parameters (eval 0)", value=False)

    st.header("Map")
    map_metric = st.selectbox("Color gauges by", METRICS, format_func=METRIC_NAMES.get)
    map_strategy = st.selectbox("Strategy", STRATEGIES)
    map_criterion = st.selectbox("Best case by", meta["criteria"], format_func=criterion_name, key="map_crit")

gauge = st.session_state.gauge
ginfo = gauges.loc[gauge]
series, obs_mean = load_series(DATA_DIR, gauge)

# --------------------------------------------------------------------------- map

st.title("Calibration strategy comparison")

map_values = []
for g in gauges.index:
    eid = pick(map_strategy, g, map_criterion, basis)
    map_values.append(np.nan if eid is None else metric_at(map_strategy, eid, map_metric, g))
map_values = np.array(map_values)

if map_metric == "volume_ratio":
    span = np.nanmax(np.abs(map_values - 1)) or 1.0
    color_kw = dict(colorscale="RdBu", cmin=1 - span, cmax=1 + span)
else:
    color_kw = dict(colorscale="Viridis", cmin=np.nanmin(map_values), cmax=np.nanmax(map_values))

map_fig = go.Figure()
map_fig.add_trace(  # halo + label under the selected gauge (drawn first so it sits below)
    go.Scattermap(
        lat=[ginfo["lat"]],
        lon=[ginfo["lon"]],
        mode="markers+text",
        marker=dict(size=30, color="black", opacity=0.35),
        text=[gauge],
        textposition="top right",
        textfont=dict(size=13, color="black"),
        hoverinfo="skip",
        showlegend=False,
    )
)
map_fig.add_trace(  # GAUGE_TRACE: the clickable gauges
    go.Scattermap(
        lat=gauges["lat"],
        lon=gauges["lon"],
        mode="markers",
        marker=dict(
            size=18,
            color=map_values,
            colorbar=dict(title=dict(text=METRIC_NAMES[map_metric], side="right"), thickness=14),
            **color_kw,
        ),
        unselected=dict(marker=dict(opacity=1)),
        customdata=[[g, site, float(v)] for g, site, v in zip(gauges.index, gauges["site_no"], map_values)],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>USGS %{customdata[1]}<br>"
            f"{METRIC_NAMES[map_metric]}: " + "%{customdata[2]:.3f}<extra></extra>"
        ),
        showlegend=False,
    )
)
map_fig.update_layout(
    map=dict(
        style="carto-positron",
        center=dict(lat=float(gauges["lat"].mean()), lon=float(gauges["lon"].mean())),
        zoom=8.4,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=420,
    clickmode="event+select",
)

map_col, info_col = st.columns([3, 1])
with map_col:
    st.caption(
        f"Color: {METRIC_NAMES[map_metric]} of the {map_strategy} best case by "
        f"{criterion_name(map_criterion).lower()} ({meta['bases'][basis].lower()}). Click a gauge to select it."
    )
    st.plotly_chart(map_fig, key="map", on_select="rerun", selection_mode="points", config={"scrollZoom": True})
with info_col:
    st.subheader(gauge)
    st.markdown(
        f"**{ginfo['site_name']}**  \n"
        f"USGS [{ginfo['site_no']}](https://waterdata.usgs.gov/monitoring-location/USGS-{ginfo['site_no']}/)"
    )
    st.metric("Observed mean flow", f"{obs_mean:.3f} m³/s")
    st.metric("Drainage area (USGS)", f"{ginfo['usgs_area_km2']:.1f} km²")
    st.caption(f"{series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d}, {len(series):,} time steps")

# --------------------------------------------------------------------------- curves

curves = []  # one per (strategy, eval id); criteria picking the same run share a curve
for s in strategies:
    by_id: dict[int, list[str]] = {}
    for c in criteria:
        eid = pick(s, gauge, c, basis)
        if eid is not None:
            by_id.setdefault(eid, []).append(c)
    for eid, crits in by_id.items():
        col = f"{s}|{eid}"
        if col in series:
            curves.append(
                dict(
                    column=col,
                    color=COLORS[s],
                    dash=DASHES[criteria.index(crits[0]) % len(DASHES)],
                    label=f"{s} · {', '.join(crits)} (#{eid})",
                )
            )
if show_default:
    default_col = next((c for c in series.columns if c.endswith("|0")), None)
    if default_col:
        curves.append(dict(column=default_col, color=DEFAULT_COLOR, dash="dashdot", label="Default parameters (#0)"))

Y_LABEL = "Discharge (m³/s)"

st.divider()
if not curves:
    st.info("Select at least one strategy and one criterion in the sidebar.")
else:
    hydro_col, dist_col = st.columns([2, 1])

    with hydro_col:
        log_y = st.toggle("Log scale", value=False, key="hydro_log")
        hfig = go.Figure()
        hfig.add_trace(
            go.Scattergl(
                x=series.index,
                y=series["observed"],
                mode="markers",
                marker=dict(color="black", size=3),
                name="Observed",
            )
        )
        for c in curves:
            hfig.add_trace(
                go.Scattergl(
                    x=series.index,
                    y=series[c["column"]],
                    mode="lines",
                    line=dict(color=c["color"], dash=c["dash"], width=1.6),
                    name=c["label"],
                )
            )
        hfig.update_layout(
            height=PLOT_HEIGHT,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f"{gauge}: discharge (simulated rescaled to the observed mean)", font=dict(size=14)),
            yaxis=dict(title=Y_LABEL, type="log" if log_y else "linear"),
            legend=dict(orientation="h", yanchor="top", y=-0.12, font=dict(size=11)),
            hovermode="x unified",
        )
        st.plotly_chart(hfig, key="hydro")

    with dist_col:
        view = st.segmented_control(
            "View", ["Duration curve", "Q-Q plot"], default="Duration curve", label_visibility="collapsed"
        ) or "Duration curve"
        q_obs = quantiles(series["observed"])
        dfig = go.Figure()
        if view == "Duration curve":
            exceed = (1 - PROBS) * 100
            dfig.add_trace(
                go.Scatter(x=exceed, y=q_obs, mode="markers", marker=dict(color="black", size=4), name="Observed")
            )
            for c in curves:
                dfig.add_trace(
                    go.Scatter(
                        x=exceed,
                        y=quantiles(series[c["column"]]),
                        mode="lines",
                        line=dict(color=c["color"], dash=c["dash"], width=1.8),
                        name=c["label"],
                    )
                )
            dfig.update_layout(
                xaxis=dict(title="Exceedance probability (%)"),
                yaxis=dict(title=Y_LABEL, type="log"),
                title=dict(text="Flow duration curve", font=dict(size=14)),
            )
        else:
            lo, hi = np.nanmin(q_obs[q_obs > 0]), np.nanmax(q_obs)
            for c in curves:
                q_sim = quantiles(series[c["column"]])
                dfig.add_trace(
                    go.Scatter(
                        x=q_obs,
                        y=q_sim,
                        mode="lines",
                        line=dict(color=c["color"], dash=c["dash"], width=1.8),
                        name=c["label"],
                    )
                )
                lo = min(lo, np.nanmin(q_sim[q_sim > 0], initial=lo))
                hi = max(hi, np.nanmax(q_sim))
            dfig.add_trace(
                go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dot", width=1), name="1:1"
                )
            )
            dfig.update_layout(
                xaxis=dict(title="Observed discharge (m³/s)", type="log"),
                yaxis=dict(title="Simulated discharge (m³/s)", type="log"),
                title=dict(text="Q-Q plot", font=dict(size=14)),
            )
        dfig.update_layout(height=PLOT_HEIGHT, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(dfig, key="dist")

# --------------------------------------------------------------------------- tables

st.divider()
st.subheader("Performance summary")

HIGHER_BETTER = [METRIC_NAMES[m] for m in meta["metrics"]]
RATIO_COLS = [METRIC_NAMES[m] for m in EXTRA_METRICS]


def style_table(df: pd.DataFrame, higher_better: list[str], ratio: list[str]):
    sty = df.style.format(precision=3, na_rep="—")
    if higher_better:
        sty = sty.highlight_max(subset=higher_better, props=BEST_STYLE)
    for col in ratio:
        dist = (df[col] - 1).abs()
        sty = sty.apply(lambda s, d=dist: np.where(d == d.min(), BEST_STYLE, ""), subset=[col])
    return sty


tab_gauge, tab_all = st.tabs([f"{gauge}", "All gauges"])

with tab_gauge:
    rows = []
    for s in strategies:
        for c in criteria:
            eid = pick(s, gauge, c, basis)
            if eid is None:
                continue
            row = {"Strategy": s, "Best by": criterion_name(c), "Eval id": eid}
            for m in METRICS:
                row[METRIC_NAMES[m]] = metric_at(s, eid, m, gauge)
            row["Evaluations"] = int(N_EVALS.get(s, 0))
            if runtimes is not None and s in runtimes.index:
                row.update(runtimes.loc[s].to_dict())
            rows.append(row)
    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(style_table(table, HIGHER_BETTER, RATIO_COLS), hide_index=True)
        st.caption(
            f"Metrics computed at {gauge} for each strategy's best case ({meta['bases'][basis].lower()}). "
            "Best value per column in green (volume ratio: closest to 1)."
            + ("" if runtimes is not None else " Run times are not available yet.")
        )
    else:
        st.info("Select at least one strategy and one criterion in the sidebar.")

with tab_all:
    all_rows = {}
    for s in strategies:
        vals = {}
        for g in gauges.index:
            eid = pick(s, g, map_criterion, basis)
            vals[g] = np.nan if eid is None else metric_at(s, eid, map_metric, g)
        all_rows[s] = vals
    if all_rows:
        wide = pd.DataFrame(all_rows).T
        wide["Mean"] = wide.mean(axis=1)
        wide.index.name = "Strategy"
        wide = wide.reset_index()
        cols = list(gauges.index) + ["Mean"]
        if map_metric in EXTRA_METRICS:
            st.dataframe(style_table(wide, [], cols), hide_index=True)
        else:
            st.dataframe(style_table(wide, cols, []), hide_index=True)
        st.caption(
            f"{METRIC_NAMES[map_metric]} at each gauge for each strategy's best case by "
            f"{criterion_name(map_criterion).lower()} ({meta['bases'][basis].lower()}), the same selection as the map."
        )
