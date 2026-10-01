"""LiftCast 2000 - Streamlit edition.

Run locally:  pip install -r requirements.txt  then  streamlit run streamlit_app.py
Deploy:       push this file + requirements.txt to GitHub, then share.streamlit.io
Uploads are processed in memory (per session) and never written to disk by this app.
"""
import io
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LiftCast 2000", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp{background:#008080;font-family:Verdana,Arial,sans-serif}
.block-container{background:#c0c0c0;border:3px outset #fff;padding:1.5rem 2rem;max-width:1000px;margin-top:1rem}
h1{font-family:"Times New Roman",serif!important;color:#000080!important;text-shadow:2px 2px #fff}
h2,h3{background:#000080;color:#fff!important;padding:3px 8px!important;font-size:16px!important}
.stButton>button,.stDownloadButton>button{background:#c0c0c0;color:#000;border:2px outset #fff;border-radius:0}
.stButton>button:active{border-style:inset}
[data-testid="stFileUploaderDropzone"]{background:#ffffcc;border:2px dashed #000080;border-radius:0}
div[data-baseweb="select"]>div,input{border-radius:0!important}
hr{height:4px;border:0;background:linear-gradient(90deg,red,orange,yellow,green,blue,purple)}
</style>""", unsafe_allow_html=True)

MAX_BYTES, MAX_ROWS, MAX_SERIES = 10 * 1024 * 1024, 200_000, 5_000
KEYS = ["location", "customer", "product"]
REQUIRED = KEYS + ["month", "volume"]


def forecast_series(y: np.ndarray, h: int):
    """y = monthly volumes, oldest first, no gaps. Returns (h forecasts, note).

    Level = last 12 months average. Seasonal index per calendar month is averaged over every
    full 12-month block of history (shrunk toward 1 when there are few years). A damped
    year-over-year trend is applied when 24+ months exist. Swap this function for
    statsforecast/AutoETS later if you want; the rest of the app won't change.
    """
    n = len(y)
    if n < 12:
        return [float(y[-3:].mean())] * h, "Short history (flat)"
    level = y[-12:].mean()
    if level <= 0:
        return [0.0] * h, "No recent volume"
    years = n // 12
    blocks = np.array([y[n - 12 * (k + 1): n - 12 * k] for k in range(years)])  # each block ends at last month
    means = blocks.mean(axis=1, keepdims=True)
    ratios = blocks / np.where(means == 0, 1, means)
    w = years / (years + 1)                      # 1 yr -> 0.5, 2 yrs -> 0.67, 3 yrs -> 0.75
    idx = 1 + w * (ratios.mean(axis=0) - 1)
    idx = idx / idx.mean()                       # idx[j] = season of month j+1 after the last month
    ratio = 1.0
    if years >= 2 and y[-24:-12].mean() > 0:
        ratio = float(np.clip(level / y[-24:-12].mean(), 0.8, 1.2))
    out = [max(0.0, level * idx[(k - 1) % 12] * ratio ** (0.5 * k / 12)) for k in range(1, h + 1)]
    seasonal = idx.min() < 0.6 or idx.max() > 1.4
    note = ("Seasonal" if seasonal else "Flat") + (" (1 yr of data)" if years == 1 else f" ({years} yrs of data)")
    return out, note


def run_forecast(df: pd.DataFrame, h: int):
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    if len(months) < 12:
        raise ValueError(f"Need at least 12 months of history; the file spans {len(months)}.")
    hist = df.groupby(KEYS + ["month"], as_index=False)["volume"].sum()
    future = pd.period_range(months[-1] + 1, periods=h, freq="M")
    rows = []
    for key, g in hist.groupby(KEYS):
        y = g.set_index("month")["volume"].reindex(months, fill_value=0).to_numpy(float)
        vals, note = forecast_series(y, h)
        for m, v in zip(future, vals):
            rows.append(dict(zip(KEYS, key), month=str(m), forecast=round(v, 1), note=note))
    hist["month"] = hist["month"].astype(str)
    return hist.rename(columns={"volume": "y"}), pd.DataFrame(rows)


def clean(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, nrows=MAX_ROWS + 1, skipinitialspace=True)
    except Exception:
        raise ValueError("That file could not be read as a CSV.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing) + ". Use the template.")
    df = df[REQUIRED].dropna(how="all")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Too many rows (limit {MAX_ROWS:,}).")
    for c in KEYS:
        df[c] = df[c].fillna("").str.strip()
        if (df[c] == "").any():
            raise ValueError(f"Blank values found in '{c}'.")
    df["volume"] = pd.to_numeric(df["volume"].str.replace(",", "", regex=False), errors="coerce")
    if df["volume"].isna().any() or (df["volume"] < 0).any():
        raise ValueError("'Volume' must be a number that is zero or higher in every row.")
    parsed = pd.to_datetime(df["month"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Could not read some 'Month' values. Use YYYY-MM, like 2026-01.")
    df["month"] = parsed.dt.to_period("M")
    if df.groupby(KEYS).ngroups > MAX_SERIES:
        raise ValueError(f"Too many location/customer/product combinations (limit {MAX_SERIES:,}).")
    return df




def template_csv() -> str:
    months = pd.period_range(end=pd.Period(date.today(), "M") - 1, periods=12, freq="M")
    vols = [410, 395, 420, 450, 470, 500, 510, 495, 460, 430, 90, 80]
    lines = ["Location,Customer,Product,Month,Volume"]
    lines += [f"Example Terminal,Example Customer,Product A,{m},{v}" for m, v in zip(months, vols)]
    return "\n".join(lines) + "\n"


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=["object", "string"]):
        df[c] = df[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v)
    return df


st.title("LiftCast 2000")
st.caption("The Monthly Lifting Forecast Calculator. Your file is processed in memory and not saved.")
st.markdown("<hr>", unsafe_allow_html=True)

st.header("Step 1: Get the template")
st.write("Columns: Location, Customer, Product, Month (YYYY-MM), Volume. Include at least 12 months; 24 to 36 is better for seasonality.")
st.download_button("Download Template (.csv)", template_csv(), "lifting_template.csv", "text/csv")

st.header("Step 2: Upload your file")
up = st.file_uploader("Drop your CSV here", type="csv")

st.header("Step 3: Calculate")
h = st.selectbox("Months to forecast", [3, 6, 12, 24], index=2)
if st.button("Calculate Forecast", disabled=up is None, type="primary"):
    try:
        with st.spinner("Calculating..."):
            hist, fc = run_forecast(clean(up.getvalue()), h)
        st.session_state["res"] = (hist, fc)
    except ValueError as e:
        st.session_state.pop("res", None)
        st.error(f"Error: {e}")

if "res" in st.session_state:
    hist, fc = st.session_state["res"]
    st.header("Step 4: Results")
    st.caption("Click a box and start typing to search. Leave a box empty to include everything.")
    c1, c2, c3 = st.columns(3)
    sel = {}
    for col, k in zip((c1, c2, c3), KEYS):
        sel[k] = col.multiselect(k.title(), sorted(hist[k].unique()), placeholder=f"Type to find a {k}...")
    q = st.text_input("Search all fields", placeholder="Type any part of a location, customer or product...")
    g = st.selectbox("Group by", KEYS, format_func=str.title)

    def keep(d):
        m = pd.Series(True, index=d.index)
        for k, v in sel.items():
            if v:
                m &= d[k].isin(v)
        if q.strip():
            m &= d[KEYS].agg(" ".join, axis=1).str.contains(q.strip(), case=False, regex=False)
        return d[m]

    h_, f_ = keep(hist), keep(fc)
    if f_.empty:
        st.warning("Nothing matches those filters.")
    else:
        st.subheader("Total volume: actual vs forecast")
        a = h_.groupby("month")["y"].sum()
        b = f_.groupby("month")["forecast"].sum()
        chart = pd.DataFrame({"Actual": a, "Forecast": b}).sort_index()
        chart.loc[a.index.max(), "Forecast"] = a.iloc[-1]          # join the two lines
        st.line_chart(chart)

        first = f_["month"].min()
        last12 = sorted(h_["month"].unique())[-12:]
        tbl = pd.DataFrame({
            f"{first} forecast": f_[f_["month"] == first].groupby(g)["forecast"].sum(),
            f"{f_['month'].nunique()}-month total": f_.groupby(g)["forecast"].sum(),
            "Last 12 mo actual": h_[h_["month"].isin(last12)].groupby(g)["y"].sum(),
        }).fillna(0).round(0).astype(int).sort_values(f"{f_['month'].nunique()}-month total", ascending=False)
        st.subheader(f"Forecast by {g}")
        st.dataframe(tbl)
        st.download_button("Download Forecast (.csv)", csv_safe(f_).to_csv(index=False),
                           "forecast.csv", "text/csv")
    st.caption("Forecasts are estimates. Seasonal patterns come from your history, so a repeating winter dip carries forward.")
