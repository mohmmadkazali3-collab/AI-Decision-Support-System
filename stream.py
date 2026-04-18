import io
import json
import os
from typing import Any

from reportlab.lib.styles import getSampleStyleSheet

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ─── HARDCODED API KEY ───────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyCtUM4x2UuTSQHkRid-PvA-NXuSnNhMkmU"
GEMINI_MODEL   = "gemini-2.0-flash"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Intelligence Dataset",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 1.4rem; }
        .kpi-card {
            background: linear-gradient(135deg, rgba(38,39,57,0.95), rgba(24,25,37,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px; padding: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }
        .kpi-label { color: #9aa4bf; font-size: 0.85rem; margin-bottom: 0.25rem; }
        .kpi-value { color: #f2f5ff; font-size: 1.65rem; font-weight: 700; line-height: 1.1; }
        .kpi-help { color: #93a0bd; font-size: 0.76rem; margin-top: 0.3rem; }
        .section-title { font-size: 1.04rem; font-weight: 650; color: #e8ecf7; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    name = file_name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(buffer)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def infer_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def infer_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        if series.dtype == "object":
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.6:
                date_cols.append(col)
    return date_cols


def detect_business_columns(df: pd.DataFrame) -> dict[str, str | None]:
    columns = list(df.columns)

    def pick(candidates: list[str], numeric: bool | None = None) -> str | None:
        for keyword in candidates:
            for col in columns:
                if keyword in col.lower():
                    if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
                        continue
                    return col
        return None

    date_col = pick(["date", "day", "booking", "checkin", "timestamp", "created"])
    if date_col is None:
        guessed_dates = infer_date_columns(df)
        date_col = guessed_dates[0] if guessed_dates else None

    return {
        "price_col": pick(["price", "rate", "cost", "amount"], numeric=True),
        "rating_col": pick(["rating", "score", "review"], numeric=True),
        "city_col": pick(["city", "location", "district", "area", "country", "region"]),
        "hotel_col": pick(["hotel", "property", "name"]),
        "date_col": date_col,
    }


def get_stats(df: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def auto_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = get_stats(df)
    cleaned = df.copy().drop_duplicates()
    num_cols = infer_numeric_columns(cleaned)
    cat_cols = infer_categorical_columns(cleaned)

    for col in num_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    for col in cat_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            cleaned[col] = cleaned[col].fillna("Unknown" if mode.empty else mode.iloc[0])

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    after = get_stats(cleaned)
    report = {
        "rows_before": before["rows"],
        "rows_after": after["rows"],
        "missing_before": before["missing"],
        "missing_after": after["missing"],
        "duplicates_before": before["duplicates"],
        "duplicates_after": after["duplicates"],
    }
    return cleaned, report


def manual_fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                if strategy == "mean":
                    out[col] = out[col].fillna(out[col].mean())
                elif strategy == "median":
                    out[col] = out[col].fillna(out[col].median())
                else:
                    mode = out[col].mode(dropna=True)
                    out[col] = out[col].fillna(out[col].median() if mode.empty else mode.iloc[0])
            else:
                mode = out[col].mode(dropna=True)
                out[col] = out[col].fillna("Unknown" if mode.empty else mode.iloc[0])
    return out


def remove_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()


def make_kpi_card(title: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_dataset_kpis(df: pd.DataFrame, detected: dict[str, str | None]) -> dict[str, Any]:
    price = detected["price_col"]
    rating = detected["rating_col"]
    hotel = detected["hotel_col"]
    return {
        "total_hotels": int(df[hotel].nunique()) if hotel and hotel in df.columns else int(df.shape[0]),
        "avg_price": float(df[price].mean()) if price and price in df.columns and pd.api.types.is_numeric_dtype(df[price]) else np.nan,
        "avg_rating": float(df[rating].mean()) if rating and rating in df.columns and pd.api.types.is_numeric_dtype(df[rating]) else np.nan,
    }


def power_bi_suggestions(df: pd.DataFrame, detected: dict[str, str | None]) -> dict[str, Any]:
    numeric = infer_numeric_columns(df)
    categorical = infer_categorical_columns(df)
    dates = infer_date_columns(df)
    has_geo = detected["city_col"] is not None or any(
        k in " ".join(df.columns).lower() for k in ["city", "country", "region", "location", "lat", "lon"]
    )

    dashboard_type = "Performance Dashboard"
    if dates and numeric:
        dashboard_type = "Trend & Performance Dashboard"
    if has_geo and numeric:
        dashboard_type = "Geographic Performance Dashboard"
    if len(categorical) >= 2 and numeric:
        dashboard_type = "Operational Comparison Dashboard"

    kpi_suggestions = []
    for col in numeric[:5]:
        kpi_suggestions.append(f"Average {col}")
        kpi_suggestions.append(f"Total {col}")
    if detected["hotel_col"]:
        kpi_suggestions.append(f"Distinct {detected['hotel_col']}")
    if detected["rating_col"]:
        kpi_suggestions.append(f"Average {detected['rating_col']}")

    dax = []
    for col in numeric[:3]:
        dax.append(f"Total {col} = SUM('Table'[{col}])")
        dax.append(f"Avg {col} = AVERAGE('Table'[{col}])")
    if dates and numeric:
        dax.append(
            f"MoM Change = DIVIDE([Total {numeric[0]}] - CALCULATE([Total {numeric[0]}], "
            f"DATEADD('Date'[Date], -1, MONTH)), CALCULATE([Total {numeric[0]}], DATEADD('Date'[Date], -1, MONTH)))"
        )

    visuals = ["Bar chart", "Line chart", "KPI cards"]
    if has_geo:
        visuals.append("Map")
    if len(numeric) >= 2:
        visuals.append("Scatter plot")

    return {
        "dashboard_type": dashboard_type,
        "kpis": sorted(set(kpi_suggestions))[:10],
        "dax": dax[:8],
        "visuals": visuals,
        "notes": [
            f"Detected numeric columns: {len(numeric)}",
            f"Detected categorical columns: {len(categorical)}",
            f"Detected date columns: {len(dates)}",
        ],
    }


def apply_filters(df: pd.DataFrame, detected: dict[str, str | None]) -> pd.DataFrame:
    out = df.copy()
    city_col = detected["city_col"]
    price_col = detected["price_col"]
    rating_col = detected["rating_col"]

    c1, c2, c3 = st.columns(3)
    with c1:
        if city_col and city_col in out.columns:
            cities = sorted(out[city_col].dropna().astype(str).unique().tolist())
            selected = st.multiselect(
                "City / Location",
                options=cities,
                default=cities[: min(8, len(cities))],
                key=f"city_filter_{city_col}",
            )
            if selected:
                out = out[out[city_col].astype(str).isin(selected)]
        else:
            st.caption("No city/location column detected.")
    with c2:
        if price_col and price_col in out.columns and pd.api.types.is_numeric_dtype(out[price_col]):
            p_min, p_max = float(out[price_col].min()), float(out[price_col].max())
            if p_min < p_max:
                rng = st.slider("Price range", p_min, p_max, (p_min, p_max), key=f"price_filter_{price_col}")
                out = out[(out[price_col] >= rng[0]) & (out[price_col] <= rng[1])]
            else:
                st.caption("Price has one unique value.")
        else:
            st.caption("No numeric price column detected.")
    with c3:
        if rating_col and rating_col in out.columns and pd.api.types.is_numeric_dtype(out[rating_col]):
            r_min, r_max = float(out[rating_col].min()), float(out[rating_col].max())
            if r_min < r_max:
                rr = st.slider("Rating range", r_min, r_max, (r_min, r_max), key=f"rating_filter_{rating_col}")
                out = out[(out[rating_col] >= rr[0]) & (out[rating_col] <= rr[1])]
            else:
                st.caption("Rating has one unique value.")
        else:
            st.caption("No numeric rating column detected.")
    return out


def init_state() -> None:
    defaults = {
        "datasets": {},
        "selected_dataset": None,
        "cleaning_reports": {},
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()

# ─── Configure Gemini once at startup ────────────────────────────────────────
_gemini_model = None
if genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as _cfg_err:
        st.error(f"Gemini config error: {_cfg_err}")
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 AI Intelligence Dataset")
st.caption("Upload, clean, compare, visualize, and generate AI insights from multiple datasets.")

with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader(
        "Upload one or more datasets",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            try:
                loaded = load_data(file.name, file.getvalue())
                cleaned, report = auto_clean_data(loaded)
                st.session_state.datasets[file.name] = {
                    "raw": loaded,
                    "cleaned": cleaned,
                    "detected": detect_business_columns(cleaned),
                }
                st.session_state.cleaning_reports[file.name] = report
            except Exception as exc:
                st.error(f"{file.name}: {exc}")

    names = list(st.session_state.datasets.keys())
    if names:
        default_idx = (
            names.index(st.session_state.selected_dataset)
            if st.session_state.selected_dataset in names
            else 0
        )
        st.session_state.selected_dataset = st.selectbox("Active dataset", options=names, index=default_idx)
        selected_for_compare = st.multiselect(
            "Datasets for comparison", options=names, default=names[: min(2, len(names))]
        )
    else:
        st.session_state.selected_dataset = None
        selected_for_compare = []

if not st.session_state.datasets:
    st.info("Upload one or more CSV/Excel files from the sidebar to begin.")
    st.stop()

active_name = st.session_state.selected_dataset or list(st.session_state.datasets.keys())[0]
active_entry = st.session_state.datasets[active_name]
active_df = active_entry["cleaned"]
detected = active_entry["detected"]
numeric_cols = infer_numeric_columns(active_df)

tabs = st.tabs(["Overview", "Data Cleaning", "Comparison", "AI Insights", "Power BI Suggestions"])

# ══════════════════════════════ TAB 0 – OVERVIEW ══════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📊 Overview Dashboard</div>', unsafe_allow_html=True)
    st.write(f"Active dataset: `{active_name}`")
    filtered_df = apply_filters(active_df, detected)

    kpis = get_dataset_kpis(filtered_df, detected)
    k1, k2, k3 = st.columns(3)
    with k1:
        make_kpi_card("🏨 Total Hotels / Records", f"{kpis['total_hotels']:,}", "Unique hotels if detected, otherwise row count.")
    with k2:
        make_kpi_card("💰 Avg Price", "N/A" if np.isnan(kpis["avg_price"]) else f"{kpis['avg_price']:,.2f}", "Current filtered selection.")
    with k3:
        make_kpi_card("⭐ Avg Rating", "N/A" if np.isnan(kpis["avg_rating"]) else f"{kpis['avg_rating']:.2f}", "Current filtered selection.")

    st.markdown('<div class="section-title">🗂️ Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df.head(40), use_container_width=True)

    st.markdown('<div class="section-title">📈 Interactive Visuals</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    price_col  = detected["price_col"]
    rating_col = detected["rating_col"]
    city_col   = detected["city_col"]
    date_col   = detected["date_col"]

    with c1:
        if price_col and price_col in filtered_df.columns:
            fig_hist = px.histogram(filtered_df, x=price_col, nbins=35, title=f"Histogram: {price_col}", template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
        if city_col and price_col and city_col in filtered_df.columns and price_col in filtered_df.columns:
            grp = (
                filtered_df.groupby(city_col)[price_col]
                .mean()
                .reset_index()
                .sort_values(price_col, ascending=False)
                .head(20)
            )
            fig_bar = px.bar(grp, x=city_col, y=price_col, color=price_col, template="plotly_dark", title=f"Avg {price_col} by {city_col}")
            st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        if price_col and rating_col and price_col in filtered_df.columns and rating_col in filtered_df.columns:
            fig_sc = px.scatter(
                filtered_df, x=price_col, y=rating_col,
                color=city_col if city_col and city_col in filtered_df.columns else None,
                template="plotly_dark", title=f"{price_col} vs {rating_col}",
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        if city_col and rating_col and city_col in filtered_df.columns and rating_col in filtered_df.columns:
            fig_box = px.box(filtered_df, x=city_col, y=rating_col, template="plotly_dark",
                             title=f"Box: {rating_col} by {city_col}", points="outliers")
            st.plotly_chart(fig_box, use_container_width=True)

    if len(numeric_cols) >= 2:
        corr = filtered_df[numeric_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap", template="plotly_dark", color_continuous_scale="RdBu")
        st.plotly_chart(fig_corr, use_container_width=True)

    if date_col and date_col in filtered_df.columns and price_col and price_col in filtered_df.columns:
        ts_df = filtered_df.copy()
        ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[date_col]).sort_values(date_col)
        if not ts_df.empty:
            ts = ts_df.groupby(date_col)[price_col].mean().reset_index()
            fig_ts = px.line(ts, x=date_col, y=price_col, markers=True, template="plotly_dark", title=f"Time Series: {price_col}")
            st.plotly_chart(fig_ts, use_container_width=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart_type = chart_col1.selectbox("Chart Type", ["Histogram", "Scatter", "Box", "Bar"])
    x_axis = chart_col2.selectbox("X-axis", options=filtered_df.columns.tolist())
    y_axis = chart_col3.selectbox("Y-axis", options=["(None)"] + filtered_df.columns.tolist())
    color_axis = st.selectbox("Color (optional)", options=["(None)"] + filtered_df.columns.tolist())
    color_arg = None if color_axis == "(None)" else color_axis
    y_arg = None if y_axis == "(None)" else y_axis

    try:
        if chart_type == "Histogram":
            fig_dynamic = px.histogram(filtered_df, x=x_axis, color=color_arg, template="plotly_dark")
        elif chart_type == "Scatter" and y_arg:
            fig_dynamic = px.scatter(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        elif chart_type == "Box" and y_arg:
            fig_dynamic = px.box(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        elif chart_type == "Bar" and y_arg:
            fig_dynamic = px.bar(filtered_df, x=x_axis, y=y_arg, color=color_arg, template="plotly_dark")
        else:
            fig_dynamic = None
            st.warning(f"{chart_type} requires a Y-axis.")
        if fig_dynamic is not None:
            st.plotly_chart(fig_dynamic, use_container_width=True)
    except Exception as err:
        st.error(f"Could not render chart: {err}")

# ══════════════════════════════ TAB 1 – DATA CLEANING ═════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)
    st.write(f"Cleaning dataset: `{active_name}`")

    before = get_stats(active_entry["cleaned"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", before["rows"])
    c2.metric("Columns", before["cols"])
    c3.metric("Missing Values", before["missing"])
    c4.metric("Duplicates", before["duplicates"])

    btn1, btn2, btn3 = st.columns(3)
    if btn1.button("Clean Data", use_container_width=True):
        cleaned, report = auto_clean_data(active_entry["cleaned"])
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.session_state.cleaning_reports[active_name] = report
        st.success("Automatic cleaning applied.")
        st.rerun()

    if btn2.button("Remove Missing Values", use_container_width=True):
        cleaned = remove_missing_rows(active_entry["cleaned"])
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.success("Rows with missing values removed.")
        st.rerun()

    fill_strategy = btn3.selectbox("Fill missing values", options=["mean", "median", "mode"], key=f"fill_strategy_{active_name}")
    if st.button("Apply Fill Strategy", use_container_width=True):
        cleaned = manual_fill_missing(active_entry["cleaned"], fill_strategy)
        st.session_state.datasets[active_name]["cleaned"] = cleaned
        st.session_state.datasets[active_name]["detected"] = detect_business_columns(cleaned)
        st.success(f"Missing values filled using `{fill_strategy}` strategy.")
        st.rerun()

    after_df = st.session_state.datasets[active_name]["cleaned"]
    after = get_stats(after_df)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Rows After", after["rows"], delta=after["rows"] - before["rows"])
    a2.metric("Missing After", after["missing"], delta=after["missing"] - before["missing"])
    a3.metric("Duplicates After", after["duplicates"], delta=after["duplicates"] - before["duplicates"])
    a4.metric("Columns After", after["cols"])

    st.download_button(
        "⬇️ Export Cleaned Dataset (CSV)",
        data=after_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cleaned_{active_name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )
    st.caption("Exported file is ready for Power BI ingestion.")

# ══════════════════════════════ TAB 2 – COMPARISON ════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">⚖️ Dataset Comparison Dashboard</div>', unsafe_allow_html=True)
    if len(selected_for_compare) < 2:
        st.info("Select at least 2 datasets from sidebar to compare.")
    else:
        chosen = selected_for_compare[:2]
        dfa = st.session_state.datasets[chosen[0]]["cleaned"]
        dfb = st.session_state.datasets[chosen[1]]["cleaned"]
        det_a = st.session_state.datasets[chosen[0]]["detected"]
        det_b = st.session_state.datasets[chosen[1]]["detected"]

        ka = get_dataset_kpis(dfa, det_a)
        kb = get_dataset_kpis(dfb, det_b)
        comp_kpi = pd.DataFrame(
            [
                {"Dataset": chosen[0], "Total Hotels/Rows": ka["total_hotels"], "Avg Price": ka["avg_price"], "Avg Rating": ka["avg_rating"]},
                {"Dataset": chosen[1], "Total Hotels/Rows": kb["total_hotels"], "Avg Price": kb["avg_price"], "Avg Rating": kb["avg_rating"]},
            ]
        )
        st.dataframe(comp_kpi, use_container_width=True)

        common_numeric = sorted(set(infer_numeric_columns(dfa)).intersection(set(infer_numeric_columns(dfb))))
        if common_numeric:
            metric = st.selectbox("Metric for comparison", options=common_numeric)
            summary = pd.DataFrame(
                [
                    {"dataset": chosen[0], "mean": dfa[metric].mean(), "sum": dfa[metric].sum(), "count": dfa[metric].count()},
                    {"dataset": chosen[1], "mean": dfb[metric].mean(), "sum": dfb[metric].sum(), "count": dfb[metric].count()},
                ]
            )
            st.markdown("#### Mean / Sum / Count Differences")
            st.dataframe(summary, use_container_width=True)
            diff = summary.iloc[0][["mean", "sum", "count"]] - summary.iloc[1][["mean", "sum", "count"]]
            st.write({"difference_first_minus_second": diff.to_dict()})

            comb = pd.concat(
                [dfa[[metric]].assign(dataset=chosen[0]), dfb[[metric]].assign(dataset=chosen[1])],
                ignore_index=True,
            )
            fig_dist = px.histogram(comb, x=metric, color="dataset", barmode="overlay", opacity=0.65,
                                    template="plotly_dark", title=f"Distribution Comparison: {metric}")
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_mean = px.bar(summary, x="dataset", y="mean", color="dataset", template="plotly_dark",
                              title=f"Average {metric} by Dataset")
            st.plotly_chart(fig_mean, use_container_width=True)
        else:
            st.warning("No common numeric columns found between selected datasets.")

# ══════════════════════════════ TAB 3 – AI INSIGHTS ══════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🤖 AI Insights (Gemini)</div>', unsafe_allow_html=True)
    st.write(f"**Analyze dataset:** `{active_name}`")

    if genai is None:
        st.error("❌ google-generativeai not installed. Run: pip install google-generativeai")
        st.stop()

    if _gemini_model is None:
        st.error("❌ Gemini model could not be initialised. Check your API key.")
        st.stop()

    # ── Automatic Analysis ───────────────────────────────────────────────────
    st.subheader("📊 Automatic Analysis")

    sample_data = active_df.head(15).to_string()

    auto_prompt = f"""You are a professional business analyst.

Analyze this dataset and provide:
- Key insights
- Trends
- Problems
- Recommendations

Data:
{sample_data}
"""

    auto_insights_key = f"auto_insights_{active_name}"

    if auto_insights_key not in st.session_state:
        with st.spinner("Generating AI insights…"):
            try:
                auto_response = _gemini_model.generate_content(auto_prompt)
                st.session_state[auto_insights_key] = auto_response.text
            except Exception as e:
                st.session_state[auto_insights_key] = f"__ERROR__: {e}"

    cached = st.session_state[auto_insights_key]
    if cached.startswith("__ERROR__:"):
        st.error(cached.replace("__ERROR__: ", ""))
    else:
        st.success(cached)

    if st.button("🔄 Refresh Analysis"):
        if auto_insights_key in st.session_state:
            del st.session_state[auto_insights_key]
        st.rerun()

    # ── Chat with Data ───────────────────────────────────────────────────────
    st.subheader("💬 Chat with Data")

    user_question = st.text_input("Ask anything about your data:", key="chat_input")

    if st.button("Send", key="chat_send") and user_question.strip():
        st.session_state.chat_history.append(("You", user_question.strip()))

        chat_prompt = f"""Dataset:
{sample_data}

Question:
{user_question}

Answer clearly with insights.
"""
        with st.spinner("Thinking…"):
            try:
                response = _gemini_model.generate_content(chat_prompt)
                st.session_state.chat_history.append(("AI", response.text))
            except Exception as e:
                st.session_state.chat_history.append(("AI", f"Error: {e}"))

    for role, msg in reversed(st.session_state.chat_history):
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")

    # ── PDF Report ───────────────────────────────────────────────────────────
    st.subheader("📄 Generate Report")

    if st.button("Generate Full Report PDF"):
        insights_text = st.session_state.get(auto_insights_key, "No insights generated yet.")
        if insights_text.startswith("__ERROR__:"):
            st.error("Cannot generate PDF — AI insights not available.")
        else:
            try:
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer)
                styles = getSampleStyleSheet()
                elements = []

                elements.append(Paragraph("AI Business Report", styles["Title"]))
                elements.append(Spacer(1, 20))
                elements.append(Paragraph("Dataset Summary:", styles["Heading2"]))

                # Convert describe() to a safe string for ReportLab
                summary_text = active_df.describe().to_string().replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(f"<pre>{summary_text}</pre>", styles["Code"]))
                elements.append(Spacer(1, 20))

                elements.append(Paragraph("AI Insights:", styles["Heading2"]))
                safe_insights = insights_text.replace("<", "&lt;").replace(">", "&gt;")
                elements.append(Paragraph(safe_insights, styles["Normal"]))

                doc.build(elements)

                st.download_button(
                    label="📥 Download Report",
                    data=pdf_buffer.getvalue(),
                    file_name="AI_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")

# ══════════════════════════════ TAB 4 – POWER BI ═════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">📊 Suggested Power BI Dashboard</div>', unsafe_allow_html=True)
    suggestion = power_bi_suggestions(active_df, detected)
    st.write(f"Recommended dashboard type: **{suggestion['dashboard_type']}**")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Suggested KPIs")
        for item in suggestion["kpis"]:
            st.markdown(f"- {item}")
        st.markdown("#### Suggested Visuals")
        for item in suggestion["visuals"]:
            st.markdown(f"- {item}")
    with p2:
        st.markdown("#### Suggested DAX Measures")
        for item in suggestion["dax"]:
            st.code(item, language="DAX")
        st.markdown("#### Dataset Notes")
        for note in suggestion["notes"]:
            st.markdown(f"- {note}")
