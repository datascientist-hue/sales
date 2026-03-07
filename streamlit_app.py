from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "Primary5YearSales .parquet"
MAPPING_PATH = BASE_DIR / "mapping.xlsx"

NUMERIC_FIELDS = [
    "Qty in Nos",
    "Qty in Cases/Bags",
    "Qty in Ltrs/Kgs",
    "NR/Unit",
    "Net Rate",
    "Disc%",
    "Disc",
    "Net Value",
    "Applied Tax",
    "Inv Line Total",
    "MRP",
    "WeekNum",
]


def _financial_year_start(fy: str) -> int | None:
    if not isinstance(fy, str) or not fy.startswith("FY "):
        return None
    try:
        return int(fy[3:7])
    except Exception:
        return None


def _format_inr_value(value: float) -> str:
    if value >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:,.2f} L"
    return f"₹{value:,.0f}"


def _parse_inv_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    d1 = pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
    d2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    d3 = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d4 = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    out = d1.fillna(d2).fillna(d3).fillna(d4)
    return out.fillna(pd.to_datetime(s, dayfirst=True, errors="coerce"))


def _compute_group_metrics(df: pd.DataFrame, key_col: str, avg_col: str, yoy_col: str) -> pd.DataFrame:
    month_key = np.where(
        df["Month Num"].notna(),
        df["Financial Year"].astype(str) + "|" + df["Month Num"].astype("Int64").astype(str),
        np.nan,
    )
    tmp = df.copy()
    tmp["_month_key"] = month_key

    avg_df = (
        tmp.groupby(key_col, dropna=False)
        .agg(net_sum=("Net Value", "sum"), month_count=("_month_key", "nunique"))
        .reset_index()
    )
    avg_df[avg_col] = np.where(
        avg_df["month_count"] > 0,
        (avg_df["net_sum"] / avg_df["month_count"]).round(2),
        np.nan,
    )

    yoy_base = (
        tmp[tmp["Financial Year"] != "Unknown FY"]
        .groupby([key_col, "Financial Year"], dropna=False)["Net Value"]
        .sum()
        .reset_index()
    )
    yoy_base["_fy_start"] = yoy_base["Financial Year"].map(_financial_year_start)
    yoy_base = yoy_base.sort_values([key_col, "_fy_start"]) 
    yoy_base["_prev_val"] = yoy_base.groupby(key_col)["Net Value"].shift(1)
    yoy_base[yoy_col] = np.where(
        yoy_base["_prev_val"] > 0,
        (((yoy_base["Net Value"] - yoy_base["_prev_val"]) / yoy_base["_prev_val"]) * 100).round(2),
        np.nan,
    )

    merged = df.merge(avg_df[[key_col, avg_col]], on=key_col, how="left")
    merged = merged.merge(yoy_base[[key_col, "Financial Year", yoy_col]], on=[key_col, "Financial Year"], how="left")
    return merged


@st.cache_data(show_spinner=False)
def _load_category_mapping(mapping_mtime: float | None) -> pd.DataFrame:
    if mapping_mtime is None or not MAPPING_PATH.exists():
        return pd.DataFrame(columns=["Prod Ctg", "Ctg Wise"])

    try:
        mapping_df = pd.read_excel(MAPPING_PATH)
    except Exception:
        return pd.DataFrame(columns=["Prod Ctg", "Ctg Wise"])

    mapping_df.columns = [str(col).strip() for col in mapping_df.columns]
    if "Prod Ctg" not in mapping_df.columns or "Ctg Wise" not in mapping_df.columns:
        return pd.DataFrame(columns=["Prod Ctg", "Ctg Wise"])

    mapping_df = mapping_df[["Prod Ctg", "Ctg Wise"]].copy()
    mapping_df["Prod Ctg"] = mapping_df["Prod Ctg"].astype(str).str.strip()
    mapping_df["Ctg Wise"] = mapping_df["Ctg Wise"].astype(str).str.strip()
    mapping_df = mapping_df[mapping_df["Prod Ctg"].ne("")]
    mapping_df = mapping_df.drop_duplicates(subset=["Prod Ctg"], keep="first")
    return mapping_df


def _attach_category_mapping(df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Prod Ctg" not in out.columns:
        out["Ctg Wise"] = ""
        return out

    out["Prod Ctg"] = out["Prod Ctg"].fillna("").astype(str).str.strip()

    if mapping_df.empty:
        out["Ctg Wise"] = out["Prod Ctg"]
        return out

    out = out.merge(mapping_df, on="Prod Ctg", how="left")
    out["Ctg Wise"] = out["Ctg Wise"].fillna(out["Prod Ctg"]).astype(str).str.strip()
    return out


def _prepare_dataframe_from_parquet() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df.columns = [str(col).strip() for col in df.columns]

    valid_columns = [col for col in df.columns if col not in {"", "--"}]
    df = df[valid_columns]

    for col in NUMERIC_FIELDS:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "--": np.nan, "None": np.nan, "nan": np.nan})
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")

    if "Inv Date" in df.columns:
        df["Inv Date"] = _parse_inv_date(df["Inv Date"])
    else:
        df["Inv Date"] = pd.NaT

    fy_start = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_date_mask = df["Inv Date"].notna()
    fy_start.loc[valid_date_mask] = np.where(
        df.loc[valid_date_mask, "Inv Date"].dt.month >= 4,
        df.loc[valid_date_mask, "Inv Date"].dt.year,
        df.loc[valid_date_mask, "Inv Date"].dt.year - 1,
    )

    df["Financial Year"] = "Unknown FY"
    fy_mask = fy_start.notna()
    fy_int = fy_start[fy_mask].astype(int)
    df.loc[fy_mask, "Financial Year"] = "FY " + fy_int.astype(str) + "-" + (fy_int + 1).astype(str).str[-2:]

    df["Month"] = np.where(df["Inv Date"].notna(), df["Inv Date"].dt.strftime("%b"), "")
    df["Month Num"] = np.where(df["Inv Date"].notna(), df["Inv Date"].dt.month, np.nan)

    if "DB" not in df.columns:
        df["DB"] = ""
    if "Depot" in df.columns:
        df["DB"] = df["DB"].fillna("").replace("", np.nan).fillna(df["Depot"].fillna(""))
    else:
        df["DB"] = df["DB"].fillna("")

    qty = df.get("Qty in Ltrs/Kgs", pd.Series(0, index=df.index)).fillna(0)
    net_value = df.get("Net Value", pd.Series(0, index=df.index)).fillna(0)
    inv_total = df.get("Inv Line Total", pd.Series(0, index=df.index)).fillna(0)

    df["Volume (Tonnes)"] = np.where(qty > 0, (qty / 1000).round(3), 0)
    df["Per Litre Value"] = np.where(qty > 0, (net_value / qty).round(2), 0)
    df["Per Litre Cost"] = np.where(qty > 0, (inv_total / qty).round(2), 0)

    df = _compute_group_metrics(df, "SKU", "Avg Monthly Sale (SKU)", "YoY Growth % (SKU)")
    df = _compute_group_metrics(df, "Prod Ctg", "Avg Monthly Sale (Prod Ctg)", "YoY Growth % (Prod Ctg)")
    df = _compute_group_metrics(df, "DB", "Avg Monthly Sale (DB)", "YoY Growth % (DB)")

    return df


def _compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    qty = df.get("Qty in Ltrs/Kgs", pd.Series(0, index=df.index)).fillna(0)
    net_value = df.get("Net Value", pd.Series(0, index=df.index)).fillna(0)
    inv_total = df.get("Inv Line Total", pd.Series(0, index=df.index)).fillna(0)

    total_qty = float(qty.sum())
    total_net = float(net_value.sum())
    total_inv = float(inv_total.sum())

    total_volume = round(total_qty / 1000, 2)
    avg_per_litre_value = round(total_net / total_qty, 2) if total_qty > 0 else 0
    avg_per_litre_cost = round(total_inv / total_qty, 2) if total_qty > 0 else 0

    vol_by_fy = df.groupby("Financial Year", dropna=False)["Volume (Tonnes)"].sum()
    if len(vol_by_fy) > 0:
        best_fy = str(vol_by_fy.idxmax())
        peak_volume = round(float(vol_by_fy.max()), 2)
    else:
        best_fy = "—"
        peak_volume = 0.0

    net_by_fy = (
        df[df["Financial Year"] != "Unknown FY"]
        .groupby("Financial Year", dropna=False)["Net Value"]
        .sum()
    )
    fy_order = sorted(net_by_fy.index.tolist(), key=lambda x: _financial_year_start(str(x)) or 0)

    yearly_growth: list[dict[str, Any]] = []
    for idx in range(1, len(fy_order)):
        prev_fy = fy_order[idx - 1]
        cur_fy = fy_order[idx]
        prev_val = float(net_by_fy.get(prev_fy, 0))
        cur_val = float(net_by_fy.get(cur_fy, 0))
        if prev_val <= 0:
            continue
        growth = round(((cur_val - prev_val) / prev_val) * 100, 2)
        yearly_growth.append({"previousFY": prev_fy, "fy": cur_fy, "growthPct": growth})

    cagr_pct = None
    if len(fy_order) >= 2:
        first_val = float(net_by_fy.get(fy_order[0], 0))
        last_val = float(net_by_fy.get(fy_order[-1], 0))
        periods = len(fy_order) - 1
        if first_val > 0 and last_val > 0 and periods > 0:
            cagr_pct = round(((last_val / first_val) ** (1 / periods) - 1) * 100, 2)

    cagr_by_year: list[dict[str, Any]] = []
    if len(fy_order) >= 2:
        base_fy = fy_order[0]
        base_val = float(net_by_fy.get(base_fy, 0))
        if base_val > 0:
            for idx in range(1, len(fy_order)):
                to_fy = fy_order[idx]
                target_val = float(net_by_fy.get(to_fy, 0))
                periods = idx
                if target_val > 0 and periods > 0:
                    cagr_by_year.append(
                        {
                            "fromFY": base_fy,
                            "toFY": to_fy,
                            "cagrPct": round(((target_val / base_val) ** (1 / periods) - 1) * 100, 2),
                        }
                    )

    return {
        "totalRows": int(len(df)),
        "bestFY": best_fy,
        "peakVolume": peak_volume,
        "totalVolume": total_volume,
        "totalNetValue": _format_inr_value(total_net),
        "avgPerLitreValue": avg_per_litre_value,
        "avgPerLitreCost": avg_per_litre_cost,
        "yearlyGrowth": yearly_growth,
        "cagrPct": cagr_pct,
        "cagrByYear": cagr_by_year,
    }


@st.cache_data(show_spinner=False)
def load_data(parquet_mtime: float):
    if parquet_mtime <= 0:
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    df = _prepare_dataframe_from_parquet()
    kpis = _compute_kpis(df)
    return df, kpis, "parquet"


def _chunk(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


_CARD_COLORS = [
    "#0078d4",  # Microsoft blue
    "#107c10",  # green
    "#ff8c00",  # orange
    "#e3008c",  # pink/magenta
    "#00b294",  # teal
    "#8764b8",  # purple
    "#038387",  # dark teal
    "#ca5010",  # rust
    "#004e8c",  # dark blue
    "#498205",  # olive green
    "#c239b3",  # orchid
    "#0099bc",  # cerulean
]

_PBI_CSS = """
<style>
/* ══════════════════════════════════════════════
   FORCE LIGHT MODE — override system dark theme
   ══════════════════════════════════════════════ */
:root {
    color-scheme: light !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, .block-container,
[class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background-color: #f3f2f1 !important;
    color: #201f1e !important;
}

/* ── Native Streamlit header — transparent, keep ⋮ menu button ── */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    border-bottom: none !important;
}
/* Make Deploy / ⋮ button text dark so it's readable */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] svg {
    color: #201f1e !important;
    fill: #201f1e !important;
}
/* Tight top padding below the fixed header */
.block-container {
    padding-top: 1rem !important;
}
/* Hide only the footer watermark */
footer { visibility: hidden; }

/* ── All input / select widget backgrounds → white ── */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="select"],
[data-baseweb="popover"],
[role="listbox"],
[data-baseweb="menu"],
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div,
[data-testid="stMultiSelect"] > div,
[data-testid="stSelectbox"] > div {
    background-color: #ffffff !important;
    color: #201f1e !important;
    border-color: #c8c6c4 !important;
}

/* Input text color */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="select"] input {
    color: #201f1e !important;
    background-color: #ffffff !important;
}

/* Dropdown list items */
[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="list-item"] {
    background-color: #ffffff !important;
    color: #201f1e !important;
}
[role="option"]:hover,
[data-baseweb="menu"] li:hover {
    background-color: #deecf9 !important;
    color: #0078d4 !important;
}

/* All labels */
label, .stSelectbox label, .stMultiSelect label,
[data-testid="stWidgetLabel"] {
    color: #323130 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Caption / small text */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: #605e5c !important;
}

/* ── Main content area ── */
.block-container {
    padding-top: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
    background-color: #f3f2f1 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #ffffff !important;
    border-right: 1px solid #e1dfdd !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] hr {
    border-color: #e1dfdd !important;
    margin: 6px 0 12px 0;
}
.sidebar-header {
    background: #0078d4;
    color: #ffffff !important;
    padding: 13px 18px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: -1rem -1rem 16px -1rem;
}

/* ── Dashboard header bar ── */
.pbi-header {
    background: #ffffff;
    color: #201f1e;
    padding: 12px 20px;
    border-radius: 4px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-left: 5px solid #0078d4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09);
}
.pbi-header-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #0078d4 !important;
}
.pbi-header-meta {
    font-size: 0.75rem;
    color: #605e5c;
    text-align: right;
    line-height: 1.6;
}
.pbi-header-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6bb700;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── Section heading ── */
.pbi-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #605e5c;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    border-bottom: 2px solid #0078d4;
    padding-bottom: 5px;
    margin: 22px 0 14px 0;
}

/* ── KPI card tiles ── */
.pbi-card {
    background: #ffffff !important;
    border-radius: 3px;
    padding: 14px 16px 12px 16px;
    border-top: 4px solid #0078d4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.09), 0 1px 2px rgba(0,0,0,0.05);
    min-height: 88px;
    transition: box-shadow 0.15s;
}
.pbi-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.13); }
.pbi-card-label {
    font-size: 0.67rem;
    font-weight: 600;
    color: #605e5c !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pbi-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #201f1e !important;
    line-height: 1.1;
    word-break: break-word;
}
.pbi-card-positive { color: #107c10 !important; font-size: 1.35rem; font-weight: 700; }
.pbi-card-negative { color: #a4262c !important; font-size: 1.35rem; font-weight: 700; }

/* ── Pivot config bar ── */
.pbi-pivot-cfg {
    background: #f8f7f6 !important;
    border: 1px solid #e1dfdd;
    border-radius: 3px;
    padding: 12px 16px 4px 16px;
    margin-bottom: 12px;
}

/* ── Dataframe table ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] iframe {
    border: 1px solid #e1dfdd !important;
    border-radius: 2px;
    background: #ffffff !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background-color: #deecf9 !important;
    color: #0078d4 !important;
    border-radius: 2px !important;
}
[data-baseweb="tag"] span { color: #0078d4 !important; }

/* ── Export / download button ── */
.stDownloadButton > button {
    background-color: #0078d4 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 6px 18px !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.stDownloadButton > button:hover { background-color: #106ebe !important; }

/* ── Warning / info / error boxes ── */
[data-testid="stAlert"] {
    background-color: #fff4ce !important;
    color: #323130 !important;
    border-left-color: #ffaa44 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #0078d4 !important; font-weight: 600; }
</style>
"""


def _kpi_card_html(label: str, value: str, color: str) -> str:
    is_pct = "%" in value
    is_negative = is_pct and value.strip().startswith("-")
    val_class = "pbi-card-negative" if is_negative else ("pbi-card-positive" if is_pct else "pbi-card-value")
    if is_pct:
        value_html = f'<div class="{val_class}" style="font-size:1.35rem;font-weight:700;">{value}</div>'
    else:
        value_html = f'<div class="pbi-card-value">{value}</div>'
    return (
        f'<div class="pbi-card" style="border-top-color:{color};">'
        f'<div class="pbi-card-label">{label}</div>'
        f'{value_html}'
        f'</div>'
    )


def _section_title_html(title: str, icon: str = "") -> str:
    prefix = f"{icon}&nbsp;&nbsp;" if icon else ""
    return f'<div class="pbi-section-title">{prefix}{title}</div>'


def main():
    st.set_page_config(
        page_title="Primary Sales Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject Power BI CSS
    st.markdown(_PBI_CSS, unsafe_allow_html=True)

    if not PARQUET_PATH.exists():
        st.error(f"Parquet data file not found: {PARQUET_PATH}")
        st.stop()

    parquet_mtime = PARQUET_PATH.stat().st_mtime
    mapping_mtime = MAPPING_PATH.stat().st_mtime if MAPPING_PATH.exists() else None

    with st.spinner("Loading data…"):
        df, kpis, source = load_data(parquet_mtime)

    mapping_df = _load_category_mapping(mapping_mtime)
    df = _attach_category_mapping(df, mapping_df)

    # ── Sidebar: filters ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔽&nbsp; Filters</div>', unsafe_allow_html=True)

        dsm_col = next((col for col in df.columns if str(col).strip().lower() == "dsm"), None)
        normalized_col_map = {
            "".join(ch for ch in str(col).strip().lower() if ch.isalnum()): col
            for col in df.columns
        }
        db_code_col = (
            normalized_col_map.get("dbcode")
            or normalized_col_map.get("db")
            or normalized_col_map.get("depot")
        )
        fy_options = (
            [x for x in sorted(df["Financial Year"].dropna().unique().tolist()) if x != ""]
            if "Financial Year" in df.columns else []
        )
        state_options = sorted(df["State"].dropna().astype(str).unique().tolist()) if "State" in df.columns else []
        db_code_options = (
            sorted(df[db_code_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
            if db_code_col
            else []
        )
        dsm_options = sorted(df[dsm_col].dropna().astype(str).unique().tolist()) if dsm_col else []
        prod_options = sorted(df["Prod Ctg"].dropna().astype(str).unique().tolist()) if "Prod Ctg" in df.columns else []
        ctg_wise_options = sorted(df["Ctg Wise"].dropna().astype(str).unique().tolist()) if "Ctg Wise" in df.columns else []

        selected_fy = st.multiselect("Financial Year", fy_options, default=[])
        st.markdown("<hr>", unsafe_allow_html=True)
        selected_state = st.multiselect("State", state_options, default=[])
        st.markdown("<hr>", unsafe_allow_html=True)
        db_label = "DB Code" if normalized_col_map.get("dbcode") else "DB Code (from DB/Depot)"
        selected_db_code = st.multiselect(db_label, db_code_options, default=[])
        st.markdown("<hr>", unsafe_allow_html=True)
        selected_dsm = st.multiselect("DSM", dsm_options, default=[])
        st.markdown("<hr>", unsafe_allow_html=True)
        selected_prod = st.multiselect("Prod Category", prod_options, default=[])
        st.markdown("<hr>", unsafe_allow_html=True)
        selected_ctg_wise = st.multiselect("Category Wise", ctg_wise_options, default=[])

        st.markdown("<br>", unsafe_allow_html=True)
        n_active = sum([
            bool(selected_fy), bool(selected_state),
            bool(selected_db_code),
            bool(selected_dsm),
            bool(selected_prod), bool(selected_ctg_wise),
        ])
        if n_active:
            st.caption(f"{n_active} filter(s) active")
        else:
            st.caption("No filters applied — showing all data")

    # Apply filters
    filtered_df = df.copy()
    if selected_fy:
        filtered_df = filtered_df[filtered_df["Financial Year"].isin(selected_fy)]
    if selected_state:
        filtered_df = filtered_df[filtered_df["State"].astype(str).isin(selected_state)]
    if selected_db_code and db_code_col:
        filtered_df = filtered_df[filtered_df[db_code_col].astype(str).str.strip().isin(selected_db_code)]
    if selected_dsm and dsm_col:
        filtered_df = filtered_df[filtered_df[dsm_col].astype(str).isin(selected_dsm)]
    if selected_prod:
        filtered_df = filtered_df[filtered_df["Prod Ctg"].astype(str).isin(selected_prod)]
    if selected_ctg_wise:
        filtered_df = filtered_df[filtered_df["Ctg Wise"].astype(str).isin(selected_ctg_wise)]

    # Recompute KPIs on filtered data so cards always reflect current selection
    flt_kpis = _compute_kpis(filtered_df) if n_active else kpis

    # ── Header bar ───────────────────────────────────────────────────────────
    rows_shown = f"{len(filtered_df):,}"
    total_rows = f"{kpis['totalRows']:,}"
    st.markdown(
        f"""
        <div class="pbi-header">
            <div class="pbi-header-title">📊&nbsp;&nbsp;Primary Sales — 5-Year Analysis</div>
            <div class="pbi-header-meta">
                <span class="pbi-header-dot"></span>Live&nbsp;&nbsp;|&nbsp;&nbsp;
                Rows:&nbsp;<strong>{rows_shown}</strong> / {total_rows}&nbsp;&nbsp;|&nbsp;&nbsp;
                Source:&nbsp;{source.upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(_section_title_html("Key Performance Indicators", "📌"), unsafe_allow_html=True)

    card_items: list[tuple[str, str]] = [
        ("Best FY by Volume", str(flt_kpis["bestFY"])),
        ("Peak Volume (T)", f"{flt_kpis['peakVolume']:,.2f}"),
        ("Total Volume (T)", f"{flt_kpis['totalVolume']:,.2f}"),
        ("Total Net Value", str(flt_kpis["totalNetValue"])),
        ("Avg / Litre Value", f"₹{flt_kpis['avgPerLitreValue']:,.2f}"),
        ("Avg / Litre Cost", f"₹{flt_kpis['avgPerLitreCost']:,.2f}"),
    ]

    if flt_kpis.get("cagrByYear"):
        for cagr_item in flt_kpis["cagrByYear"]:
            card_items.append(
                (f"CAGR {cagr_item['fromFY']}→{cagr_item['toFY']}", f"{cagr_item['cagrPct']:,.2f}%")
            )
    else:
        cagr_fallback = "—" if flt_kpis.get("cagrPct") is None else f"{flt_kpis['cagrPct']:,.2f}%"
        card_items.append(("CAGR % (Net Value)", cagr_fallback))

    for growth_item in flt_kpis.get("yearlyGrowth", []):
        card_items.append(
            (f"Growth {growth_item['fy']}", f"{growth_item['growthPct']:,.2f}%")
        )

    for row_items in _chunk(card_items, 6):
        cols = st.columns(len(row_items))
        for idx, (label, value) in enumerate(row_items):
            color = _CARD_COLORS[card_items.index((label, value)) % len(_CARD_COLORS)]
            cols[idx].markdown(_kpi_card_html(label, value, color), unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

    tabs = st.tabs(["Pivot Table", "Trend Chart"])

    with tabs[0]:
        # ── Pivot Table ───────────────────────────────────────────────────────
        st.markdown(_section_title_html("Pivot Table", "📋"), unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="pbi-pivot-cfg">', unsafe_allow_html=True)
            pivot_cfg = st.columns([3, 2, 3, 2])
            all_columns = filtered_df.columns.tolist()
            numeric_columns = [c for c in all_columns if pd.api.types.is_numeric_dtype(filtered_df[c])]
            dimension_columns = [c for c in all_columns if c not in numeric_columns]

            pivot_index = pivot_cfg[0].multiselect(
                "Rows",
                dimension_columns,
                default=[c for c in ["Financial Year", "Prod Ctg"] if c in dimension_columns],
            )
            pivot_columns = pivot_cfg[1].selectbox("Columns", ["(None)"] + dimension_columns, index=0)
            default_values = [c for c in ["Net Value", "Volume (Tonnes)"] if c in numeric_columns]
            if not default_values and numeric_columns:
                default_values = [numeric_columns[0]]
            pivot_values = pivot_cfg[2].multiselect("Values", numeric_columns, default=default_values)
            agg_name = pivot_cfg[3].selectbox("Aggregation", ["sum", "mean", "count", "min", "max"], index=0)
            st.markdown("</div>", unsafe_allow_html=True)

        pivot_col_arg = None if pivot_columns == "(None)" else pivot_columns
        try:
            if pivot_index and pivot_values:
                pivot_df = pd.pivot_table(
                    filtered_df,
                    index=pivot_index,
                    columns=pivot_col_arg,
                    values=pivot_values,
                    aggfunc=agg_name,
                    fill_value=0,
                )
            elif pivot_values:
                summary_data: dict[str, list[float]] = {}
                for metric in pivot_values:
                    series = filtered_df[metric]
                    agg_result = getattr(series, agg_name)() if hasattr(series, agg_name) else series.sum()
                    summary_data[f"{agg_name}({metric})"] = [agg_result]
                pivot_df = pd.DataFrame(summary_data)
            else:
                pivot_df = pd.DataFrame()
        except Exception as error:
            st.error(f"Failed to build pivot table: {error}")
            pivot_df = pd.DataFrame()

        if isinstance(pivot_df.columns, pd.MultiIndex):
            if pivot_col_arg is not None and len(pivot_values) > 1 and pivot_df.columns.nlevels >= 2:
                pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
                pivot_df = pivot_df.sort_index(axis=1, level=0)
            pivot_df.columns = [
                " | ".join([str(x) for x in col if str(x) != ""]).strip()
                for col in pivot_df.columns.to_flat_index()
            ]

        pivot_display = pivot_df.reset_index() if not pivot_df.empty else pd.DataFrame()

        max_cells = 180_000
        if not pivot_display.empty and pivot_display.shape[0] * max(pivot_display.shape[1], 1) > max_cells:
            st.warning("Pivot result is large — showing first 10,000 rows.")
            st.dataframe(pivot_display.head(10_000), use_container_width=True, height=480)
        elif not pivot_display.empty:
            st.dataframe(pivot_display, use_container_width=True, height=480)
        else:
            st.info("Select at least one Row and one Value to build the pivot table.")

        col_export, col_info = st.columns([2, 8])
        with col_export:
            st.download_button(
                label="⬇ Export Pivot CSV",
                data=pivot_display.to_csv(index=False).encode("utf-8"),
                file_name="primary-sales-pivot-export.csv",
                mime="text/csv",
                disabled=pivot_display.empty,
            )
        with col_info:
            if not pivot_display.empty:
                st.caption(
                    f"{pivot_display.shape[0]:,} rows × {pivot_display.shape[1]:,} columns"
                )

    with tabs[1]:
        st.markdown(_section_title_html("DSM Wise Trend Chart", "📈"), unsafe_allow_html=True)

        if dsm_col is None:
            st.info("DSM column not found in data, so trend chart is unavailable.")
        elif "Inv Date" not in filtered_df.columns:
            st.info("Invoice date column is missing, so trend chart is unavailable.")
        else:
            metric_options = [c for c in ["Net Value", "Volume (Tonnes)", "Qty in Ltrs/Kgs"] if c in filtered_df.columns]
            if not metric_options:
                st.info("No numeric metric available for trend chart.")
            else:
                dsm_cfg_cols = st.columns([2, 2])
                metric_col = dsm_cfg_cols[0].selectbox("Trend Metric", metric_options, index=0, key="dsm_metric")
                time_grain = dsm_cfg_cols[1].selectbox("View By", ["Year", "Month"], index=0, key="dsm_view_by")

                chart_base = filtered_df[["Inv Date", dsm_col, metric_col]].copy()
                chart_base = chart_base.dropna(subset=["Inv Date"])
                chart_base[dsm_col] = chart_base[dsm_col].astype(str).str.strip()
                chart_base = chart_base[chart_base[dsm_col] != ""]

                if chart_base.empty:
                    st.info("No dated records available for DSM trend chart with current filters.")
                else:
                    dsm_totals = (
                        chart_base.groupby(dsm_col, dropna=False)[metric_col]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    default_dsm = dsm_totals.head(5).index.tolist()
                    selected_chart_dsms = st.multiselect(
                        "DSM for Trend",
                        options=dsm_totals.index.tolist(),
                        default=default_dsm,
                    )

                    if selected_chart_dsms:
                        chart_base = chart_base[chart_base[dsm_col].isin(selected_chart_dsms)]

                    if time_grain == "Year":
                        chart_base["Time Axis"] = np.where(
                            chart_base["Inv Date"].dt.month >= 4,
                            "FY " + chart_base["Inv Date"].dt.year.astype(str) + "-" + (chart_base["Inv Date"].dt.year + 1).astype(str).str[-2:],
                            "FY " + (chart_base["Inv Date"].dt.year - 1).astype(str) + "-" + chart_base["Inv Date"].dt.year.astype(str).str[-2:],
                        )
                        trend_df = (
                            chart_base.groupby(["Time Axis", dsm_col], dropna=False)[metric_col]
                            .sum()
                            .reset_index()
                        )
                        trend_df["_fy_start"] = trend_df["Time Axis"].map(_financial_year_start)
                        trend_df = trend_df.sort_values(["_fy_start", dsm_col]).drop(columns=["_fy_start"])
                    else:
                        chart_base["Time Axis"] = chart_base["Inv Date"].dt.to_period("M").dt.to_timestamp()
                        trend_df = (
                            chart_base.groupby(["Time Axis", dsm_col], dropna=False)[metric_col]
                            .sum()
                            .reset_index()
                            .sort_values(["Time Axis", dsm_col])
                        )

                    trend_pivot = trend_df.pivot(index="Time Axis", columns=dsm_col, values=metric_col).sort_index()

                    if trend_pivot.empty:
                        st.info("No trend data to plot for the selected DSM values.")
                    else:
                        dsm_x_label = "Financial Year" if time_grain == "Year" else "Month"
                        st.line_chart(trend_pivot, use_container_width=True)
                        st.caption(f"Chart Labels → X-axis: {dsm_x_label} | Y-axis: {metric_col} | Series: DSM")
                        st.caption(f"{time_grain}ly trend by DSM for {metric_col}.")

        st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)
        st.markdown(_section_title_html("Product Category Wise Trend Chart", "📈"), unsafe_allow_html=True)

        if "Prod Ctg" not in filtered_df.columns:
            st.info("Product Category column not found in data, so trend chart is unavailable.")
        elif "Inv Date" not in filtered_df.columns:
            st.info("Invoice date column is missing, so trend chart is unavailable.")
        else:
            metric_options = [c for c in ["Net Value", "Volume (Tonnes)", "Qty in Ltrs/Kgs"] if c in filtered_df.columns]
            if not metric_options:
                st.info("No numeric metric available for trend chart.")
            else:
                prod_cfg_cols = st.columns([2, 2])
                metric_col = prod_cfg_cols[0].selectbox("Trend Metric", metric_options, index=0, key="prod_metric")
                prod_time_grain = prod_cfg_cols[1].selectbox("View By", ["Month", "Year"], index=0, key="prod_view_by")

                chart_base = filtered_df[["Inv Date", "Prod Ctg", metric_col]].copy()
                chart_base = chart_base.dropna(subset=["Inv Date"])
                chart_base["Prod Ctg"] = chart_base["Prod Ctg"].astype(str).str.strip()
                chart_base = chart_base[chart_base["Prod Ctg"] != ""]

                if chart_base.empty:
                    st.info("No dated records available for Product Category trend chart with current filters.")
                else:
                    prod_totals = (
                        chart_base.groupby("Prod Ctg", dropna=False)[metric_col]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    default_prod = prod_totals.head(5).index.tolist()
                    selected_prod_trend = st.multiselect(
                        "Product Category for Trend",
                        options=prod_totals.index.tolist(),
                        default=default_prod,
                        key="prod_trend_select",
                    )

                    if selected_prod_trend:
                        chart_base = chart_base[chart_base["Prod Ctg"].isin(selected_prod_trend)]

                    if prod_time_grain == "Year":
                        chart_base["Time Axis"] = np.where(
                            chart_base["Inv Date"].dt.month >= 4,
                            "FY " + chart_base["Inv Date"].dt.year.astype(str) + "-" + (chart_base["Inv Date"].dt.year + 1).astype(str).str[-2:],
                            "FY " + (chart_base["Inv Date"].dt.year - 1).astype(str) + "-" + chart_base["Inv Date"].dt.year.astype(str).str[-2:],
                        )
                        trend_df = (
                            chart_base.groupby(["Time Axis", "Prod Ctg"], dropna=False)[metric_col]
                            .sum()
                            .reset_index()
                        )
                        trend_df["_fy_start"] = trend_df["Time Axis"].map(_financial_year_start)
                        trend_df = trend_df.sort_values(["_fy_start", "Prod Ctg"]).drop(columns=["_fy_start"])
                    else:
                        chart_base["Time Axis"] = chart_base["Inv Date"].dt.to_period("M").dt.to_timestamp()
                        trend_df = (
                            chart_base.groupby(["Time Axis", "Prod Ctg"], dropna=False)[metric_col]
                            .sum()
                            .reset_index()
                            .sort_values(["Time Axis", "Prod Ctg"])
                        )

                    trend_pivot = trend_df.pivot(index="Time Axis", columns="Prod Ctg", values=metric_col).sort_index()

                    if trend_pivot.empty:
                        st.info("No trend data to plot for the selected Product Categories.")
                    else:
                        prod_x_label = "Financial Year" if prod_time_grain == "Year" else "Month"
                        st.line_chart(trend_pivot, use_container_width=True)
                        st.caption(f"Chart Labels → X-axis: {prod_x_label} | Y-axis: {metric_col} | Series: Product Category")
                        st.caption(f"{prod_time_grain}ly trend by Product Category for {metric_col}.")


if __name__ == "__main__":
    main()
