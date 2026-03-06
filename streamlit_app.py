from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Primary5YearSales.csv"
PARQUET_PATH = BASE_DIR / "Primary5YearSales.parquet"
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


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    read_attempts = [
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "utf-8-sig", "low_memory": False},
        {"encoding": "cp1252", "low_memory": False},
        {"encoding": "latin1", "low_memory": False},
    ]

    last_error: Exception | None = None
    for params in read_attempts:
        try:
            return pd.read_csv(path, **params)
        except UnicodeDecodeError as error:
            last_error = error

    raw_text = path.read_bytes().decode("latin1", errors="replace").replace("\xa0", " ")
    return pd.read_csv(io.StringIO(raw_text), low_memory=False)


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


def _prepare_dataframe_from_csv() -> pd.DataFrame:
    df = _read_csv_with_fallbacks(CSV_PATH)
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
def load_data(csv_mtime: float, parquet_mtime: float | None):
    source = "parquet"
    if PARQUET_PATH.exists() and parquet_mtime is not None and parquet_mtime >= csv_mtime:
        df = pd.read_parquet(PARQUET_PATH)
    else:
        df = _prepare_dataframe_from_csv()
        df.to_parquet(PARQUET_PATH, index=False)
        source = "csv"

    kpis = _compute_kpis(df)
    return df, kpis, source


def _chunk(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main():
    st.set_page_config(page_title="Primary Sales Dashboard", layout="wide")
    st.title("📊 Primary Sales — 5-Year Analysis")
    st.caption("Pivot table + filters + CSV export")

    if not CSV_PATH.exists():
        st.error(f"Data file not found: {CSV_PATH}")
        st.stop()

    csv_mtime = CSV_PATH.stat().st_mtime
    parquet_mtime = PARQUET_PATH.stat().st_mtime if PARQUET_PATH.exists() else None
    mapping_mtime = MAPPING_PATH.stat().st_mtime if MAPPING_PATH.exists() else None

    with st.spinner("Loading data..."):
        df, kpis, source = load_data(csv_mtime, parquet_mtime)

    mapping_df = _load_category_mapping(mapping_mtime)
    df = _attach_category_mapping(df, mapping_df)

    st.caption(f"Loaded {kpis['totalRows']:,} rows (source: {source.upper()})")

    card_items = [
        ("Best FY by Volume", str(kpis["bestFY"])),
        ("Peak Volume (Tonnes)", f"{kpis['peakVolume']:,.2f}"),
        ("Total Volume (Tonnes)", f"{kpis['totalVolume']:,.2f}"),
        ("Total Net Value", str(kpis["totalNetValue"])),
        ("Avg Per Litre Value", f"₹{kpis['avgPerLitreValue']:,.2f}"),
        ("Avg Per Litre Cost", f"₹{kpis['avgPerLitreCost']:,.2f}"),
    ]

    if kpis.get("cagrByYear"):
        for cagr_item in kpis["cagrByYear"]:
            card_items.append(
                (
                    f"CAGR {cagr_item['fromFY']} to {cagr_item['toFY']}",
                    f"{cagr_item['cagrPct']:,.2f}%",
                )
            )
    else:
        cagr_fallback = "—" if kpis.get("cagrPct") is None else f"{kpis['cagrPct']:,.2f}%"
        card_items.append(("CAGR % (Net Value)", cagr_fallback))

    for growth_item in kpis.get("yearlyGrowth", []):
        card_items.append(
            (
                f"Growth {growth_item['fy']} vs {growth_item['previousFY']}",
                f"{growth_item['growthPct']:,.2f}%",
            )
        )

    for row in _chunk(card_items, 6):
        cols = st.columns(len(row))
        for idx, (label, value) in enumerate(row):
            cols[idx].metric(label, value)

    st.subheader("Filters")
    filter_cols = st.columns(4)
    fy_options = [x for x in sorted(df["Financial Year"].dropna().unique().tolist()) if x != ""] if "Financial Year" in df.columns else []
    state_options = sorted(df["State"].dropna().astype(str).unique().tolist()) if "State" in df.columns else []
    prod_options = sorted(df["Prod Ctg"].dropna().astype(str).unique().tolist()) if "Prod Ctg" in df.columns else []
    ctg_wise_options = sorted(df["Ctg Wise"].dropna().astype(str).unique().tolist()) if "Ctg Wise" in df.columns else []

    selected_fy = filter_cols[0].multiselect("Financial Year", fy_options, default=[])
    selected_state = filter_cols[1].multiselect("State", state_options, default=[])
    selected_prod = filter_cols[2].multiselect("Prod Ctg", prod_options, default=[])
    selected_ctg_wise = filter_cols[3].multiselect("Ctg Wise", ctg_wise_options, default=[])

    filtered_df = df.copy()
    if selected_fy:
        filtered_df = filtered_df[filtered_df["Financial Year"].isin(selected_fy)]
    if selected_state:
        filtered_df = filtered_df[filtered_df["State"].astype(str).isin(selected_state)]
    if selected_prod:
        filtered_df = filtered_df[filtered_df["Prod Ctg"].astype(str).isin(selected_prod)]
    if selected_ctg_wise:
        filtered_df = filtered_df[filtered_df["Ctg Wise"].astype(str).isin(selected_ctg_wise)]

    st.subheader("Pivot Table")
    pivot_cfg = st.columns(4)
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
    pivot_values = pivot_cfg[2].multiselect(
        "Values",
        numeric_columns,
        default=default_values,
    )
    agg_name = pivot_cfg[3].selectbox("Aggregation", ["sum", "mean", "count", "min", "max"], index=0)

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
        st.warning("Pivot result is large, showing first 10,000 rows to keep UI responsive.")
        st.dataframe(pivot_display.head(10_000), use_container_width=True, height=450)
    else:
        st.dataframe(pivot_display, use_container_width=True, height=450)

    st.download_button(
        label="Export Pivot CSV",
        data=pivot_display.to_csv(index=False).encode("utf-8"),
        file_name="primary-sales-pivot-export.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
