"""Retail sales analysis: clean the raw orders, answer business questions, save charts.

Run:  python generate_data.py   (once, creates data/raw_orders.csv)
      python analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw_orders.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"figure.figsize": (9, 5), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 11})
ACCENT = "#2f6fed"


# --------------------------------------------------------------------------
# 1. Load + audit
# --------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    """Count the data-quality problems before touching anything."""
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_city": int(df["city"].isna().sum()),
        "non_positive_quantity": int((df["quantity"] <= 0).sum()),
        "inconsistent_category_labels": int(
            (df["category"] != df["category"].str.strip().str.title()).sum()),
        "slash_format_dates": int(df["order_date"].astype(str).str.contains("/").sum()),
    }


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    # Dates arrive in two formats: ISO (2025-04-25) and day-first (25/04/2025)
    s = df["order_date"].astype(str)
    iso = pd.to_datetime(s.where(~s.str.contains("/")), format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s.where(s.str.contains("/")), format="%d/%m/%Y", errors="coerce")
    df["order_date"] = iso.fillna(dmy)

    df["category"] = df["category"].str.strip().str.title()
    df["city"] = df["city"].fillna("Unknown")

    # Negative quantities look like returns / entry errors: exclude from sales analysis
    df = df[df["quantity"] > 0]

    df["revenue"] = df["unit_price"] * df["quantity"] * (1 - df["discount"])
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["order_date"].dt.year
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Analysis + charts
# --------------------------------------------------------------------------
def fmt_try(x):
    return f"{x:,.0f} TRY"


def main():
    raw = pd.read_csv(RAW)
    quality = audit(raw)
    df = clean(raw)
    findings = []

    # ---- KPIs
    total_rev = df["revenue"].sum()
    n_orders = df["order_id"].nunique()
    aov = total_rev / n_orders
    n_cust = df["customer_id"].nunique()
    order_counts = df.groupby("customer_id")["order_id"].nunique()
    median_orders = order_counts.median()
    findings.append(f"- **Total revenue:** {fmt_try(total_rev)} across {n_orders:,} orders "
                    f"and {n_cust:,} customers.")
    findings.append(f"- **Average order value:** {fmt_try(aov)}.")
    findings.append(f"- **Orders per customer:** the median customer placed {median_orders:.0f} orders "
                    f"(most active customer: {order_counts.max()}).")

    # ---- Monthly revenue trend
    monthly = df.groupby("month")["revenue"].sum()
    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values / 1000, color=ACCENT, marker="o", lw=2)
    ax.set_title("Monthly revenue")
    ax.set_ylabel("Revenue (thousand TRY)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "monthly_revenue.png", dpi=150)
    plt.close(fig)
    peak_month = monthly.idxmax()
    low_month = monthly.idxmin()
    findings.append(f"- **Seasonality:** revenue peaks in {peak_month:%B %Y} "
                    f"({fmt_try(monthly.max())}) and is lowest in {low_month:%B %Y} "
                    f"({fmt_try(monthly.min())}).")

    # Year-over-year, same-month comparison
    by_year = df.groupby("year")["revenue"].sum()
    if len(by_year) >= 2:
        y0, y1 = by_year.index[0], by_year.index[-1]
        growth = by_year[y1] / by_year[y0] - 1
        findings.append(f"- **Year over year:** {y1} revenue is {growth:+.1%} versus {y0}.")

    # ---- Revenue by category
    cat = df.groupby("category")["revenue"].sum().sort_values()
    fig, ax = plt.subplots()
    ax.barh(cat.index, cat.values / 1000, color=ACCENT)
    ax.set_title("Revenue by category")
    ax.set_xlabel("Revenue (thousand TRY)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "revenue_by_category.png", dpi=150)
    plt.close(fig)
    top_cat = cat.index[-1]
    findings.append(f"- **Top category:** {top_cat} brings in {cat.iloc[-1] / total_rev:.0%} of revenue.")

    # ---- Top products
    prod = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(8)[::-1]
    fig, ax = plt.subplots()
    ax.barh(prod.index, prod.values / 1000, color=ACCENT)
    ax.set_title("Top 8 products by revenue")
    ax.set_xlabel("Revenue (thousand TRY)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "top_products.png", dpi=150)
    plt.close(fig)
    findings.append(f"- **Best-selling product by revenue:** {prod.index[-1]} "
                    f"({fmt_try(prod.iloc[-1])}).")

    # ---- Revenue by city
    city = df[df["city"] != "Unknown"].groupby("city")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.bar(city.index, city.values / 1000, color=ACCENT)
    ax.set_title("Revenue by city")
    ax.set_ylabel("Revenue (thousand TRY)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "revenue_by_city.png", dpi=150)
    plt.close(fig)
    findings.append(f"- **Top city:** {city.index[0]} generates {city.iloc[0] / city.sum():.0%} "
                    f"of revenue where the city is known.")

    # ---- Discount impact: does discounting lead to bigger baskets?
    disc = df.groupby("discount").agg(orders=("order_id", "nunique"),
                                       avg_units=("quantity", "mean"),
                                       revenue=("revenue", "sum"))
    fig, ax = plt.subplots()
    ax.bar([f"{int(d * 100)}%" for d in disc.index], disc["avg_units"], color=ACCENT)
    ax.set_title("Average units per order line, by discount level")
    ax.set_ylabel("Average units")
    ax.set_xlabel("Discount")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "discount_vs_units.png", dpi=150)
    plt.close(fig)
    lift = disc["avg_units"].iloc[-1] / disc["avg_units"].iloc[0] - 1
    deepest = int(disc.index[-1] * 100)
    if abs(lift) < 0.05:
        verdict = ("Basket sizes barely change with the discount level, so deeper discounts give "
                   "away margin without a visible lift in volume. Worth testing before repeating them.")
    elif lift > 0:
        verdict = ("Deeper discounts come with larger baskets. Compare the extra volume against the "
                   "margin given up before concluding they pay off.")
    else:
        verdict = "Deeper discounts come with smaller baskets, the opposite of the usual assumption."
    findings.append(f"- **Discounts:** lines sold at {deepest}% off average "
                    f"{disc['avg_units'].iloc[-1]:.2f} units vs {disc['avg_units'].iloc[0]:.2f} at full "
                    f"price ({lift:+.0%}). {verdict}")

    # ---- Customer concentration
    cust_rev = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    top10pct = cust_rev.head(max(1, int(len(cust_rev) * 0.10))).sum() / total_rev
    findings.append(f"- **Customer concentration:** the top 10% of customers account for "
                    f"{top10pct:.0%} of revenue.")

    # ---- Write the report
    quality_lines = "\n".join(f"- {k.replace('_', ' ')}: {v:,}" for k, v in quality.items())
    report = (
        "# Findings\n\n"
        "_Generated by `analysis.py`. Numbers come from the synthetic dataset in `data/`._\n\n"
        "## Key results\n\n" + "\n".join(findings) + "\n\n"
        "## Data quality issues found and fixed\n\n" + quality_lines + "\n\n"
        "## Cleaning decisions\n\n"
        "- Exact duplicate rows were dropped.\n"
        "- Two date formats (ISO and day/month/year) were parsed separately to avoid mixing up "
        "day and month.\n"
        "- Category labels were trimmed and normalised to title case.\n"
        "- Missing cities were labelled `Unknown` and excluded from the city chart only.\n"
        "- Rows with non-positive quantities were excluded, since they look like returns or entry "
        "errors. In a real project, ask the data owner which it is.\n"
    )
    (OUT / "findings.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
