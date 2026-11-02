# Retail-Sales-Analysis-python-pandas


An end-to-end exploratory analysis of a retail orders dataset: audit the raw data, clean it, answer business questions, and visualise the results.

## Business questions answered

1. How much revenue do we make, and what is the average order value?
2. When does revenue peak, and is the business growing year over year?
3. Which categories, products and cities drive revenue?
4. Do discounts actually lead to bigger baskets?
5. How concentrated is revenue among the top customers?

## Data 

`generate_data.py` creates a synthetic dataset of ~6,000 orders (2024-2025) so the project runs without any downloads. It is **deliberately messy**: duplicate rows, mixed date formats (`2025-04-25` and `25/04/2025`), inconsistent category labels, missing cities and negative quantities.

> The data is synthetic, so the numbers in `outputs/findings.md` illustrate the method, not real-world facts. To get genuine findings, replace `data/raw_orders.csv` with a real dataset (Kaggle has many retail ones) and adjust the column names in `analysis.py`.

## How to run

```bash
pip install -r requirements.txt
python generate_data.py   # creates data/raw_orders.csv
python analysis.py        # writes charts and findings to outputs/
```

## Output

- `outputs/findings.md`: key results, data-quality issues found, and every cleaning decision
- `outputs/*.png`: monthly revenue, revenue by category / city, top products, discount vs basket size

![Monthly revenue](outputs/monthly_revenue.png)

## Approach

1. **Audit** the raw file and count each data-quality problem before changing anything.
2. **Clean**: drop duplicates, parse both date formats separately (so day and month are never swapped), normalise category labels, label missing cities `Unknown`, exclude non-positive quantities.
3. **Engineer** `revenue = unit_price * quantity * (1 - discount)` plus month and year fields.
4. **Analyse and visualise** with pandas and matplotlib.

## Skills demonstrated

Data cleaning and validation, pandas aggregation, time-series trends, KPI design, data visualisation, and writing up findings and assumptions.

## Ideas to extend it

- Add a customer segmentation (RFM or k-means)
- Build an interactive dashboard with Streamlit
- Run a proper significance test on the discount question
