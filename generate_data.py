"""Generate a synthetic, deliberately messy retail orders dataset.

Why synthetic? It lets the project run anywhere with no downloads. To make the
project your own, swap `data/raw_orders.csv` for a real dataset (e.g. from
Kaggle) and adjust the column names in analysis.py.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ORDERS = 6000
N_CUSTOMERS = 450
START, END = date(2024, 1, 1), date(2025, 12, 31)

PRODUCTS = {
    "Electronics": [("Wireless Earbuds", 1200), ("Phone Case", 250), ("Power Bank", 650),
                    ("Smart Watch", 2800), ("USB-C Cable", 180), ("Bluetooth Speaker", 1500)],
    "Home": [("Desk Lamp", 450), ("Throw Blanket", 600), ("Coffee Maker", 1900),
             ("Storage Box", 220), ("Wall Clock", 380), ("Scented Candle", 160)],
    "Fashion": [("Cotton T-Shirt", 350), ("Denim Jacket", 1600), ("Sneakers", 2200),
                ("Leather Wallet", 700), ("Wool Scarf", 420), ("Sunglasses", 900)],
    "Sports": [("Yoga Mat", 550), ("Dumbbell Set", 1300), ("Water Bottle", 200),
               ("Running Belt", 280), ("Resistance Bands", 240), ("Jump Rope", 130)],
    "Books": [("Python Handbook", 480), ("Data Science Guide", 560), ("Novel Bestseller", 210),
              ("Design Basics", 390), ("Cookbook", 330), ("Language Workbook", 260)],
}
CITIES = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"]
CITY_WEIGHTS = [0.40, 0.20, 0.17, 0.12, 0.11]
CATEGORY_WEIGHTS = {"Electronics": 0.28, "Home": 0.22, "Fashion": 0.26, "Sports": 0.13, "Books": 0.11}
MONTH_WEIGHT = {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.05,
                7: 1.1, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.6, 12: 1.5}


def random_date():
    days = (END - START).days
    while True:
        d = START + timedelta(days=random.randint(0, days))
        # Accept with probability proportional to the month's seasonality weight
        if random.random() < MONTH_WEIGHT[d.month] / 1.6:
            return d


def main():
    rows = []
    customer_city = {f"C{str(i).zfill(4)}": np.random.choice(CITIES, p=CITY_WEIGHTS)
                     for i in range(1, N_CUSTOMERS + 1)}
    customer_ids = list(customer_city)
    # A minority of customers order much more often (repeat buyers)
    cust_weights = np.random.lognormal(0, 0.7, N_CUSTOMERS)
    cust_weights /= cust_weights.sum()

    cats = list(CATEGORY_WEIGHTS)
    cat_p = list(CATEGORY_WEIGHTS.values())

    for i in range(1, N_ORDERS + 1):
        cust = np.random.choice(customer_ids, p=cust_weights)
        cat = np.random.choice(cats, p=cat_p)
        product, price = random.choice(PRODUCTS[cat])
        qty = int(np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.09, 0.04]))
        discount = float(np.random.choice([0, 0.05, 0.10, 0.20], p=[0.6, 0.15, 0.15, 0.10]))
        rows.append({
            "order_id": f"ORD{str(i).zfill(6)}",
            "order_date": random_date(),
            "customer_id": cust,
            "city": customer_city[cust],
            "category": cat,
            "product": product,
            "unit_price": price,
            "quantity": qty,
            "discount": discount,
        })

    df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)

    # ---- Inject realistic mess so the cleaning step has real work to do ----
    # 1. Mixed date formats
    df["order_date"] = df["order_date"].astype(str)
    idx = df.sample(frac=0.08, random_state=1).index
    df.loc[idx, "order_date"] = pd.to_datetime(df.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
    # 2. Inconsistent category casing / whitespace
    idx = df.sample(frac=0.06, random_state=2).index
    df.loc[idx, "category"] = df.loc[idx, "category"].str.upper()
    idx = df.sample(frac=0.04, random_state=3).index
    df.loc[idx, "category"] = " " + df.loc[idx, "category"].str.lower() + " "
    # 3. Missing cities
    idx = df.sample(frac=0.03, random_state=4).index
    df.loc[idx, "city"] = np.nan
    # 4. Invalid quantities (returns / data-entry errors)
    idx = df.sample(frac=0.01, random_state=5).index
    df.loc[idx, "quantity"] = -df.loc[idx, "quantity"]
    # 5. Exact duplicate rows
    dupes = df.sample(frac=0.02, random_state=6)
    df = pd.concat([df, dupes]).sample(frac=1, random_state=7).reset_index(drop=True)

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "raw_orders.csv", index=False)
    print(f"Wrote {len(df):,} rows to {out / 'raw_orders.csv'}")


if __name__ == "__main__":
    main()
