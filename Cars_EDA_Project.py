"""
Cars Dataset - Exploratory Data Analysis (EDA)
A single-file AIML mini project.

How to run:
1. Install dependencies:
   python -m pip install pandas numpy matplotlib seaborn
2. Place your cars CSV file in the same folder as this script.
3. Change DATASET_PATH below to your CSV filename.
4. Run:
   python Cars_EDA_Project.py

The script saves cleaned data, charts, and text summaries in a folder
named cars_eda_outputs.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# -------------------- 1. CONFIGURATION --------------------

DATASET_PATH = "cars.csv"  # Change this to your dataset filename
OUTPUT_DIR = Path("cars_eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Common alternative names used by different cars datasets.
COLUMN_ALIASES = {
    "brand": ["brand", "make", "manufacturer", "car_brand", "company"],
    "model": ["model", "car_model", "name", "vehicle"],
    "year": ["year", "model_year", "manufacture_year", "registration_year"],
    "price": ["price", "selling_price", "sellingprice", "msrp", "cost"],
    "fuel": ["fuel", "fuel_type", "fueltype", "fuel_type_name"],
    "mileage": ["mileage", "km_driven", "kms_driven", "kilometers_driven",
                "odometer", "distance_driven"],
    "transmission": ["transmission", "gearbox", "transmission_type"],
    "engine": ["engine", "engine_cc", "engine_size", "displacement"],
    "horsepower": ["horsepower", "hp", "power", "max_power"],
    "owner": ["owner", "owner_type", "number_of_owners"],
    "body_type": ["body", "body_type", "vehicle_type", "car_type"],
}

def normalise_name(name):
    """Convert a column name to a simple lowercase form."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def find_column(columns, logical_name):
    """Find a dataset column matching a known logical feature."""
    normalised = {normalise_name(c): c for c in columns}
    for alias in COLUMN_ALIASES.get(logical_name, []):
        key = normalise_name(alias)
        if key in normalised:
            return normalised[key]
    return None

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

def write_report(lines, filename="eda_summary.txt"):
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as file:
        file.write("\n".join(str(line) for line in lines))

# -------------------- 2. LOAD DATASET --------------------

dataset_file = Path(DATASET_PATH)
if not dataset_file.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Put your cars CSV file in this folder or update DATASET_PATH."
    )

df = pd.read_csv(dataset_file)
if df.empty:
    raise ValueError("The CSV file is empty. Please use a valid cars dataset.")

print("\n========== 1. DATASET OVERVIEW ==========")
print("Dataset shape (rows, columns):", df.shape)
print("\nColumn names:\n", list(df.columns))
print("\nFirst five records:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nNumerical columns:", list(df.select_dtypes(include=np.number).columns))
print("\nCategorical columns:",
      list(df.select_dtypes(include=["object", "category", "bool"]).columns))

original_rows, original_cols = df.shape
overview_lines = [
    "CARS DATASET - EDA SUMMARY",
    f"Dataset source/file: {DATASET_PATH}",
    f"Original rows: {original_rows}",
    f"Original columns: {original_cols}",
    f"Columns: {list(df.columns)}",
    "",
    "Data types:",
    df.dtypes.to_string(),
]
write_report(overview_lines)

# -------------------- 3. DATA QUALITY CHECKS --------------------

print("\n========== 2. DATA QUALITY CHECKS ==========")
missing_before = df.isnull().sum()
duplicate_count = int(df.duplicated().sum())

print("\nMissing values by column:\n", missing_before)
print("\nDuplicate rows:", duplicate_count)

quality_lines = [
    "\nDATA QUALITY BEFORE CLEANING",
    "Missing values by column:",
    missing_before.to_string(),
    f"Duplicate rows: {duplicate_count}",
]

# Standardise text values and trim extra spaces.
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace({
        "": pd.NA, "nan": pd.NA, "None": pd.NA,
        "null": pd.NA, "NULL": pd.NA
    })

# Remove exact duplicate rows.
df = df.drop_duplicates().copy()

# Detect useful columns.
columns = {name: find_column(df.columns, name) for name in COLUMN_ALIASES}
print("\nRecognised columns:", columns)

# Convert common numerical features safely.
numeric_features = ["year", "price", "mileage", "engine", "horsepower"]
for feature in numeric_features:
    col = columns.get(feature)
    if col is not None:
        # Remove common units and commas before conversion.
        cleaned = (df[col].astype("string")
                   .str.replace(",", "", regex=False)
                   .str.replace(r"[^0-9.\-]", "", regex=True))
        df[col] = pd.to_numeric(cleaned, errors="coerce")

# Convert year only if it looks like a year column.
year_col = columns.get("year")
if year_col is not None:
    current_year = pd.Timestamp.today().year
    invalid_year = df[year_col].notna() & (
        (df[year_col] < 1886) | (df[year_col] > current_year + 1)
    )
    df.loc[invalid_year, year_col] = np.nan

# Fill missing numerical values with median and categorical with mode.
# This keeps rows while using robust, simple imputation.
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        median_value = df[col].median()
        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    if df[col].isna().any():
        mode_values = df[col].mode(dropna=True)
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

# Remove rows that remain completely empty.
df = df.dropna(how="all").copy()

clean_rows, clean_cols = df.shape
missing_after = df.isnull().sum()

print("\n========== CLEANING RESULTS ==========")
print("Rows before cleaning:", original_rows)
print("Rows after cleaning:", clean_rows)
print("Columns after cleaning:", clean_cols)
print("Missing values after cleaning:\n", missing_after)
print("Duplicate rows after cleaning:", int(df.duplicated().sum()))

cleaning_lines = [
    "\nCLEANING DECISIONS",
    "- Trimmed leading/trailing spaces in text columns.",
    "- Converted recognised year, price, mileage, engine and horsepower fields to numeric where available.",
    "- Replaced invalid year values with missing values where a year column was recognised.",
    "- Removed exact duplicate rows.",
    "- Filled remaining numeric missing values with column medians.",
    "- Filled remaining categorical missing values with column modes.",
    f"Rows before cleaning: {original_rows}",
    f"Rows after cleaning: {clean_rows}",
    f"Missing values after cleaning:\n{missing_after.to_string()}",
]
write_report(overview_lines + quality_lines + cleaning_lines)

# Save cleaned dataset for checking and reuse.
df.to_csv(OUTPUT_DIR / "cleaned_cars_data.csv", index=False)

# -------------------- 4. DESCRIPTIVE STATISTICS --------------------

print("\n========== 3. DESCRIPTIVE STATISTICS ==========")
numeric_summary = df.describe(include=[np.number]).T
categorical_summary = df.describe(include=["object", "category", "string"]).T

print("\nNumerical summary:\n", numeric_summary)
print("\nCategorical summary:\n", categorical_summary)

with open(OUTPUT_DIR / "descriptive_statistics.txt", "w", encoding="utf-8") as f:
    f.write("NUMERICAL DESCRIPTIVE STATISTICS\n")
    f.write(numeric_summary.to_string())
    f.write("\n\nCATEGORICAL DESCRIPTIVE STATISTICS\n")
    f.write(categorical_summary.to_string())

# -------------------- 5. PREPARE INSIGHTS --------------------

insights = []
brand_col = columns.get("brand")
price_col = columns.get("price")
fuel_col = columns.get("fuel")
mileage_col = columns.get("mileage")
trans_col = columns.get("transmission")

if brand_col:
    counts = df[brand_col].value_counts()
    if not counts.empty:
        insights.append(
            f"Most represented brand/category: {counts.index[0]} "
            f"({counts.iloc[0]} records)."
        )

if fuel_col:
    counts = df[fuel_col].value_counts()
    if not counts.empty:
        insights.append(
            f"Most common fuel type: {counts.index[0]} "
            f"({counts.iloc[0]} records)."
        )

if price_col and pd.api.types.is_numeric_dtype(df[price_col]):
    insights.append(
        f"Price: median = {df[price_col].median():,.2f}; "
        f"mean = {df[price_col].mean():,.2f}."
    )
    if brand_col:
        avg_price = df.groupby(brand_col)[price_col].mean().sort_values(ascending=False)
        if len(avg_price):
            insights.append(
                f"Highest average price among groups: {avg_price.index[0]} "
                f"({avg_price.iloc[0]:,.2f}); interpret cautiously if the group is small."
            )

if year_col and price_col:
    valid = df[[year_col, price_col]].dropna()
    if len(valid) >= 3:
        corr = valid[year_col].corr(valid[price_col])
        insights.append(
            f"Correlation between year and price: {corr:.2f}. "
            "A positive value means newer model years tend to have higher prices in this dataset."
        )

if mileage_col and price_col:
    valid = df[[mileage_col, price_col]].dropna()
    if len(valid) >= 3:
        corr = valid[mileage_col].corr(valid[price_col])
        insights.append(
            f"Correlation between mileage and price: {corr:.2f}. "
            "A negative value means higher mileage tends to be associated with lower prices."
        )

if trans_col and price_col:
    grouped = df.groupby(trans_col)[price_col].mean().sort_values(ascending=False)
    if len(grouped) >= 2:
        insights.append(
            f"Average price by transmission: {grouped.to_dict()}. "
            "This is an association, not proof that transmission causes price differences."
        )

# -------------------- 6. VISUAL EDA: SIX OR MORE CHARTS --------------------

chart_observations = []

# Chart 1: Price distribution (histogram).
if price_col and df[price_col].notna().any():
    plt.figure(figsize=(9, 5))
    sns.histplot(df[price_col], bins=30, kde=True)
    plt.title("Distribution of Car Prices")
    plt.xlabel(price_col)
    plt.ylabel("Number of cars")
    save_plot("01_price_distribution.png")
    chart_observations.append(
        "Chart 1 - Price distribution: inspect the shape, spread and skewness of car prices."
    )

# Chart 2: Most common brands (bar chart).
if brand_col:
    top_brands = df[brand_col].value_counts().head(10)
    if not top_brands.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_brands.values, y=top_brands.index)
        plt.title("Top 10 Most Common Car Brands")
        plt.xlabel("Number of cars")
        plt.ylabel("Brand")
        save_plot("02_top_brands.png")
        chart_observations.append(
            f"Chart 2 - Brands: {top_brands.index[0]} is the most frequent brand/category in the records."
        )

# Chart 3: Fuel type count plot.
if fuel_col:
    fuel_counts = df[fuel_col].value_counts().head(12)
    if not fuel_counts.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(x=fuel_counts.index.astype(str), y=fuel_counts.values)
        plt.title("Cars by Fuel Type")
        plt.xlabel("Fuel type")
        plt.ylabel("Number of cars")
        plt.xticks(rotation=35, ha="right")
        save_plot("03_fuel_type_count.png")
        chart_observations.append(
            f"Chart 3 - Fuel type: {fuel_counts.index[0]} is the most common fuel category."
        )

# Chart 4: Price outliers by fuel type (box plot), or by transmission.
box_category = fuel_col or trans_col or brand_col
if price_col and box_category:
    plot_data = df[[box_category, price_col]].dropna().copy()
    top_categories = plot_data[box_category].value_counts().head(8).index
    plot_data = plot_data[plot_data[box_category].isin(top_categories)]
    if not plot_data.empty:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=plot_data, x=box_category, y=price_col)
        plt.title(f"Car Price Distribution by {box_category}")
        plt.xlabel(box_category)
        plt.ylabel(price_col)
        plt.xticks(rotation=30, ha="right")
        save_plot("04_price_boxplot.png")
        chart_observations.append(
            "Chart 4 - Box plot: compare medians, spread and possible price outliers across categories."
        )

# Chart 5: Year vs price scatter plot.
if year_col and price_col:
    plot_data = df[[year_col, price_col]].dropna()
    if len(plot_data):
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=plot_data, x=year_col, y=price_col, alpha=0.6)
        plt.title("Relationship Between Model Year and Car Price")
        plt.xlabel("Model year")
        plt.ylabel("Car price")
        save_plot("05_year_vs_price.png")
        chart_observations.append(
            "Chart 5 - Year vs price: observe whether newer model years tend to have higher prices."
        )

# Chart 6: Mileage vs price scatter plot.
if mileage_col and price_col:
    plot_data = df[[mileage_col, price_col]].dropna()
    if len(plot_data):
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=plot_data, x=mileage_col, y=price_col, alpha=0.6)
        plt.title("Relationship Between Mileage and Car Price")
        plt.xlabel("Mileage / distance driven")
        plt.ylabel("Car price")
        save_plot("06_mileage_vs_price.png")
        chart_observations.append(
            "Chart 6 - Mileage vs price: look for a downward or upward pattern and unusual points."
        )

# Chart 7: Correlation heatmap for numeric features.
numeric_df = df.select_dtypes(include=np.number)
if numeric_df.shape[1] >= 2:
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Features")
    save_plot("07_correlation_heatmap.png")
    chart_observations.append(
        "Chart 7 - Correlation heatmap: values near +1 or -1 indicate stronger linear relationships."
    )

# Chart 8: Average price by transmission.
if trans_col and price_col:
    average_by_trans = df.groupby(trans_col)[price_col].mean().sort_values(ascending=False)
    if not average_by_trans.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=average_by_trans.index.astype(str), y=average_by_trans.values)
        plt.title("Average Car Price by Transmission")
        plt.xlabel("Transmission")
        plt.ylabel("Average car price")
        plt.xticks(rotation=25, ha="right")
        save_plot("08_average_price_by_transmission.png")
        chart_observations.append(
            "Chart 8 - Transmission: compare average prices across the available transmission groups."
        )

# -------------------- 7. OUTLIER CHECK --------------------

outlier_lines = ["\nOUTLIER CHECK (IQR METHOD)"]
for feature in ["price", "mileage", "engine", "horsepower"]:
    col = columns.get(feature)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        values = df[col].dropna()
        if len(values) >= 4:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((values < lower) | (values > upper)).sum())
            outlier_lines.append(
                f"{col}: lower bound={lower:.2f}, upper bound={upper:.2f}, "
                f"possible outliers={count}"
            )

# -------------------- 8. FINAL FINDINGS AND LIMITATIONS --------------------

if not insights:
    insights.append(
        "The dataset did not contain enough recognised car attributes to generate automatic findings. "
        "Review the column names and update COLUMN_ALIASES if needed."
    )

limitations = [
    "The findings describe patterns in this dataset and may not represent the entire car market.",
    "Missing values were filled using median or mode; this can affect distributions and relationships.",
    "Outliers are flagged using the IQR method and are not automatically removed because they may be valid cars.",
    "Correlations show association, not causation.",
    "Results depend on the dataset source, collection period, location, and the meaning/units of its columns.",
]

final_lines = [
    "\nCHART OBSERVATIONS",
    *chart_observations,
    "\nAT LEAST FIVE DATA-DRIVEN FINDINGS (where columns permit)",
    *[f"{i}. {item}" for i, item in enumerate(insights, start=1)],
    *outlier_lines,
    "\nLIMITATIONS",
    *[f"- {item}" for item in limitations],
    "\nCONCLUSION",
    "This EDA inspected the cars dataset, checked data quality, cleaned selected issues, "
    "summarised numerical and categorical features, visualised distributions and relationships, "
    "and identified possible outliers and patterns. Interpret every finding in the context "
    "of the dataset and its limitations.",
]
write_report(overview_lines + quality_lines + cleaning_lines + final_lines)

print("\n========== 4. CHART OBSERVATIONS ==========")
for item in chart_observations:
    print("-", item)

print("\n========== 5. KEY FINDINGS ==========")
for number, item in enumerate(insights, start=1):
    print(f"{number}. {item}")

print("\n========== 6. OUTLIER CHECK ==========")
for item in outlier_lines[1:]:
    print("-", item)

print("\n========== 7. LIMITATIONS ==========")
for item in limitations:
    print("-", item)

print("\n========== PROJECT COMPLETED ==========")
print(f"Original dataset size: {original_rows} rows x {original_cols} columns")
print(f"Cleaned dataset size: {clean_rows} rows x {clean_cols} columns")
print(f"Charts and summaries saved in: {OUTPUT_DIR.resolve()}")
print("Review the generated observations and ensure they match the charts before submitting.")
