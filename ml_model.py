import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def _extract_year_columns(df):
    # Find columns that end with 'Population' and extract a 4-digit year from the name
    cols = [c for c in df.columns if c.strip().endswith('Population')]
    years = []
    for c in cols:
        parts = c.split()
        year_found = None
        for p in parts:
            if p.isdigit() and len(p) == 4:
                year_found = int(p)
                break
        if year_found is not None:
            years.append((year_found, c))
    return sorted(years, key=lambda x: x[0])


def prepare_data(df: pd.DataFrame, country: str = 'India') -> pd.DataFrame:
    # Attempt common country column names
    country_cols = ['Country/Territory', 'Country', 'country']
    country_col = None
    for c in country_cols:
        if c in df.columns:
            country_col = c
            break
    if country_col is None:
        # fallback to first column
        country_col = df.columns[0]

    row = df[df[country_col] == country]
    if row.empty:
        raise ValueError(f"Country '{country}' not found in column '{country_col}'")

    year_cols = _extract_year_columns(df)
    if not year_cols:
        raise ValueError("No population year columns found (expected columns like '1970 Population')")

    data = {'Year': [], 'Population': []}
    for year, col in year_cols:
        val = row[col].values[0]
        try:
            pop = float(val)
        except Exception:
            pop = np.nan
        if not np.isnan(pop):
            data['Year'].append(year)
            data['Population'].append(pop)

    new_df = pd.DataFrame(data)
    new_df = new_df.sort_values('Year').reset_index(drop=True)
    return new_df


def train_polynomial_model(df: pd.DataFrame, degree: int = 3):
    X = df[['Year']].values
    y = df['Population'].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    return poly, model


def predict_years(poly: PolynomialFeatures, model: LinearRegression, years):
    yrs = np.array(years).reshape(-1, 1)
    yrs_poly = poly.transform(yrs)
    preds = model.predict(yrs_poly)
    return preds


def plot_fit(df: pd.DataFrame, poly: PolynomialFeatures, model: LinearRegression):
    X = df[['Year']].values
    y = df['Population'].values
    x_min, x_max = X.min(), X.max()
    X_seq = np.linspace(x_min, max(x_max + 1, x_max + 30), 300).reshape(-1, 1)
    y_seq = model.predict(poly.transform(X_seq))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(X, y, label='data', color='tab:blue')
    ax.plot(X_seq, y_seq, color='tab:red', label='polynomial fit')
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')
    ax.legend()
    ax.grid(alpha=0.2)
    return fig
