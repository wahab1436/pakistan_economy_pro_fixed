# pakistan_economy_pro_fixed.py
"""
Pakistan Economic Insights — Professional (Fixed)
This file is the error-free variant where all OLS trendlines are computed via scikit-learn
to avoid statsmodels/scipy compatibility issues that break Plotly's trendline feature.

Save and run:
streamlit run "H:/PYTHON/New folder/pakistan_economy_pro_fixed.py"
"""

import math
import time
import textwrap
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# optional libs
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# color palette
PALETTE = {
    "bg": "#0f1724",
    "card": "#ffffff",
    "text": "#0b1220",
    "gdp": "#1f77b4",
    "inflation": "#ff7f0e",
    "trade_pos": "#2ca02c",
    "trade_neg": "#d62728",
    "employment": "#9467bd",
    "remit": "#17becf",
    "gdppercap": "#bcbd22"
}

def main():
    st.set_page_config(page_title="🇵🇰 Pakistan Economic Insights", layout="wide")

    # custom css for stylish titles + cards
    st.markdown(
        """
        <style>
        .big-title {
            font-size:38px;
            font-weight:800;
            text-align:center;
            color: #ffffff;
            padding: 16px;
            border-radius: 12px;
            background: linear-gradient(90deg, #1E3C72, #2A5298);
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        }
        .sub-title {
            font-size:16px;
            text-align:center;
            color: #f1f5f9;
            margin-bottom: 20px;
        }
        .kpi {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 12px;
            padding: 16px;
            text-align:center;
            font-weight:600;
            box-shadow: 0 3px 8px rgba(0,0,0,0.15);
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # dashboard title
    st.markdown("<div class='big-title'>🇵🇰 Pakistan Economy Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Data-driven Insights with Machine Learning & AI</div>", unsafe_allow_html=True)

def generate_dummy_pakistan_data(start_year: int = 2000, end_year: int = 2023, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    years = np.arange(start_year, end_year + 1)
    n = len(years)

    base_gdp = 2.8 + 0.06 * (years - start_year)
    noise_gdp = np.random.normal(0, 1.2, n)
    shock = np.zeros(n)
    shock[years == 2008] = -2.5
    shock[years == 2010] = -1.2
    shock[years == 2011] = -0.8
    shock[years == 2020] = -3.2
    shock[years == 2022] = -1.6
    gdp_growth = np.round(base_gdp + noise_gdp + shock, 2)

    base_infl = 5.8 + 0.15 * (years - start_year)
    infl_noise = np.random.normal(0, 1.8, n)
    inflation = np.clip(np.round(base_infl + infl_noise + -0.8 * shock, 2), 0.5, 40)

    exports = np.round(12 + (years - start_year) * 1.05 + np.random.normal(0, 2.8, n), 2)
    imports = np.round(exports * (1.05 + 0.05 * np.random.randn(n)) + np.linspace(4, 35, n) * 0.5, 2)

    remittances = np.round(3 + (years - start_year) * 0.9 + np.random.normal(0, 1.6, n) + (years == 2020) * 4.8, 2)

    unemployment = np.round(4 + 0.04 * (years - start_year) + np.random.normal(0, 0.7, n) + 0.4 * (-shock), 2)
    unemployment = np.clip(unemployment, 2.0, 25.0)

    base_gdp_pc = 800 + (years - start_year) * 45 + np.random.normal(0, 80, n)
    base_gdp_pc = np.round(base_gdp_pc + -50 * shock, 2)

    df = pd.DataFrame({
        "Year": years,
        "GDP_Growth": gdp_growth,
        "Inflation": inflation,
        "Exports_BnUSD": exports,
        "Imports_BnUSD": imports,
        "Remittances_BnUSD": remittances,
        "Unemployment_pct": unemployment,
        "GDP_per_capita_USD": base_gdp_pc
    })
    df["Trade_Balance_BnUSD"] = (df["Exports_BnUSD"] - df["Imports_BnUSD"]).round(2)
    df["Current_Account_Adjusted_BnUSD"] = (df["Trade_Balance_BnUSD"] + df["Remittances_BnUSD"]).round(2)
    return df


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    d = df.copy().set_index("Year")
    d["GDP_Growth_Lag1"] = d["GDP_Growth"].shift(1)
    d["GDP_Growth_Lag2"] = d["GDP_Growth"].shift(2)
    d["Remit_to_Imports"] = d["Remittances_BnUSD"] / (d["Imports_BnUSD"].replace(0, np.nan))
    d["Export_Import_Ratio"] = d["Exports_BnUSD"] / (d["Imports_BnUSD"].replace(0, np.nan))
    d["Trade_to_GDPpc"] = d["Trade_Balance_BnUSD"] / (d["GDP_per_capita_USD"].replace(0, np.nan))
    d = d.fillna(method="bfill").fillna(method="ffill")
    features = ["Inflation", "Exports_BnUSD", "Imports_BnUSD", "Remittances_BnUSD",
                "Unemployment_pct", "Remit_to_Imports", "Export_Import_Ratio",
                "GDP_Growth_Lag1", "GDP_Growth_Lag2", "Trade_to_GDPpc"]
    X = d[features].copy()
    y_gdp = d["GDP_Growth"].copy()
    y_gdppc = d["GDP_per_capita_USD"].copy()
    return X, y_gdp, y_gdppc


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def regression_pipeline_with_sklearn(X: pd.DataFrame, y: pd.Series, use_xgb: bool = False) -> Dict[str, Any]:
    results = {}
    X_np = X.values
    y_np = y.values

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_np, y_np)
    pred_lr = lr.predict(X_np)
    results["LinearRegression"] = {"model": lr, "pred": pred_lr, "metrics": compute_regression_metrics(y_np, pred_lr)}

    # RandomForest
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42))])
    rf_pipe.fit(X_np, y_np)
    pred_rf = rf_pipe.predict(X_np)
    results["RandomForest"] = {"model": rf_pipe, "pred": pred_rf, "metrics": compute_regression_metrics(y_np, pred_rf)}

    # XGBoost optional
    if use_xgb and XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline([("scaler", StandardScaler()), ("xgb", xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, tree_method="hist"))])
        xgb_pipe.fit(X_np, y_np)
        pred_xgb = xgb_pipe.predict(X_np)
        results["XGBoost"] = {"model": xgb_pipe, "pred": pred_xgb, "metrics": compute_regression_metrics(y_np, pred_xgb)}

    return results


def train_neural_network(X: pd.DataFrame, y: pd.Series, epochs: int = 300, verbose: int = 0) -> Dict[str, Any]:
    if not TF_AVAILABLE:
        return {"error": "TensorFlow not installed"}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    ys = y.values
    model = keras.Sequential([
        keras.layers.Input(shape=(Xs.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=40, restore_best_weights=True, verbose=0)
    history = model.fit(Xs, ys, epochs=epochs, batch_size=8, verbose=verbose, callbacks=[early_stop])
    pred = model.predict(Xs).flatten()
    metrics = compute_regression_metrics(ys, pred)
    return {"model": model, "scaler": scaler, "history": history.history, "pred": pred, "metrics": metrics}


def human_friendly_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    recent_growth = df["GDP_Growth"].iloc[-3:].mean()
    insights.append(f"Recent average GDP growth (last 3 years) is about {recent_growth:.2f}% — this indicates short-term momentum.")
    peak_year = int(df.loc[df["GDP_Growth"].idxmax(), "Year"])
    trough_year = int(df.loc[df["GDP_Growth"].idxmin(), "Year"])
    insights.append(f"GDP growth peaked in {peak_year} and the trough was in {trough_year}. Investigate those years for shock/context.")
    high_infl_year = int(df.loc[df["Inflation"].idxmax(), "Year"])
    insights.append(f"Inflation spiked most in {high_infl_year}; link spikes to policy or external shocks in narrative.")
    worst_deficit = df["Trade_Balance_BnUSD"].min()
    worst_year = int(df.loc[df["Trade_Balance_BnUSD"].idxmin(), "Year"])
    insights.append(f"Trade deficit worst in {worst_year} at {worst_deficit:.2f} Bn USD — trade policy matters.")
    remit_growth_pct = (df["Remittances_BnUSD"].iloc[-1] / df["Remittances_BnUSD"].iloc[0] - 1) * 100
    insights.append(f"Remittances grew ~{remit_growth_pct:.1f}% across the period, acting as a current account buffer.")
    corr_unemp = df[["Unemployment_pct", "GDP_Growth"]].corr().iloc[0, 1]
    insights.append(f"Unemployment & GDP growth correlation: r={corr_unemp:.2f}; rising unemployment often accompanies weaker growth.")
    return insights


def format_kpis(df: pd.DataFrame) -> Dict[str, str]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    kpis = {
        "GDP_Growth": f"{latest['GDP_Growth']:.2f}%",
        "Inflation": f"{latest['Inflation']:.2f}%",
        "Trade_Balance": f"{latest['Trade_Balance_BnUSD']:.2f} B USD",
        "GDP_per_capita": f"{latest['GDP_per_capita_USD']:.0f} USD",
        "Unemployment": f"{latest['Unemployment_pct']:.2f}%"
    }

    def delta(curr, prev):
        d = curr - prev
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
        return f"{arrow} {abs(d):.2f}"

    kpis["GDP_Growth_delta"] = delta(latest['GDP_Growth'], prev['GDP_Growth'])
    kpis["Inflation_delta"] = delta(latest['Inflation'], prev['Inflation'])
    return kpis


def plot_actual_vs_predicted_years(years, actual, predicted, title):
    plot_df = pd.DataFrame({"Year": years, "Actual": actual, "Predicted": predicted})
    fig = px.line(plot_df, x="Year", y=["Actual", "Predicted"], title=title)
    return fig


def plot_regression_scatter_with_sklearn(x, y, x_label: str, y_label: str, title: str, color: str = "#000000"):
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    lr = LinearRegression()
    lr.fit(X, Y)
    y_pred = lr.predict(X)
    # For smoother regression line, build sorted x
    sort_idx = np.argsort(X.flatten())
    x_sorted = X.flatten()[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(color=color)))
    fig.add_trace(go.Scatter(x=x_sorted, y=y_pred_sorted, mode="lines", name="OLS (sklearn)", line=dict(color="red", width=2)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
    return fig, lr


# --------------
# Page display functions
# --------------
def show_overview(df: pd.DataFrame):
    st.header("Overview — Key Time Series")
    st.markdown("Overview of GDP, Inflation, Unemployment and Remittances.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["GDP_Growth"], mode="lines+markers", name="GDP Growth", line=dict(color=PALETTE["gdp"])))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Inflation"], mode="lines+markers", name="Inflation", line=dict(color=PALETTE["inflation"])))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Unemployment_pct"], mode="lines+markers", name="Unemployment", line=dict(color=PALETTE["employment"])))
    fig.add_trace(go.Bar(x=df["Year"], y=df["Remittances_BnUSD"], name="Remittances (Bn USD)", marker_color=PALETTE["remit"], opacity=0.6, yaxis="y2"))
    fig.update_layout(title="Macro overview", xaxis_title="Year", yaxis_title="Percent / Values", yaxis2=dict(title="Remittances (Bn USD)", overlaying="y", side="right"), template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 3-year moving averages")
    df["GDP_MA3"] = df["GDP_Growth"].rolling(window=3, min_periods=1).mean()
    df["Inflation_MA3"] = df["Inflation"].rolling(window=3, min_periods=1).mean()
    fig2 = px.line(df, x="Year", y=["GDP_MA3", "Inflation_MA3"], labels={"value": "3-year moving avg"}, title="3-year Moving Averages")
    st.plotly_chart(fig2, use_container_width=True)


def show_gdp_page(df: pd.DataFrame):
    st.header("GDP — Trend & Per Capita")
    st.markdown("Trend of GDP Growth and GDP per capita, plus regression relationship with Inflation.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["GDP_Growth"], mode="lines+markers", name="GDP Growth", line=dict(color=PALETTE["gdp"])))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["GDP_per_capita_USD"], mode="lines", name="GDP per capita", line=dict(color=PALETTE["gdppercap"]), yaxis="y2"))
    fig.update_layout(title="GDP Growth & GDP per Capita", xaxis_title="Year", yaxis_title="GDP Growth (%)", yaxis2=dict(title="GDP per capita (USD)", overlaying="y", side="right"), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### GDP vs Inflation — regression (sklearn)")
    scatter_fig, lr_model = plot_regression_scatter_with_sklearn(df["Inflation"], df["GDP_Growth"], "Inflation (%)", "GDP Growth (%)", "GDP Growth vs Inflation (OLS)", color=PALETTE["inflation"])
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Show regression coefficients and intercept
    coef = lr_model.coef_[0]
    intercept = lr_model.intercept_
    st.markdown(f"**OLS result (sklearn):** GDP_Growth = {intercept:.3f} + {coef:.3f} * Inflation")


def show_inflation_page(df: pd.DataFrame):
    st.header("Inflation — Volatility & Peaks")
    st.markdown("Inflation levels and rolling volatility.")
    df["Inflation_ROLL_STD"] = df["Inflation"].rolling(window=3, min_periods=1).std()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Year"], y=df["Inflation"], name="Inflation", marker_color=PALETTE["inflation"], opacity=0.7))
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Inflation_ROLL_STD"], mode="lines+markers", name="Rolling STD (3y)", line=dict(color="#999999")))
    fig.update_layout(title="Inflation & Volatility", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    top = df.nlargest(5, "Inflation")[["Year", "Inflation"]]
    st.markdown("Top 5 inflation years")
    st.table(top.reset_index(drop=True))


def show_trade_page(df: pd.DataFrame):
    st.header("Trade — Exports vs Imports")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Year"], y=df["Exports_BnUSD"], name="Exports", marker_color=PALETTE["trade_pos"]))
    fig.add_trace(go.Bar(x=df["Year"], y=df["Imports_BnUSD"], name="Imports", marker_color=PALETTE["trade_neg"]))
    fig.update_layout(barmode="group", title="Exports vs Imports (Bn USD)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Trade Balance")
    fig2 = px.bar(df, x="Year", y="Trade_Balance_BnUSD", color="Trade_Balance_BnUSD", color_continuous_scale=px.colors.diverging.RdYlGn, title="Trade Balance (Bn USD)")
    st.plotly_chart(fig2, use_container_width=True)
    worst = df.loc[df["Trade_Balance_BnUSD"].idxmin()]
    st.markdown(f"**Worst trade deficit:** {worst['Trade_Balance_BnUSD']:.2f} Bn USD in {int(worst['Year'])}")


def show_employment_page(df: pd.DataFrame):
    st.header("Employment — Labour Market")
    st.markdown("Unemployment trend and relationship with GDP growth.")
    fig = px.line(df, x="Year", y="Unemployment_pct", title="Unemployment (%)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Unemployment vs GDP Growth — regression (sklearn)")
    scatter_fig, lr_emp = plot_regression_scatter_with_sklearn(df["Unemployment_pct"], df["GDP_Growth"], "Unemployment (%)", "GDP Growth (%)", "Unemployment vs GDP Growth (OLS)", color=PALETTE["employment"])
    st.plotly_chart(scatter_fig, use_container_width=True)

    coef = lr_emp.coef_[0]
    intercept = lr_emp.intercept_
    st.markdown(f"**OLS result (sklearn):** GDP_Growth = {intercept:.3f} + {coef:.3f} * Unemployment")


def show_remittances_page(df: pd.DataFrame):
    st.header("Remittances — External Cushion")
    fig = px.bar(df, x="Year", y="Remittances_BnUSD", title="Remittances (Bn USD)", color_discrete_sequence=[PALETTE["remit"]])
    st.plotly_chart(fig, use_container_width=True)
    df["Remit_pct_of_imports"] = (df["Remittances_BnUSD"] / df["Imports_BnUSD"] * 100).round(2)
    fig2 = px.line(df, x="Year", y="Remit_pct_of_imports", title="Remittances as % of Imports", markers=True)
    st.plotly_chart(fig2, use_container_width=True)


def show_correlation(df: pd.DataFrame):
    st.header("Correlation Matrix")
    corr_df = df[["GDP_Growth", "Inflation", "Exports_BnUSD", "Imports_BnUSD", "Remittances_BnUSD", "Unemployment_pct", "GDP_per_capita_USD", "Trade_Balance_BnUSD"]].corr()
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)


def show_ml_page(df: pd.DataFrame, use_xgb: bool = False, use_nn: bool = False):
    st.header("ML & Forecast")
    st.markdown("Regression models to explain/predict GDP Growth and Neural Net for GDP per capita (optional).")
    X, y_gdp, y_gdppc = prepare_ml_features(df)
    st.markdown("Features used:")
    st.write(list(X.columns))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        btn_rf = st.button("Train regression models (LR + RF + XGB if enabled)")
    with col2:
        btn_nn = st.button("Train neural network (GDP per capita)") if TF_AVAILABLE else st.button("Train NN (TF not installed)")
    with col3:
        btn_clear = st.button("Clear saved models")

    if btn_clear:
        st.session_state.pop("reg_results", None)
        st.session_state.pop("nn_res", None)
        st.success("Cleared saved models from session state.")

    if btn_rf:
        with st.spinner("Training regression models..."):
            reg_results = regression_pipeline_with_sklearn(X, y_gdp, use_xgb=use_xgb)
            st.session_state["reg_results"] = reg_results
            st.success("Regression models trained.")
            for name, info in reg_results.items():
                st.markdown(f"**{name}** — RMSE: {info['metrics']['RMSE']:.3f}, MAE: {info['metrics']['MAE']:.3f}, R2: {info['metrics']['R2']:.3f}")
            best = min(reg_results.items(), key=lambda kv: kv[1]["metrics"]["RMSE"])[0]
            st.info(f"Best regression model (in-sample RMSE): {best}")
            # plot in-sample actual vs pred
            best_pred = reg_results[best]["pred"]
            fig = plot_actual_vs_predicted_years(df["Year"].values, df["GDP_Growth"].values, best_pred, f"Actual vs Predicted GDP Growth ({best})")
            st.plotly_chart(fig, use_container_width=True)

    if btn_nn:
        if TF_AVAILABLE:
            with st.spinner("Training Neural Network..."):
                nn_res = train_neural_network(X, y_gdppc, epochs=500, verbose=0)
                if "error" in nn_res:
                    st.error("Neural net training failed: " + str(nn_res["error"]))
                else:
                    st.session_state["nn_res"] = nn_res
                    st.success("Neural network trained.")
                    st.markdown(f"NN in-sample metrics: RMSE {nn_res['metrics']['RMSE']:.2f}, R2 {nn_res['metrics']['R2']:.2f}")
                    fig = plot_actual_vs_predicted_years(df["Year"].values, df["GDP_per_capita_USD"].values, nn_res["pred"], "Actual vs Predicted GDP per capita (NN)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("TensorFlow not installed. Install to enable NN training.")

    if "reg_results" in st.session_state:
        st.subheader("Regression comparison (in-sample)")
        rows = []
        for name, info in st.session_state["reg_results"].items():
            rows.append({"Model": name, **info["metrics"]})
        st.table(pd.DataFrame(rows).set_index("Model"))


def show_insights(df: pd.DataFrame):
    st.header("Automated Insights")
    ins = human_friendly_insights(df)
    for i, s in enumerate(ins, 1):
        st.markdown(f"**{i}.** {s}")
    st.markdown("---")
    st.markdown("### Story angles (suggestions)")
    st.write([
        "1. Explain how remittances have cushioned current account pressures.",
        "2. Relate inflation spikes to policy or external events and propose timing-sensitive measures.",
        "3. Discuss structural policies to reduce trade deficits.",
        "4. Link unemployment dynamics to short-term stabilization policy."
    ])


def show_download(df: pd.DataFrame):
    st.header("Download Data")
    st.markdown("Download the generated dummy dataset (CSV).")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "pakistan_dummy_economics.csv", "text/csv")


def main():
    st.sidebar.markdown("## Navigation")
    menu = st.sidebar.radio("", [
        "Overview",
        "GDP",
        "Inflation",
        "Trade",
        "Employment",
        "Remittances",
        "Correlation",
        "ML & Forecast",
        "Insights",
        "Download Data"
    ])
    st.sidebar.markdown("---")
    years_to_use = st.sidebar.slider("Years to include (most recent)", min_value=10, max_value=24, value=24, step=1)
    st.sidebar.markdown("---")
    use_xgb = st.sidebar.checkbox("Enable XGBoost (if installed)", value=False and XGBOOST_AVAILABLE)
    use_nn = st.sidebar.checkbox("Enable Neural Net (TF)", value=False and TF_AVAILABLE)
    show_raw = st.sidebar.checkbox("Show raw table", value=False)

    df_full = generate_dummy_pakistan_data()
    df = df_full.tail(years_to_use).reset_index(drop=True)

    st.markdown('<div class="big-title">Pakistan Economic Insights — 2000 to 2023</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dummy dataset generated for storytelling, visualization and ML exploration.</div>', unsafe_allow_html=True)
    st.markdown("")

    kpis = format_kpis(df)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
    with c1:
        st.markdown(f"<div class='kpi'><strong>GDP Growth</strong><br><span style='font-size:20px'>{kpis['GDP_Growth']}</span><br><small>{kpis['GDP_Growth_delta']} vs prev</small></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><strong>Inflation</strong><br><span style='font-size:20px'>{kpis['Inflation']}</span><br><small>{kpis['Inflation_delta']} vs prev</small></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><strong>Trade Balance</strong><br><span style='font-size:20px'>{kpis['Trade_Balance']}</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi'><strong>GDP per capita</strong><br><span style='font-size:20px'>{kpis['GDP_per_capita']}</span></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='kpi'><strong>Unemployment</strong><br><span style='font-size:20px'>{kpis['Unemployment']}</span></div>", unsafe_allow_html=True)

    if show_raw:
        st.write("Data sample:")
        st.dataframe(df.tail(10))

    if menu == "Overview":
        show_overview(df)
    elif menu == "GDP":
        show_gdp_page(df)
    elif menu == "Inflation":
        show_inflation_page(df)
    elif menu == "Trade":
        show_trade_page(df)
    elif menu == "Employment":
        show_employment_page(df)
    elif menu == "Remittances":
        show_remittances_page(df)
    elif menu == "Correlation":
        show_correlation(df)
    elif menu == "ML & Forecast":
        show_ml_page(df, use_xgb=use_xgb, use_nn=use_nn)
    elif menu == "Insights":
        show_insights(df)
    elif menu == "Download Data":
        show_download(df)


if __name__ == "__main__":
    main()
