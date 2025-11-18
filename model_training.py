# app.py
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.express as px
import xgboost as xgb

# LSTM imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Walmart Sales Forecasting (Advanced)", layout="wide")
st.title("🏬 Walmart Sales Forecasting & Business Insights (Advanced)")

st.markdown(
    "Upload the Walmart dataset CSV (columns like **Store, Date, Weekly_Sales, Holiday_Flag, "
    "Temperature, Fuel_Price, CPI, Unemployment**)."
)

uploaded_file = st.file_uploader("Upload Walmart Sales CSV", type=["csv"])

if not uploaded_file:
    st.info("👆 Please upload Walmart sales dataset (CSV) to begin.")
    st.stop()

# -------------------------
# Load & basic cleaning
# -------------------------
data = pd.read_csv(uploaded_file)
data.columns = [c.strip() for c in data.columns]

required_cols = {"Store", "Date", "Weekly_Sales"}
missing = required_cols - set(data.columns)
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

# Parse dates (handle dd-mm-yyyy)
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
if data["Date"].isna().any():
    st.warning("Some dates could not be parsed and were dropped.")
    data = data.dropna(subset=["Date"])

# Ensure numeric sales
data["Weekly_Sales"] = pd.to_numeric(data["Weekly_Sales"], errors="coerce")
data = data.dropna(subset=["Weekly_Sales"])

# Inject synthetic external features if missing (for demo / resume purposes)
if "CPI" not in data.columns:
    data["CPI"] = 200 + np.random.normal(0, 5, size=len(data))
if "Fuel_Price" not in data.columns:
    data["Fuel_Price"] = 3 + np.random.normal(0, 0.1, size=len(data))
if "Unemployment" not in data.columns:
    data["Unemployment"] = 6 + np.random.normal(0, 0.5, size=len(data))
if "Holiday_Flag" not in data.columns:
    data["Holiday_Flag"] = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])


# Sort
data = data.sort_values(["Store", "Date"]).reset_index(drop=True)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
stores = np.sort(data["Store"].unique())
store_num = st.sidebar.selectbox("Select Store", stores)
model_name = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "XGBoost", "LSTM"])
future_weeks = st.sidebar.slider("Forecast Horizon (weeks)", 10, 52, 30)

# -------------------------
# What-If sliders always visible for XGBoost
# -------------------------
if model_name == "XGBoost":
    st.sidebar.subheader("🔧 What-If Scenario (XGBoost)")

    # Always show sliders (default 0% if column missing)
    adjust_cpi = st.sidebar.slider("CPI Adjustment (%)", -20, 20, 0)
    adjust_fuel = st.sidebar.slider("Fuel Price Adjustment (%)", -20, 20, 0)
    adjust_unemp = st.sidebar.slider("Unemployment Adjustment (%)", -20, 20, 0)

    # Holiday override always available
    holiday_override = st.sidebar.selectbox(
        "Holiday Flag Override",
        ["Keep as is", "Force Holiday", "Force Non-Holiday"]
    )
else:
    adjust_cpi = adjust_fuel = adjust_unemp = 0
    holiday_override = "Keep as is"


# -------------------------
# Store-specific series
# -------------------------
store_data = (
    data.loc[data["Store"] == store_num, ["Date", "Weekly_Sales"]]
    .rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    .dropna()
)

if store_data.empty or len(store_data) < 20:
    st.error("Not enough data for this store to build a forecast (need at least ~20 points).")
    st.stop()

# Enforce weekly frequency and fill small gaps
store_series = store_data.set_index("ds").sort_index()["y"]
store_series = store_series.asfreq("W", method="pad")

st.subheader(f"📈 Historical Weekly Sales — Store {store_num}")
st.line_chart(store_series)

# -------------------------
# Validation split
# -------------------------
test_horizon = min(12, max(4, int(len(store_series) * 0.2)))
if len(store_series) <= test_horizon + 5:
    test_horizon = max(4, len(store_series) // 5)

train = store_series.iloc[:-test_horizon]
test = store_series.iloc[-test_horizon:]

# -------------------------
# Modeling helpers
# -------------------------
def run_prophet(train_series, full_series, future_weeks, test_series):
    train_df = train_series.reset_index().rename(columns={"ds": "ds", "y": "y"})
    m = Prophet()
    m.fit(train_df)

    future_valid = m.make_future_dataframe(periods=len(test_series), freq="W")
    fc_valid = m.predict(future_valid).set_index("ds")["yhat"].iloc[-len(test_series):]
    rmse = sqrt(mean_squared_error(test_series.values, fc_valid.values))

    full_df = full_series.reset_index().rename(columns={"ds": "ds", "y": "y"})
    m_full = Prophet()
    m_full.fit(full_df)
    future_full = m_full.make_future_dataframe(periods=future_weeks, freq="W")
    forecast_full = m_full.predict(future_full)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return forecast_full, rmse


def run_arima(train_series, full_series, future_weeks, test_series):
    from statsmodels.tsa.arima.model import ARIMA
    order = (1, 1, 1)
    arima_model = ARIMA(train_series, order=order)
    arima_res = arima_model.fit()
    fc_valid = arima_res.forecast(steps=len(test_series))
    rmse = sqrt(mean_squared_error(test_series.values, fc_valid.values))

    arima_full = ARIMA(full_series, order=order).fit()
    arima_future = arima_full.forecast(steps=future_weeks)
    last_date = full_series.index[-1]
    future_index = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq="W")
    forecast_full = pd.DataFrame({"ds": future_index, "yhat": arima_future.values})
    forecast_full["yhat_lower"] = np.nan
    forecast_full["yhat_upper"] = np.nan
    return forecast_full, rmse


def run_xgboost_for_store(global_data, store_number, future_weeks,
                          cpi_adj=0, fuel_adj=0, unemp_adj=0, holiday_override="Keep as is"):
    """
    Build lag + external features and train XGBoost.
    Applies adjustments to external variables for what-if simulation.
    Returns (forecast_df, rmse, importance_df) or (None, None, None) if not enough data.
    """
    # Prepare store dataframe and align weekly
    df_store = global_data.loc[global_data["Store"] == store_number].copy()
    df_store = df_store.sort_values("Date")[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    df_store["ds"] = pd.to_datetime(df_store["ds"], dayfirst=True, errors="coerce")
    df_store = df_store.set_index("ds").asfreq("W", method="pad").reset_index()

    # Lag features
    df_store["lag1"] = df_store["y"].shift(1)
    df_store["lag2"] = df_store["y"].shift(2)

    # External features aligned for this store if present
    external_cols = []
    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]:
        if col in global_data.columns:
            ext_series = global_data.loc[global_data["Store"] == store_number, ["Date", col]].copy()
            ext_series.columns = ["ds", col]
            ext_series["ds"] = pd.to_datetime(ext_series["ds"], dayfirst=True, errors="coerce")
            ext_series = ext_series.set_index("ds").asfreq("W", method="pad").reset_index()
            df_store = df_store.merge(ext_series, on="ds", how="left")
            external_cols.append(col)

    df_store = df_store.dropna().reset_index(drop=True)
    if len(df_store) < 10:
        return None, None, None

    features = ["lag1", "lag2"] + external_cols
    for f in features:
        df_store[f] = pd.to_numeric(df_store[f], errors="coerce")
    df_store = df_store.dropna(subset=features + ["y"])
    if len(df_store) < 10:
        return None, None, None

    X = df_store[features]
    y = df_store["y"]

    # Train/test split for validation
    test_size = min(12, int(0.2 * len(df_store)))
    if test_size < 1:
        test_size = 1
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))

    # Prepare last known external values and apply adjustments for what-if
    last_externals = {col: float(df_store[col].iloc[-1]) for col in external_cols} if external_cols else {}
    # apply adjustments
    if "CPI" in last_externals:
        last_externals["CPI"] *= (1 + cpi_adj / 100.0)
    if "Fuel_Price" in last_externals:
        last_externals["Fuel_Price"] *= (1 + fuel_adj / 100.0)
    if "Unemployment" in last_externals:
        last_externals["Unemployment"] *= (1 + unemp_adj / 100.0)
    if "Holiday_Flag" in last_externals:
        if holiday_override == "Force Holiday":
            last_externals["Holiday_Flag"] = 1
        elif holiday_override == "Force Non-Holiday":
            last_externals["Holiday_Flag"] = 0
        # else keep as is

    # Forecast future using naive propagation of externals and lag feedback
    last_row = X.iloc[-1:].to_dict(orient="records")[0]
    future_preds = []
    last_lag1 = last_row.get("lag1", X.iloc[-1]["lag1"])
    last_lag2 = last_row.get("lag2", X.iloc[-1]["lag2"])

    for i in range(future_weeks):
        x_input = {"lag1": last_lag1, "lag2": last_lag2}
        for col in external_cols:
            x_input[col] = last_externals[col]
        X_future = pd.DataFrame([x_input])
        pred = model.predict(X_future)[0]
        future_preds.append(pred)
        # update lags
        last_lag2 = last_lag1
        last_lag1 = pred

    last_date = df_store["ds"].iloc[-1]
    future_index = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq="W")
    forecast_df = pd.DataFrame({"ds": future_index, "yhat": future_preds})
    forecast_df["yhat_lower"] = np.nan
    forecast_df["yhat_upper"] = np.nan

    importance = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance", ascending=False)

    return forecast_df, rmse, importance_df


def run_lstm(store_series, future_weeks, test_series):
    values = store_series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    window = 5
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    if len(X) < 10:
        return None, None

    X = X.reshape((X.shape[0], X.shape[1], 1))
    test_size = min(12, int(0.2 * len(X)))
    if test_size < 1:
        test_size = 1
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

    preds = model.predict(X_test)
    preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = sqrt(mean_squared_error(y_test_rescaled, preds_rescaled))

    last_window = scaled[-window:].reshape(1, window, 1)
    future_preds = []
    for i in range(future_weeks):
        pred_scaled = model.predict(last_window)[0, 0]
        pred_value = scaler.inverse_transform([[pred_scaled]])[0, 0]
        future_preds.append(pred_value)
        last_window = np.append(last_window[:, 1:, :], [[[pred_scaled]]], axis=1)

    last_date = store_series.index[-1]
    future_index = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq="W")
    forecast_df = pd.DataFrame({"ds": future_index, "yhat": future_preds})
    forecast_df["yhat_lower"] = np.nan
    forecast_df["yhat_upper"] = np.nan
    return forecast_df, rmse

# -------------------------
# Run selected model & (for XGBoost) scenario
# -------------------------
with st.spinner("Training and forecasting..."):
    full_series = store_series
    importance_df = None
    if model_name == "Prophet":
        forecast_full, rmse = run_prophet(train, full_series, future_weeks, test)
    elif model_name == "ARIMA":
        forecast_full, rmse = run_arima(train, full_series, future_weeks, test)
    elif model_name == "XGBoost":
        # base forecast
        forecast_base, rmse_base, importance_df = run_xgboost_for_store(data, store_num, future_weeks)
        # scenario forecast with adjustments
        forecast_scenario, rmse_scenario, _ = run_xgboost_for_store(
            data, store_num, future_weeks,
            cpi_adj=adjust_cpi, fuel_adj=adjust_fuel, unemp_adj=adjust_unemp,
            holiday_override=holiday_override
        )
        # choose base to show numeric RMSE (we can show both later)
        forecast_full, rmse = forecast_base, rmse_base
    else:  # LSTM
        forecast_full, rmse = run_lstm(store_series, future_weeks, test)
        importance_df = None

# handle failures
if model_name == "XGBoost" and (forecast_full is None):
    st.error("Not enough/usable feature data for XGBoost for this store. Try Prophet/ARIMA/LSTM or choose another store.")
    st.stop()
if model_name == "LSTM" and (forecast_full is None):
    st.error("Not enough data to train LSTM reliably. Try other models or choose another store.")
    st.stop()

# -------------------------
# Plots & tables
# -------------------------
st.subheader(f"🔮 Forecast — Store {store_num} ({model_name}) — Next {future_weeks} Weeks")

# For XGBoost show base vs scenario if scenario exists
if model_name == "XGBoost" and (forecast_scenario is not None):
    fig = px.line(forecast_base, x="ds", y="yhat",
                  labels={"ds": "Week", "yhat": "Forecasted Sales"},
                  title=f"Store {store_num} — XGBoost Forecast (Base vs What-If)")
    
    # Add base forecast (blue line)
    fig.add_scatter(x=forecast_base["ds"], y=forecast_base["yhat"],
                    mode="lines", name="Base Forecast", line=dict(color="blue"))
    
    # Add scenario forecast (red dashed line)
    fig.add_scatter(x=forecast_scenario["ds"], y=forecast_scenario["yhat"],
                    mode="lines", name="What-If Scenario", line=dict(color="red", dash="dot"))
    
    st.plotly_chart(fig, use_container_width=True)

else:
    fig = px.line(forecast_full, x="ds", y="yhat",
                  labels={"ds": "Week", "yhat": "Forecasted Sales"},
                  title=f"Store {store_num} — {model_name} Forecast")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Validation RMSE (last {test_horizon} weeks): **{rmse:,.2f}**")
    st.subheader("📑 Forecast Data (Next Period)")
    st.dataframe(forecast_full.tail(future_weeks))

    # ============================
    # KPI SUMMARY OF WHAT-IF IMPACT
    # ============================
    base_mean = forecast_base["yhat"].mean()
    scenario_mean = forecast_scenario["yhat"].mean()
    impact_pct = ((scenario_mean - base_mean) / base_mean) * 100 if base_mean else 0

    if impact_pct > 0:
        st.success(f"📈 What-If Scenario projects an **increase of {impact_pct:.2f}%** in forecasted sales compared to base.")
    elif impact_pct < 0:
        st.error(f"📉 What-If Scenario projects a **decrease of {abs(impact_pct):.2f}%** in forecasted sales compared to base.")
    else:
        st.info("ℹ️ What-If Scenario shows **no change** compared to base forecast.")


# Download CSV for the shown (primary) forecast
csv_bytes = forecast_full.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Forecast CSV (Primary)",
    data=csv_bytes,
    file_name=f"store_{store_num}_{model_name}_forecast.csv",
    mime="text/csv",
)

# If XGBoost scenario exists allow download too
if model_name == "XGBoost" and (forecast_scenario is not None):
    csv_bytes_s = forecast_scenario.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast CSV (What-If)",
        data=csv_bytes_s,
        file_name=f"store_{store_num}_xgboost_forecast_whatif.csv",
        mime="text/csv",
    )

# -------------------------
# Business Insights (All Stores)
# -------------------------
st.markdown("---")
st.subheader("📊 Business Insights (All Stores)")

col1, col2 = st.columns(2)
with col1:
    top_stores = data.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False).head(5)
    st.write("🏆 Top 5 Stores by Average Weekly Sales")
    st.bar_chart(top_stores)

with col2:
    if "Holiday_Flag" in data.columns:
        holiday_sales = data.loc[data["Holiday_Flag"] == 1, "Weekly_Sales"].mean()
        nonholiday_sales = data.loc[data["Holiday_Flag"] == 0, "Weekly_Sales"].mean()
        uplift = ((holiday_sales - nonholiday_sales) / nonholiday_sales) * 100 if nonholiday_sales else np.nan
        st.write("🎉 Holiday Impact on Sales")
        st.write(f"Avg Weekly Sales on Holidays: **{holiday_sales:,.2f}**")
        st.write(f"Avg Weekly Sales on Non-Holidays: **{nonholiday_sales:,.2f}**")
        if pd.notna(uplift):
            st.write(f"Estimated Uplift on Holidays: **{uplift:.2f}%**")

# Trend for selected store
trend = store_series.to_frame("y").copy()
trend["y_shifted"] = trend["y"].shift(1)
trend["growth"] = trend["y"] - trend["y_shifted"]
avg_growth = trend["growth"].mean()
if pd.notna(avg_growth):
    if avg_growth > 0:
        st.success(f"📈 Store {store_num} shows an upward trend (+{avg_growth:,.2f} avg weekly growth).")
    else:
        st.error(f"📉 Store {store_num} shows a downward trend ({avg_growth:,.2f} avg weekly decline).")

# Feature importance (XGBoost)
if model_name == "XGBoost" and (importance_df is not None):
    st.markdown("---")
    st.subheader("📌 Feature Importance (XGBoost)")
    st.bar_chart(importance_df.set_index("Feature")["Importance"])
