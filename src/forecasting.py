import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def create_time_series_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Constructs financial and calendar lag features for tabular ML time-series models.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values("date", inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Calendar / Business Day Features
    data["dayofweek"] = data["date"].dt.dayofweek
    data["dayofmonth"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["is_month_start"] = (data["dayofmonth"] <= 3).astype(int)
    data["is_month_end"] = (data["dayofmonth"] >= 27).astype(int)
    data["is_payroll_week"] = ((data["dayofmonth"] >= 25) & (data["dayofmonth"] <= 28)).astype(int)

    # Fourier terms for smooth seasonality
    day_of_year = data["date"].dt.dayofyear
    data["sin_annual"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["cos_annual"] = np.cos(2 * np.pi * day_of_year / 365.25)
    data["sin_weekly"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
    data["cos_weekly"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)

    # Lag features
    for lag in [1, 2, 7, 14, 21, 28, 30]:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    # Rolling window aggregations
    data["rolling_mean_7"] = data[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    data["rolling_std_7"] = data[target_col].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    data["rolling_mean_14"] = data[target_col].shift(1).rolling(window=14, min_periods=1).mean()
    data["rolling_mean_30"] = data[target_col].shift(1).rolling(window=30, min_periods=1).mean()

    return data

def train_and_evaluate_models(
    ledger_df: pd.DataFrame,
    target_col: str = "inflow",
    test_days: int = 60
) -> dict:
    """
    Benchmarks multiple models on the holdout test period:
    1. Baseline Exponential Smoothing (Holt-Winters)
    2. Ridge Regressor
    3. Random Forest Regressor
    4. Gradient Boosting Regressor
    """
    df_feat = create_time_series_features(ledger_df, target_col).dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df_feat.columns if c not in ["date", "inflow", "outflow", "net_cash_flow", "cash_balance"]]
    
    train_df = df_feat.iloc[:-test_days].copy()
    test_df = df_feat.iloc[-test_days:].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4)
    }

    results = {}
    
    # ML Models
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)  # inflows/outflows cannot be negative

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        # Handle zeros safely for MAPE
        non_zero_mask = y_test > 10.0
        mape = mean_absolute_percentage_error(y_test[non_zero_mask], preds[non_zero_mask]) if non_zero_mask.sum() > 0 else 0.0

        results[name] = {
            "model": model,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE_%": round(mape * 100, 2),
            "predictions": preds
        }

    # Baseline Holt-Winters Exponential Smoothing
    try:
        hw_train = ledger_df[target_col].iloc[:-test_days]
        hw_model = ExponentialSmoothing(hw_train, seasonal_periods=7, trend="add", seasonal="add").fit()
        hw_preds = hw_model.forecast(test_days)
        hw_preds = np.clip(hw_preds.values, 0, None)
        hw_mae = mean_absolute_error(y_test, hw_preds)
        hw_rmse = np.sqrt(mean_squared_error(y_test, hw_preds))
        non_zero = y_test > 10.0
        hw_mape = mean_absolute_percentage_error(y_test[non_zero], hw_preds[non_zero]) if non_zero.sum() > 0 else 0.0

        results["Holt-Winters Baseline"] = {
            "model": hw_model,
            "MAE": round(hw_mae, 2),
            "RMSE": round(hw_rmse, 2),
            "MAPE_%": round(hw_mape * 100, 2),
            "predictions": hw_preds
        }
    except Exception as e:
        pass

    # Pick best model based on MAE
    best_model_name = min(results.keys(), key=lambda k: results[k]["MAE"])

    return {
        "results_summary": pd.DataFrame([
            {"Model": m, "MAE (INR)": d["MAE"], "RMSE (INR)": d["RMSE"], "MAPE (%)": d["MAPE_%"]}
            for m, d in results.items()
        ]),
        "best_model_name": best_model_name,
        "best_model_data": results[best_model_name],
        "test_actuals": y_test.values,
        "test_dates": test_df["date"].dt.strftime("%Y-%m-%d").values,
        "feature_cols": feature_cols
    }

def get_single_step_features(history: list, f_date: pd.Timestamp) -> dict:
    """
    Ultra-fast scalar feature generator for single-step recursive prediction.
    Avoids re-running heavy rolling and shift transforms across the entire historical DataFrame.
    """
    hist_arr = np.array(history)
    dayofweek = f_date.dayofweek
    dayofmonth = f_date.day
    month = f_date.month
    quarter = f_date.quarter
    dayofyear = f_date.dayofyear

    feat = {
        "dayofweek": dayofweek,
        "dayofmonth": dayofmonth,
        "month": month,
        "quarter": quarter,
        "is_weekend": int(dayofweek in [5, 6]),
        "is_month_start": int(dayofmonth <= 3),
        "is_month_end": int(dayofmonth >= 27),
        "is_payroll_week": int(25 <= dayofmonth <= 28),
        "sin_annual": np.sin(2 * np.pi * dayofyear / 365.25),
        "cos_annual": np.cos(2 * np.pi * dayofyear / 365.25),
        "sin_weekly": np.sin(2 * np.pi * dayofweek / 7.0),
        "cos_weekly": np.cos(2 * np.pi * dayofweek / 7.0),
        "lag_1": hist_arr[-1],
        "lag_2": hist_arr[-2],
        "lag_7": hist_arr[-7],
        "lag_14": hist_arr[-14],
        "lag_21": hist_arr[-21],
        "lag_28": hist_arr[-28],
        "lag_30": hist_arr[-30],
        "rolling_mean_7": float(np.mean(hist_arr[-7:])),
        "rolling_std_7": float(np.std(hist_arr[-7:])),
        "rolling_mean_14": float(np.mean(hist_arr[-14:])),
        "rolling_mean_30": float(np.mean(hist_arr[-30:]))
    }
    return feat

def generate_future_forecast(
    ledger_df: pd.DataFrame,
    horizon_days: int = 60,
    model_type: str = "Gradient Boosting"
) -> pd.DataFrame:
    """
    Generates multi-step recursive forecasts for both Cash Inflows and Outflows,
    then combines them to project the Net Cash Flow and Cash Balance curve with confidence bounds.
    """
    last_date = pd.to_datetime(ledger_df["date"].max())
    current_cash = float(ledger_df["cash_balance"].iloc[-1])

    # 1. Feature extraction for training
    feat_in_base = create_time_series_features(ledger_df, "inflow")
    feat_out_base = create_time_series_features(ledger_df, "outflow")
    feature_cols = [c for c in feat_in_base.columns if c not in ["date", "inflow", "outflow", "net_cash_flow", "cash_balance"]]

    X_train_in = feat_in_base[feature_cols].dropna()
    y_train_in = feat_in_base.loc[X_train_in.index, "inflow"]

    X_train_out = feat_out_base[feature_cols].dropna()
    y_train_out = feat_out_base.loc[X_train_out.index, "outflow"]

    if model_type == "Random Forest":
        reg_in = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_out = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Ridge Regression":
        reg_in = Ridge(alpha=1.0)
        reg_out = Ridge(alpha=1.0)
    else:
        reg_in = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
        reg_out = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)

    reg_in.fit(X_train_in, y_train_in)
    reg_out.fit(X_train_out, y_train_out)

    forecast_records = []
    
    # Maintain rolling history lists
    inflow_history = list(ledger_df["inflow"].values)
    outflow_history = list(ledger_df["outflow"].values)
    running_cash = current_cash

    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]

    for f_date in future_dates:
        feat_in_dict = get_single_step_features(inflow_history, f_date)
        feat_out_dict = get_single_step_features(outflow_history, f_date)

        X_step_in = pd.DataFrame([feat_in_dict])[feature_cols]
        X_step_out = pd.DataFrame([feat_out_dict])[feature_cols]

        pred_in = max(0.0, float(reg_in.predict(X_step_in)[0]))
        pred_out = max(0.0, float(reg_out.predict(X_step_out)[0]))
        pred_net = pred_in - pred_out

        inflow_history.append(pred_in)
        outflow_history.append(pred_out)
        running_cash += pred_net

        forecast_records.append({
            "date": f_date.strftime("%Y-%m-%d"),
            "forecast_inflow": round(pred_in, 2),
            "forecast_outflow": round(pred_out, 2),
            "forecast_net_flow": round(pred_net, 2),
            "projected_cash_balance": round(running_cash, 2)
        })

    forecast_df = pd.DataFrame(forecast_records)

    # 2. Add Confidence Intervals (80% and 95%) based on historical residual standard deviation
    hist_std_net = float(ledger_df["net_cash_flow"].tail(60).std())
    steps = np.arange(1, len(forecast_df) + 1)
    uncertainty_80 = 1.28 * hist_std_net * np.sqrt(steps / 7.0)
    uncertainty_95 = 1.96 * hist_std_net * np.sqrt(steps / 7.0)

    forecast_df["cash_balance_lower_80"] = forecast_df["projected_cash_balance"] - uncertainty_80
    forecast_df["cash_balance_upper_80"] = forecast_df["projected_cash_balance"] + uncertainty_80

    forecast_df["cash_balance_lower_95"] = forecast_df["projected_cash_balance"] - uncertainty_95
    forecast_df["cash_balance_upper_95"] = forecast_df["projected_cash_balance"] + uncertainty_95

    return forecast_df

if __name__ == "__main__":
    ledger = pd.read_csv("data/processed/daily_cash_flow.csv")
    print("Evaluating models on Inflow forecasting...")
    eval_res = train_and_evaluate_models(ledger, target_col="inflow", test_days=60)
    print(eval_res["results_summary"].to_string(index=False))
    print(f"\nBest Model: {eval_res['best_model_name']}")
    
    print("\nGenerating 60-day multi-step recursive cash forecast...")
    future_fc = generate_future_forecast(ledger, horizon_days=30, model_type=eval_res['best_model_name'])
    print(future_fc.head(10).to_string(index=False))
