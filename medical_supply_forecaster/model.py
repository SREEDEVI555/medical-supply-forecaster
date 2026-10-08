import pandas as pd
import numpy as np
import warnings
import csv
import os

warnings.filterwarnings("ignore")

# ML & Statistical Models imports with graceful fallbacks
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGBOOST = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_EXP_SMOOTH = True
except Exception:
    HAS_EXP_SMOOTH = False

def load_data(filepath="medical_demand.csv"):
    if not os.path.exists(filepath):
        filepath = "medical_supply_data.csv"
        
    df = pd.read_csv(filepath)
    
    col_map = {
        "Date": "date",
        "Clinic_ID": "centre",
        "Medicine_Name": "item",
        "Quantity_Consumed": "demand",
        "Stockout_Days": "stockout_days"
    }
    df = df.rename(columns=col_map)
    
    if os.path.exists("clinics.csv"):
        df_c = pd.read_csv("clinics.csv")
        cmap = dict(zip(df_c["Clinic_ID"], df_c["Clinic_Name"]))
        if "centre" in df.columns:
            df["centre_name"] = df["centre"].map(cmap).fillna("")
            df["centre_display"] = df["centre"] + " - " + df["centre_name"]
            df["centre"] = df["centre_display"]
            
    df["date"] = pd.to_datetime(df["date"])
    return df

def calculate_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calculate_mape(y_true, y_pred):
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def fit_predict_arima(series, test_len=2):
    train = series[:-test_len]
    test = series[-test_len:]
    
    if HAS_ARIMA:
        try:
            model_eval = ARIMA(train, order=(1, 1, 1)).fit()
            test_preds = model_eval.forecast(steps=test_len)
            mae = calculate_mae(test, test_preds)
            rmse = calculate_rmse(test, test_preds)
            mape = calculate_mape(test, test_preds)
            
            model_full = ARIMA(series, order=(1, 1, 1)).fit()
            next_pred = float(model_full.forecast(steps=1).iloc[0])
            return max(5, int(round(next_pred))), round(mae, 2), round(rmse, 2), round(mape, 2)
        except Exception:
            pass

    test_preds = np.repeat(train.mean(), test_len)
    mae = calculate_mae(test, test_preds)
    rmse = calculate_rmse(test, test_preds)
    mape = calculate_mape(test, test_preds)
    next_pred = float(series[-3:].mean())
    return max(5, int(round(next_pred))), round(mae, 2), round(rmse, 2), round(mape, 2)

def fit_predict_prophet(df_item, test_len=2):
    if HAS_PROPHET:
        pdf = df_item[["date", "demand"]].rename(columns={"date": "ds", "demand": "y"})
        train = pdf.iloc[:-test_len]
        test = pdf.iloc[-test_len:]
        
        try:
            m_eval = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            m_eval.fit(train)
            future_eval = m_eval.make_future_dataframe(periods=test_len, freq="MS")
            forecast_eval = m_eval.predict(future_eval)
            test_preds = forecast_eval["yhat"].iloc[-test_len:].values
            mae = calculate_mae(test["y"].values, test_preds)
            rmse = calculate_rmse(test["y"].values, test_preds)
            mape = calculate_mape(test["y"].values, test_preds)
            
            m_full = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            m_full.fit(pdf)
            future_full = m_full.make_future_dataframe(periods=1, freq="MS")
            forecast_full = m_full.predict(future_full)
            next_pred = float(forecast_full["yhat"].iloc[-1])
            return max(5, int(round(next_pred))), round(mae, 2), round(rmse, 2), round(mape, 2)
        except Exception:
            pass
            
    mae, rmse, mape, next_pred = fit_predict_exponential_smoothing(df_item["demand"].values, test_len)
    return next_pred, mae, rmse, mape

def fit_predict_exponential_smoothing(series_vals, test_len=2):
    train = series_vals[:-test_len]
    test = series_vals[-test_len:]
    if HAS_EXP_SMOOTH:
        try:
            model = ExponentialSmoothing(train, initialization_method="estimated").fit()
            test_preds = model.forecast(test_len)
            mae = calculate_mae(test, test_preds)
            rmse = calculate_rmse(test, test_preds)
            mape = calculate_mape(test, test_preds)
            
            model_full = ExponentialSmoothing(series_vals, initialization_method="estimated").fit()
            next_pred = float(model_full.forecast(1)[0])
            return round(mae, 2), round(rmse, 2), round(mape, 2), max(5, int(round(next_pred)))
        except Exception:
            pass

    test_preds = np.repeat(train.mean(), test_len)
    mae = calculate_mae(test, test_preds)
    rmse = calculate_rmse(test, test_preds)
    mape = calculate_mape(test, test_preds)
    next_pred = float(series_vals[-3:].mean())
    return round(mae, 2), round(rmse, 2), round(mape, 2), max(5, int(round(next_pred)))

def fit_predict_xgboost(df_item, test_len=2):
    s = df_item.copy().sort_values("date")
    s["lag_1"] = s["demand"].shift(1)
    s["lag_2"] = s["demand"].shift(2)
    s["roll_mean"] = s["demand"].shift(1).rolling(3, min_periods=1).mean()
    s["month_num"] = s["date"].dt.month
    
    clean_df = s.dropna(subset=["lag_1", "lag_2"]).copy()
    features = ["month_num", "lag_1", "lag_2", "roll_mean"]
    
    X = clean_df[features]
    y = clean_df["demand"]
    
    X_train = X.iloc[:-test_len]
    y_train = y.iloc[:-test_len]
    X_test = X.iloc[-test_len:]
    y_test = y.iloc[-test_len:]
    
    if HAS_XGBOOST:
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        
    try:
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        mae = calculate_mae(y_test, test_preds)
        rmse = calculate_rmse(y_test, test_preds)
        mape = calculate_mape(y_test, test_preds)
        
        model.fit(X, y)
        last_row = clean_df.iloc[-1]
        next_features = pd.DataFrame([{
            "month_num": 1,
            "lag_1": last_row["demand"],
            "lag_2": last_row["lag_1"],
            "roll_mean": clean_df["demand"].iloc[-3:].mean()
        }])[features]
        
        next_pred = float(model.predict(next_features)[0])
    except Exception:
        test_preds = np.repeat(y_train.mean(), test_len)
        mae = calculate_mae(y_test, test_preds)
        rmse = calculate_rmse(y_test, test_preds)
        mape = calculate_mape(y_test, test_preds)
        next_pred = float(y.mean())
        
    return max(5, int(round(next_pred))), round(mae, 2), round(rmse, 2), round(mape, 2)

def compare_models_and_predict(df):
    comparison_results = []
    final_predictions = []
    
    for (centre, item), group in df.groupby(["centre", "item"]):
        group_sorted = group.sort_values("date").reset_index(drop=True)
        
        arima_pred, arima_mae, arima_rmse, arima_mape = fit_predict_arima(group_sorted["demand"].values)
        prophet_pred, prophet_mae, prophet_rmse, prophet_mape = fit_predict_prophet(group_sorted)
        xgb_pred, xgb_mae, xgb_rmse, xgb_mape = fit_predict_xgboost(group_sorted)
        
        comparison_results.extend([
            {"centre": centre, "item": item, "model": "ARIMA", "mae": arima_mae, "rmse": arima_rmse, "mape": arima_mape, "forecast": arima_pred},
            {"centre": centre, "item": item, "model": "Prophet", "mae": prophet_mae, "rmse": prophet_rmse, "mape": prophet_mape, "forecast": prophet_pred},
            {"centre": centre, "item": item, "model": "XGBoost", "mae": xgb_mae, "rmse": xgb_rmse, "mape": xgb_mape, "forecast": xgb_pred}
        ])
        
        model_scores = {
            "ARIMA": (arima_mae, arima_rmse, arima_mape, arima_pred),
            "Prophet": (prophet_mae, prophet_rmse, prophet_mape, prophet_pred),
            "XGBoost": (xgb_mae, xgb_rmse, xgb_mape, xgb_pred)
        }
        
        best_model_name = min(model_scores.keys(), key=lambda k: (model_scores[k][0], model_scores[k][1]))
        best_mae, best_rmse, best_mape, chosen_forecast = model_scores[best_model_name]
        
        latest_row = group_sorted.iloc[-1]
        
        final_predictions.append({
            "centre": centre,
            "item": item,
            "predicted_demand": chosen_forecast,
            "current_stock": latest_row.get("stock", 40),
            "criticality": latest_row.get("criticality", 4),
            "lead_time": latest_row.get("lead_time", 5),
            "selected_model": best_model_name,
            "mae": best_mae,
            "rmse": best_rmse,
            "mape": best_mape
        })
        
    comp_df = pd.DataFrame(comparison_results)
    pred_df = pd.DataFrame(final_predictions)
    
    risk_df = calculate_risk_metrics(pred_df)
    return comp_df, risk_df

def calculate_risk_metrics(df_pred):
    df = df_pred.copy()
    
    df["expected_shortage"] = np.maximum(0, df["predicted_demand"] - df["current_stock"])
    df["shortage_ratio"] = df["expected_shortage"] / (df["predicted_demand"] + 1e-5)
    df["crit_norm"] = df["criticality"] / 5.0
    df["lead_norm"] = np.minimum(1.0, df["lead_time"] / 14.0)
    
    df["risk_score"] = np.where(
        df["expected_shortage"] == 0,
        df["crit_norm"] * 0.1,
        np.clip((df["shortage_ratio"] * 0.5) + (df["crit_norm"] * 0.3) + (df["lead_norm"] * 0.2), 0.0, 1.0)
    )
    df["risk_score"] = df["risk_score"].round(2)
    
    def assign_badge(score):
        if score >= 0.65:
            return "High"
        elif score >= 0.40:
            return "Medium"
        else:
            return "Low"
            
    df["status"] = df["risk_score"].apply(assign_badge)
    df["rank"] = df.groupby("centre")["risk_score"].rank(ascending=False, method="min").astype(int)
    
    return df.sort_values(by=["centre", "risk_score"], ascending=[True, False]).reset_index(drop=True)
