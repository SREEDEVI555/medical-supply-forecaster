import pandas as pd
import numpy as np

# Try importing XGBoost; fallback to RandomForestRegressor if missing
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    HAS_XGBOOST = False

def load_data(filepath="medical_supply_data.csv"):
    """Step 1: Load data using Pandas"""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df

def prepare_features(df):
    """Create lag features and temporal encodings for forecasting."""
    df_sorted = df.sort_values(by=["centre", "item", "date"]).copy()
    
    # Extract month numerical feature
    df_sorted["month_num"] = df_sorted["date"].dt.month
    
    # Lag 1 and Lag 2 features
    df_sorted["demand_lag_1"] = df_sorted.groupby(["centre", "item"])["demand"].shift(1)
    df_sorted["demand_lag_2"] = df_sorted.groupby(["centre", "item"])["demand"].shift(2)
    
    # 3-month rolling mean demand feature
    df_sorted["demand_roll_mean"] = (
        df_sorted.groupby(["centre", "item"])["demand"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    
    return df_sorted

def train_and_predict_next_month(df):
    """
    Step 3 & Step 4:
    Train forecasting model on historical 12 months data
    and predict next month's demand (2027-01 / Month 13).
    """
    df_feat = prepare_features(df)
    
    # Drop initial rows with NaNs from lag creation
    train_data = df_feat.dropna(subset=["demand_lag_1", "demand_lag_2"]).copy()
    
    # Categorical encodings
    centre_dummies = pd.get_dummies(train_data["centre"], prefix="centre")
    item_dummies = pd.get_dummies(train_data["item"], prefix="item")
    
    feature_cols = ["month_num", "demand_lag_1", "demand_lag_2", "demand_roll_mean", "criticality", "lead_time"]
    X = pd.concat([train_data[feature_cols], centre_dummies, item_dummies], axis=1)
    y = train_data["demand"]
    
    # Initialize and train model
    if HAS_XGBOOST:
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
    model.fit(X, y)
    
    # Prepare test row for next month (Month 1 = Jan 2027) for each (centre, item) pair
    latest_month_df = df_feat[df_feat["date"] == df_feat["date"].max()].copy()
    
    predictions = []
    
    for (centre, item), group in latest_month_df.groupby(["centre", "item"]):
        # Latest demand values for lag construction
        hist = df_feat[(df_feat["centre"] == centre) & (df_feat["item"] == item)].sort_values("date")
        
        last_demand = hist["demand"].iloc[-1]
        prev_demand = hist["demand"].iloc[-2] if len(hist) > 1 else last_demand
        roll_mean = hist["demand"].iloc[-3:].mean()
        
        latest_row = group.iloc[0]
        
        pred_dict = {
            "month_num": 1, # Next month is January (1)
            "demand_lag_1": last_demand,
            "demand_lag_2": prev_demand,
            "demand_roll_mean": roll_mean,
            "criticality": latest_row["criticality"],
            "lead_time": latest_row["lead_time"]
        }
        
        # Add centre & item one-hot fields matching X columns
        row_df = pd.DataFrame([pred_dict])
        for col in X.columns:
            if col not in row_df.columns:
                if col.startswith("centre_"):
                    row_df[col] = 1 if col == f"centre_{centre}" else 0
                elif col.startswith("item_"):
                    row_df[col] = 1 if col == f"item_{item}" else 0
                else:
                    row_df[col] = 0
                    
        # Reorder columns to match training set
        row_df = row_df[X.columns]
        
        pred_demand = max(5, int(round(model.predict(row_df)[0])))
        
        predictions.append({
            "centre": centre,
            "item": item,
            "predicted_demand": pred_demand,
            "current_stock": latest_row["stock"],
            "criticality": latest_row["criticality"],
            "lead_time": latest_row["lead_time"]
        })
        
    pred_df = pd.DataFrame(predictions)
    return calculate_risk_metrics(pred_df)

def calculate_risk_metrics(df_pred):
    """
    Step 5:
    Calculation Chain:
    Predicted Demand
        ↓
    Current Stock
        ↓
    Expected Shortage = max(0, Predicted Demand - Current Stock)
        ↓
    Criticality (Scale 1 to 5)
        ↓
    Lead Time (Days)
        ↓
    Risk Score (0.0 to 1.0)
    """
    df = df_pred.copy()
    
    # 1. Expected Shortage
    df["expected_shortage"] = np.maximum(0, df["predicted_demand"] - df["current_stock"])
    
    # 2. Shortage Severity Ratio relative to predicted demand
    df["shortage_ratio"] = df["expected_shortage"] / (df["predicted_demand"] + 1e-5)
    
    # 3. Normalized Criticality (1-5 mapped to 0.2-1.0)
    df["crit_norm"] = df["criticality"] / 5.0
    
    # 4. Lead Time Risk Factor (Normalized up to 14 days)
    df["lead_norm"] = np.minimum(1.0, df["lead_time"] / 14.0)
    
    # 5. Composite Risk Score
    # Risk is 0 if no shortage exists, otherwise combines shortage ratio, criticality, and lead time
    df["risk_score"] = np.where(
        df["expected_shortage"] == 0,
        df["crit_norm"] * 0.1,  # Low background risk
        np.clip(
            (df["shortage_ratio"] * 0.5) + (df["crit_norm"] * 0.3) + (df["lead_norm"] * 0.2),
            0.0, 1.0
        )
    )
    
    df["risk_score"] = df["risk_score"].round(2)
    
    # Step 6: Assign Risk Category Indicator Badges
    def assign_badge(score):
        if score >= 0.65:
            return "🔴 High"
        elif score >= 0.40:
            return "🟡 Medium"
        else:
            return "🟢 Low"
            
    df["status"] = df["risk_score"].apply(assign_badge)
    
    # Rank items within each centre by Risk Score descending
    df["rank"] = df.groupby("centre")["risk_score"].rank(ascending=False, method="min").astype(int)
    
    df_sorted = df.sort_values(by=["centre", "risk_score"], ascending=[True, False]).reset_index(drop=True)
    return df_sorted
