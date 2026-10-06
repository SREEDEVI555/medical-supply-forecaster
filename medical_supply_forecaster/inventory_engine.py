import pandas as pd
import numpy as np

def load_relational_datasets(data_dir="."):
    """Loads all 7 relational datasets matching PDF schema requirements."""
    df_demand = pd.read_csv(f"{data_dir}/medical_demand.csv")
    df_inventory = pd.read_csv(f"{data_dir}/inventory.csv")
    df_clinics = pd.read_csv(f"{data_dir}/clinics.csv")
    df_hospitals = pd.read_csv(f"{data_dir}/hospitals.csv")
    df_referrals = pd.read_csv(f"{data_dir}/referral_cases.csv")
    df_access = pd.read_csv(f"{data_dir}/accessibility.csv")
    df_medicines = pd.read_csv(f"{data_dir}/medicines.csv")
    
    return df_demand, df_inventory, df_clinics, df_hospitals, df_referrals, df_access, df_medicines

def compute_inventory_metrics(df_demand, df_inventory, df_access, df_medicines):
    """
    Computes Safety Stock (SS), Reorder Point (ROP), Economic Order Quantity (EOQ),
    Stockout Risk, Accessibility Risk, and Replenishment Priority Score.
    """
    # 1. Historical demand statistics per clinic and medicine
    demand_stats = df_demand.groupby(["Clinic_ID", "Medicine_ID"])["Quantity_Consumed"].agg(
        avg_monthly_demand="mean",
        std_monthly_demand="std"
    ).reset_index()
    
    # Merge with inventory & master tables
    inv_df = pd.merge(df_inventory, demand_stats, on=["Clinic_ID", "Medicine_ID"], how="left")
    inv_df["avg_monthly_demand"] = inv_df["avg_monthly_demand"].fillna(20.0)
    inv_df["std_monthly_demand"] = inv_df["std_monthly_demand"].fillna(5.0)
    
    inv_df = pd.merge(inv_df, df_medicines[["Medicine_ID", "Medicine_Name", "Category"]], on="Medicine_ID", how="left")
    
    # Service level Z-factor based on criticality
    crit_z = {"Critical": 2.33, "High": 1.65, "Medium": 1.28, "Low": 1.00}
    inv_df["Z_factor"] = inv_df["Criticality_Level"].map(crit_z).fillna(1.28)
    
    # Lead time in months
    inv_df["lead_time_months"] = inv_df["Lead_Time_Days"] / 30.0
    
    # 2. Safety Stock (SS) Formula: Z * std_d * sqrt(L_months)
    inv_df["Safety_Stock"] = (
        inv_df["Z_factor"] * inv_df["std_monthly_demand"] * np.sqrt(inv_df["lead_time_months"])
    ).fillna(5.0).round(0).astype(int)
    
    # 3. Reorder Point (ROP) Formula: (avg_demand * lead_time_months) + Safety_Stock
    inv_df["Reorder_Point"] = (
        (inv_df["avg_monthly_demand"] * inv_df["lead_time_months"]) + inv_df["Safety_Stock"]
    ).fillna(10.0).round(0).astype(int)
    
    # 4. Economic Order Quantity (EOQ) Formula: sqrt((2 * Annual_Demand * Ordering_Cost) / Holding_Cost)
    ordering_cost = 50.0  # Fixed ordering setup cost
    inv_df["annual_demand"] = inv_df["avg_monthly_demand"] * 12.0
    inv_df["unit_holding_cost"] = np.maximum(0.5, inv_df["Unit_Cost"] * 0.20)
    
    inv_df["EOQ"] = np.sqrt(
        (2 * inv_df["annual_demand"] * ordering_cost) / inv_df["unit_holding_cost"]
    ).fillna(50.0).round(0).astype(int)

    
    # 5. Stockout Risk
    inv_df["Stockout_Risk"] = np.maximum(0.0, (inv_df["Reorder_Point"] - inv_df["Current_Stock"]) / (inv_df["Reorder_Point"] + 1e-5))
    inv_df["Stockout_Risk"] = np.clip(inv_df["Stockout_Risk"], 0.0, 1.0).round(2)
    
    # 6. Accessibility Risk calculation from accessibility.csv
    access_stats = df_access.groupby("Source_ID").agg(
        avg_distance_km=("Distance_KM", "mean"),
        avg_travel_time_min=("Travel_Time_Minutes", "mean"),
        road_condition=("Road_Condition", lambda x: "Poor" if "Poor" in x.values else "Good")
    ).reset_index().rename(columns={"Source_ID": "Clinic_ID"})
    
    road_weights = {"Poor": 1.0, "Moderate": 0.5, "Good": 0.2}
    access_stats["road_penalty"] = access_stats["road_condition"].map(road_weights)
    access_stats["Accessibility_Risk"] = np.clip(
        (access_stats["avg_distance_km"] / 150.0 * 0.4) +
        (access_stats["avg_travel_time_min"] / 180.0 * 0.3) +
        (access_stats["road_penalty"] * 0.3), 0.0, 1.0
    ).round(2)
    
    inv_df = pd.merge(inv_df, access_stats[["Clinic_ID", "Accessibility_Risk"]], on="Clinic_ID", how="left")
    inv_df["Accessibility_Risk"] = inv_df["Accessibility_Risk"].fillna(0.3)
    
    # 7. Multi-Factor Replenishment Priority Score
    crit_weight = {"Critical": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.2}
    inv_df["crit_w"] = inv_df["Criticality_Level"].map(crit_weight).fillna(0.5)
    inv_df["lead_w"] = np.minimum(1.0, inv_df["Lead_Time_Days"] / 14.0)
    
    inv_df["Replenishment_Priority_Score"] = (
        (inv_df["Stockout_Risk"] * 0.40) +
        (inv_df["crit_w"] * 0.30) +
        (inv_df["lead_w"] * 0.15) +
        (inv_df["Accessibility_Risk"] * 0.15)
    ).round(2)
    
    def assign_priority_badge(score):
        if score >= 0.65:
            return "🔴 URGENT REPLENISHMENT"
        elif score >= 0.45:
            return "🟡 HIGH PRIORITY"
        else:
            return "🟢 NORMAL"
            
    inv_df["Priority_Badge"] = inv_df["Replenishment_Priority_Score"].apply(assign_priority_badge)
    
    return inv_df.sort_values("Replenishment_Priority_Score", ascending=False).reset_index(drop=True)
