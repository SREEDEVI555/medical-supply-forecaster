import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from generate_dataset import generate_medical_dataset
from model import load_data, compare_models_and_predict
from allocation import calculate_allocations, get_default_partner_inventory

def test_pipeline():
    print("--- STEP 1: Generating & Loading Dataset ---")
    df_raw = generate_medical_dataset("medical_supply_data.csv")
    df_loaded = load_data("medical_supply_data.csv")
    print(f"Loaded CSV successfully: {len(df_loaded)} rows.")
    
    print("\n--- STEP 3: Multi-Model Evaluation (ARIMA vs Prophet vs XGBoost) ---")
    df_comp, df_risk = compare_models_and_predict(df_loaded)
    
    avg_mae = df_comp.groupby("model")[["mae", "rmse"]].mean()
    print("Average Error Metrics across all 25 Time Series (Lower is Better):")
    print(avg_mae)
    
    print("\nModel Selection Wins:")
    wins = df_risk["selected_model"].value_counts()
    for model_name, count in wins.items():
        print(f"  🏆 {model_name:<10}: {count} time series won")
        
    print("\n--- STEP 4 - 6: Best-Model Predictions & Risk Ranking (Centre A Example) ---")
    centre_a = df_risk[df_risk["centre"] == "Centre A"].sort_values("risk_score", ascending=False)
    for idx, r in centre_a.reset_index(drop=True).iterrows():
        print(f"  {idx+1}. {r['item']:<18} [Winner: {r['selected_model']:<7} MAE: {r['mae']:<4.1f}] Risk: {r['risk_score']:.2f}  Pred: {r['predicted_demand']:<4} Stock: {r['current_stock']:<4} Shortage: {r['expected_shortage']:<4} {r['status']}")
        
    print("\n--- STEP 7: Partner Hospital Allocation Engine ---")
    central_inv = get_default_partner_inventory()
    alloc_df, remaining_pool = calculate_allocations(df_risk, central_inv)
    
    print("Recommended Directives (First 5):")
    for _, r in alloc_df.head(5).iterrows():
        print(f"  {r['from']} → {r['to']} : {r['allocated_quantity']} units of {r['item']} ({r['risk_status']} Risk)")
        
    print("\n✅ MULTI-MODEL COMPARISON PIPELINE TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
