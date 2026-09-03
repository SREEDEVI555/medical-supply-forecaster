import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from generate_dataset import generate_medical_dataset
from model import load_data, train_and_predict_next_month
from allocation import calculate_allocations, get_default_partner_inventory

def test_pipeline():
    print("--- STEP 1: Generating & Loading Dataset ---")
    df_raw = generate_medical_dataset("medical_supply_data.csv")
    df_loaded = load_data("medical_supply_data.csv")
    print(f"Loaded CSV successfully: {len(df_loaded)} rows.")
    print("Sample rows:")
    print(df_loaded.head(3))
    
    print("\n--- STEP 3 - 6: Training Model, Forecasting Demand & Risk Ranking ---")
    df_risk = train_and_predict_next_month(df_loaded)
    print(f"Predicted next month demand & risk score for {len(df_risk)} centre-item pairs.")
    
    print("\nExample Risk Ranking Output (Centre A):")
    centre_a = df_risk[df_risk["centre"] == "Centre A"].sort_values("risk_score", ascending=False)
    for idx, r in centre_a.reset_index(drop=True).iterrows():
        print(f"  {idx+1}. {r['item']:<18} Risk: {r['risk_score']:.2f}  Demand: {r['predicted_demand']:<4} Stock: {r['current_stock']:<4} Shortage: {r['expected_shortage']:<4} {r['status']}")
        
    print("\n--- STEP 7: Partner Hospital Allocation Algorithm ---")
    central_inv = get_default_partner_inventory()
    print("Partner Hospital Central Pool Stock:")
    for item, qty in central_inv.items():
        print(f"  {item:<18}: {qty}")
        
    alloc_df, remaining_pool = calculate_allocations(df_risk, central_inv)
    print("\nRecommended Directives:")
    for _, r in alloc_df.iterrows():
        print(f"  {r['from']} → {r['to']} : {r['allocated_quantity']} units of {r['item']} ({r['risk_status']} Risk)")
        
    print("\nRemaining Central Pool:")
    for item, qty in remaining_pool.items():
        print(f"  {item:<18}: {qty}")
        
    print("\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
