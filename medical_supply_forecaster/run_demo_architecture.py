import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from generate_all_datasets import generate_full_system_datasets
from model import load_data, compare_models_and_predict
from inventory_engine import load_relational_datasets, compute_inventory_metrics
from referral_engine import evaluate_patient_referral

def test_full_architecture():
    print("--- 1. Generating & Loading 7 Relational Datasets ---")
    generate_full_system_datasets(".")
    df_demand, df_inventory, df_clinics, df_hospitals, df_referrals, df_access, df_medicines = load_relational_datasets(".")
    print(f"Loaded 7 datasets: Demand={len(df_demand)}, Inventory={len(df_inventory)}, Clinics={len(df_clinics)}, Hospitals={len(df_hospitals)}.")
    
    print("\n--- 2. Operations Management Inventory Engine (Safety Stock, ROP & EOQ) ---")
    df_inv_opt = compute_inventory_metrics(df_demand, df_inventory, df_access, df_medicines)
    print("Sample Top Urgent Replenishments:")
    for idx, r in df_inv_opt.head(5).iterrows():
        print(f"  {r['Clinic_ID']} - {r['Medicine_Name']:<25} Stock: {r['Current_Stock']:<4} ROP: {r['Reorder_Point']:<4} SS: {r['Safety_Stock']:<4} EOQ: {r['EOQ']:<4} Priority: {r['Replenishment_Priority_Score']:.2f} {r['Priority_Badge']}")
        
    print("\n--- 3. Patient Referral & Tele-Triage Engine ---")
    sample_case = {
        "Clinic_ID": "C002",
        "Required_Service": "Surgery",
        "Urgency_Level": "Emergency"
    }
    res = evaluate_patient_referral(sample_case, df_clinics.to_dict("records"), df_hospitals.to_dict("records"), df_access.to_dict("records"))
    print(f"Patient Case: Clinic C002 needing Emergency Surgery")
    print(f"  Decision: {res['Decision']}")
    print(f"  Recommendation: {res['Recommendation']}")
    print(f"  Score: {res['Score']}")
    
    print("\n✅ FULL ARCHITECTURE PIPELINE VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_architecture()
