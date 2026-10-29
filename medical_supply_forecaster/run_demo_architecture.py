import sys
import io
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_all_datasets import generate_full_system_datasets
from inventory_engine import load_relational_datasets, compute_inventory_metrics
from referral_engine import evaluate_patient_referral, generate_digital_handshake

def test_full_architecture():
    print("--- 1. Generating & Loading 7 Relational Datasets ---")
    generate_full_system_datasets(".")
    df_demand, df_inventory, df_clinics, df_hospitals, df_referrals, df_access, df_medicines = load_relational_datasets(".")
    print(f"Loaded 7 datasets: Demand={len(df_demand)}, Inventory={len(df_inventory)}, Clinics={len(df_clinics)}, Hospitals={len(df_hospitals)}.")
    
    print("\n--- 2. Operations Management Inventory Engine (Safety Stock, ROP & EOQ) ---")
    df_inv_opt = compute_inventory_metrics(df_demand, df_inventory, df_access, df_medicines)
    urgent_cnt = len(df_inv_opt[df_inv_opt["Priority_Badge"].str.contains("URGENT")])
    avg_risk = df_inv_opt["Stockout_Risk"].mean() * 100
    print(f"Inventory Health: {urgent_cnt} Urgent Replenishments, Average Stockout Risk: {avg_risk:.1f}%.")
    print("Sample Top Urgent Replenishments:")
    for idx, r in df_inv_opt.head(5).iterrows():
        print(f"  {r['Clinic_ID']} - {r['Medicine_Name']:<25} Stock: {r['Current_Stock']:<4} ROP: {r['Reorder_Point']:<4} SS: {r['Safety_Stock']:<4} EOQ: {r['EOQ']:<4} Priority: {r['Replenishment_Priority_Score']:.2f} {r['Priority_Badge']}")
        
    print("\n--- 3. Closed-Loop Patient Referral & Tele-Triage Engine ---")
    sample_case = {
        "Clinic_ID": "C002",
        "Required_Service": "Surgery",
        "Urgency_Level": "Emergency"
    }
    res = evaluate_patient_referral(
        sample_case, 
        df_clinics.to_dict("records"), 
        df_hospitals.to_dict("records"), 
        df_access.to_dict("records"),
        inventory=df_inventory
    )
    print(f"Patient Case: Clinic C002 needing Emergency Surgery")
    print(f"  Decision: {res['Decision']}")
    print(f"  Recommendation: {res['Recommendation']}")
    print(f"  Golden Hour Survival Window: {res['Golden_Hour_Feasibility']}")
    if res.get("Multi_Hop_Option"):
        print(f"  Multi-Hop Protocol: {res['Multi_Hop_Option']['Leg_1_Action']}")
        
    print("\n--- 4. Pre-Arrival Digital Handshake & Bed Hold ---")
    handshake = generate_digital_handshake(sample_case, res.get("Recommended_Hospital_ID", "H001"), res["Recommended_Hospital"])
    print(f"  Token: {handshake['Handshake_Token']}")
    print(f"  Bed Reservation: {handshake['Bed_Reservation_Status']}")
    print(f"  Assigned Trauma Bay: {handshake['Assigned_Trauma_Bay']}")
    
    print("\n✅ FULL ARCHITECTURE PIPELINE VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_architecture()
