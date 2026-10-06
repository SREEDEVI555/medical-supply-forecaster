import csv
import math

def load_csv_data(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

import os

def evaluate_patient_referral(case_dict, clinics, hospitals, accessibility):
    """
    Implements the Activity Diagram Patient Referral & Local Capability Decision Tree:
    Patient / Treatment Requirement
        ↓
    Local Treatment Capability Check
        ↓
    Available? --[YES]--> Local Care
        |
      [NO]
        ↓
    Hospital Capability Matching + Distance & Accessibility Scoring
        ↓
    Referral Recommendation
    """
    clinic_id = case_dict.get("Clinic_ID", "C001")
    req_service = case_dict.get("Required_Service", "Basic Care")
    urgency = case_dict.get("Urgency_Level", "Medium")
    
    clinic_info = next((c for c in clinics if c["Clinic_ID"] == clinic_id), clinics[0])
    
    # 1. Local Capability Check
    local_available = False
    if req_service == "Basic Care":
        local_available = True
    elif req_service == "Minor Wound Care" and clinic_info.get("Minor_Wound_Care") == "Yes":
        local_available = True
    elif req_service == "IV Fluid Care" and clinic_info.get("IV_Fluid_Care") == "Yes":
        local_available = True
    elif req_service == "Basic Diagnostics" and clinic_info.get("Basic_Diagnostics") == "Yes":
        local_available = True
        
    if local_available:
        return {
            "Decision": "Local Care",
            "Recommendation": f"Treat locally at {clinic_info['Clinic_Name']}",
            "Referral_Required": "No",
            "Recommended_Hospital": "None (Handled Locally)",
            "Score": 1.0,
            "Details": f"Clinic {clinic_id} possesses required capability ({req_service})."
        }
        
    # 2. Local Capability NOT available -> Hospital Capability Matching & Route Scoring
    candidate_scores = []
    
    for hosp in hospitals:
        hid = hosp["Hospital_ID"]
        hname = hosp["Hospital_Name"]
        avail_beds = int(hosp.get("Available_Beds", 10))
        total_beds = int(hosp.get("Bed_Capacity", 100))
        bed_ratio = min(1.0, avail_beds / max(1, total_beds * 0.2)) # normalized
        
        # Capability Match Score
        cap_match = 0.5
        if req_service == "Surgery" and hosp.get("Surgery_Available") == "Yes":
            cap_match = 1.0
        elif req_service == "ICU Care" and hosp.get("ICU_Available") == "Yes":
            cap_match = 1.0
        elif req_service == "Emergency Care" and hosp.get("Emergency_Care") == "Yes":
            cap_match = 1.0
        elif req_service == "Advanced Diagnostics" and "Advanced" in hosp.get("Diagnostic_Facilities", ""):
            cap_match = 1.0
            
        # Distance & Accessibility from Route matrix
        route = next((r for r in accessibility if r["Source_ID"] == clinic_id and r["Destination_ID"] == hid), None)
        if route:
            dist = float(route.get("Distance_KM", 100.0))
            travel_min = float(route.get("Travel_Time_Minutes", 120.0))
            road_cond = route.get("Road_Condition", "Moderate")
        else:
            dist = 120.0
            travel_min = 150.0
            road_cond = "Moderate"
            
        distance_score = max(0.0, 1.0 - (dist / 200.0))
        travel_score = max(0.0, 1.0 - (travel_min / 240.0))
        road_score = 1.0 if road_cond == "Good" else (0.6 if road_cond == "Moderate" else 0.3)
        
        # Final Hospital Referral Suitability Score
        referral_score = round(
            (cap_match * 0.40) + (bed_ratio * 0.25) + (distance_score * 0.20) + (travel_score * 0.15), 2
        )
        
        candidate_scores.append({
            "Hospital_ID": hid,
            "Hospital_Name": hname,
            "Referral_Score": referral_score,
            "Distance_KM": dist,
            "Travel_Time_Min": travel_min,
            "Available_Beds": avail_beds,
            "Road_Condition": road_cond,
            "Capability_Match": "Full Match" if cap_match == 1.0 else "Partial Match"
        })
        
    candidate_scores.sort(key=lambda x: x["Referral_Score"], reverse=True)
    best_hosp = candidate_scores[0]
    
    return {
        "Decision": "Hospital Referral Required",
        "Recommendation": f"Refer patient to {best_hosp['Hospital_Name']} ({best_hosp['Distance_KM']} km, ~{best_hosp['Travel_Time_Min']} mins)",
        "Referral_Required": "Yes",
        "Recommended_Hospital": best_hosp["Hospital_Name"],
        "Score": best_hosp["Referral_Score"],
        "Ranked_Candidates": candidate_scores,
        "Details": f"Local clinic lacks {req_service}. Top hospital matched based on capability, bed availability ({best_hosp['Available_Beds']} beds), and transit time."
    }
