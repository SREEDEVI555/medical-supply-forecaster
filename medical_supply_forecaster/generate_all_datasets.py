import csv
import os
import random
import math
from datetime import datetime, timedelta

def generate_full_system_datasets(output_dir="."):
    random.seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. MEDICINES MASTER DATASET (medicines.csv)
    # -------------------------------------------------------------
    medicines = [
        {"Medicine_ID": "M001", "Medicine_Name": "ORS (Oral Rehydration)", "Category": "Rehydration", "Unit": "Sachet", "Criticality_Level": "High", "Shelf_Life_Days": 730, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M002", "Medicine_Name": "Amoxicillin Antibiotic", "Category": "Antibiotic", "Unit": "Capsule", "Criticality_Level": "High", "Shelf_Life_Days": 540, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M003", "Medicine_Name": "IV Normal Saline 500ml", "Category": "IV Fluid", "Unit": "Bottle", "Criticality_Level": "Critical", "Shelf_Life_Days": 365, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M004", "Medicine_Name": "Sterile Gauze Bandage", "Category": "Wound Care", "Unit": "Pack", "Criticality_Level": "Medium", "Shelf_Life_Days": 1095, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M005", "Medicine_Name": "Paracetamol 500mg", "Category": "Analgesic", "Unit": "Tablet", "Criticality_Level": "Low", "Shelf_Life_Days": 730, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M006", "Medicine_Name": "Pulse Oximeter Sensor", "Category": "Diagnostic Device", "Unit": "Piece", "Criticality_Level": "High", "Shelf_Life_Days": 1825, "Storage_Type": "Normal", "Essential_Medicine": "Yes"},
        {"Medicine_ID": "M007", "Medicine_Name": "Injectable Antivenom", "Category": "Antidote", "Unit": "Vial", "Criticality_Level": "Critical", "Shelf_Life_Days": 365, "Storage_Type": "Refrigerated", "Essential_Medicine": "Yes"}
    ]
    with open(os.path.join(output_dir, "medicines.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(medicines[0].keys()))
        writer.writeheader()
        writer.writerows(medicines)

    # -------------------------------------------------------------
    # 2. REMOTE CLINICS DATASET (clinics.csv)
    # -------------------------------------------------------------
    clinics = [
        {
            "Clinic_ID": "C001", "Clinic_Name": "Rameswaram Island Sub-Centre", "Location": "Rameswaram Coastal Zone",
            "Latitude": 9.2876, "Longitude": 79.3129, "Population_Served": 12500, "Average_Daily_Patients": 45,
            "Staff_Count": 4, "Doctor_Available": "Yes", "Nurse_Available": "Yes", "Basic_Diagnostics": "Yes",
            "Emergency_Care": "No", "IV_Fluid_Care": "Yes", "Minor_Wound_Care": "Yes", "Pharmacy_Available": "Yes",
            "Storage_Capacity": 500, "Accessibility_Level": "Moderate"
        },
        {
            "Clinic_ID": "C002", "Clinic_Name": "Kurusadai Marine Outpost Clinic", "Location": "Gulf of Mannar Island",
            "Latitude": 9.2450, "Longitude": 79.2083, "Population_Served": 6200, "Average_Daily_Patients": 22,
            "Staff_Count": 2, "Doctor_Available": "No", "Nurse_Available": "Yes", "Basic_Diagnostics": "No",
            "Emergency_Care": "No", "IV_Fluid_Care": "No", "Minor_Wound_Care": "Yes", "Pharmacy_Available": "Yes",
            "Storage_Capacity": 250, "Accessibility_Level": "Difficult"
        },
        {
            "Clinic_ID": "C003", "Clinic_Name": "Mandapam Primary Health Centre", "Location": "Mandapam Peninsula",
            "Latitude": 9.2778, "Longitude": 79.1247, "Population_Served": 18900, "Average_Daily_Patients": 85,
            "Staff_Count": 8, "Doctor_Available": "Yes", "Nurse_Available": "Yes", "Basic_Diagnostics": "Yes",
            "Emergency_Care": "Yes", "IV_Fluid_Care": "Yes", "Minor_Wound_Care": "Yes", "Pharmacy_Available": "Yes",
            "Storage_Capacity": 1200, "Accessibility_Level": "Easy"
        },
        {
            "Clinic_ID": "C004", "Clinic_Name": "Dhanushkodi Border Health Post", "Location": "Dhanushkodi Point",
            "Latitude": 9.1764, "Longitude": 79.4189, "Population_Served": 4100, "Average_Daily_Patients": 18,
            "Staff_Count": 2, "Doctor_Available": "No", "Nurse_Available": "Yes", "Basic_Diagnostics": "No",
            "Emergency_Care": "No", "IV_Fluid_Care": "Yes", "Minor_Wound_Care": "Yes", "Pharmacy_Available": "Yes",
            "Storage_Capacity": 200, "Accessibility_Level": "Difficult"
        },
        {
            "Clinic_ID": "C005", "Clinic_Name": "Pamban Western Health Clinic", "Location": "Pamban West",
            "Latitude": 9.2811, "Longitude": 79.2155, "Population_Served": 9800, "Average_Daily_Patients": 38,
            "Staff_Count": 5, "Doctor_Available": "Yes", "Nurse_Available": "Yes", "Basic_Diagnostics": "Yes",
            "Emergency_Care": "No", "IV_Fluid_Care": "Yes", "Minor_Wound_Care": "Yes", "Pharmacy_Available": "Yes",
            "Storage_Capacity": 450, "Accessibility_Level": "Moderate"
        }
    ]
    with open(os.path.join(output_dir, "clinics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clinics[0].keys()))
        writer.writeheader()
        writer.writerows(clinics)

    # -------------------------------------------------------------
    # 3. HOSPITALS DATASET (hospitals.csv)
    # -------------------------------------------------------------
    hospitals = [
        {
            "Hospital_ID": "H001", "Hospital_Name": "Ramanathapuram District Headquarters Hospital", "Location": "Ramanathapuram City",
            "Latitude": 9.3716, "Longitude": 78.8311, "Emergency_Care": "Yes", "ICU_Available": "Yes",
            "Surgery_Available": "Yes", "Specialist_Available": "Yes", "Diagnostic_Facilities": "Advanced (CT, Ultrasound, X-Ray, Lab)",
            "Ambulance_Available": "Yes", "Bed_Capacity": 350, "Available_Beds": 42, "Operating_Hours": "24/7",
            "Specialties": "General Surgery, Trauma, Cardiology, Pediatrics, Infectious Disease", "Treatment_Capabilities": "ICU, Surgery, Antivenom, Emergency Care"
        },
        {
            "Hospital_ID": "H002", "Hospital_Name": "Madurai Medical College & Hospital Hub", "Location": "Madurai Metro Region",
            "Latitude": 9.9252, "Longitude": 78.1198, "Emergency_Care": "Yes", "ICU_Available": "Yes",
            "Surgery_Available": "Yes", "Specialist_Available": "Yes", "Diagnostic_Facilities": "Comprehensive (MRI, CT, Bio-Lab)",
            "Ambulance_Available": "Yes", "Bed_Capacity": 1200, "Available_Beds": 115, "Operating_Hours": "24/7",
            "Specialties": "Super Speciality, Trauma, Toxicology, Critical Care", "Treatment_Capabilities": "Comprehensive Critical Care, Emergency Surgery"
        },
        {
            "Hospital_ID": "H003", "Hospital_Name": "Paramakudi Community Sub-Hospital", "Location": "Paramakudi Town",
            "Latitude": 9.5441, "Longitude": 78.5882, "Emergency_Care": "Yes", "ICU_Available": "No",
            "Surgery_Available": "Yes", "Specialist_Available": "No", "Diagnostic_Facilities": "Basic (X-Ray, Blood Lab)",
            "Ambulance_Available": "Yes", "Bed_Capacity": 100, "Available_Beds": 18, "Operating_Hours": "24/7",
            "Specialties": "General Medicine, Basic Surgery, Obstetrics", "Treatment_Capabilities": "Basic Surgery, Inpatient Care"
        }
    ]
    with open(os.path.join(output_dir, "hospitals.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(hospitals[0].keys()))
        writer.writeheader()
        writer.writerows(hospitals)

    # -------------------------------------------------------------
    # 4. ACCESSIBILITY & TRANSPORTATION DATASET (accessibility.csv)
    # -------------------------------------------------------------
    accessibility = [
        {"Route_ID": "R001", "Source_ID": "C001", "Destination_ID": "H001", "Distance_KM": 55.4, "Travel_Time_Minutes": 75, "Road_Condition": "Good", "Accessibility_Level": "Easy", "Seasonal_Access": "All-season", "Transport_Available": "Yes", "Average_Delivery_Time_Days": 2},
        {"Route_ID": "R002", "Source_ID": "C001", "Destination_ID": "H002", "Distance_KM": 168.2, "Travel_Time_Minutes": 210, "Road_Condition": "Good", "Accessibility_Level": "Easy", "Seasonal_Access": "All-season", "Transport_Available": "Yes", "Average_Delivery_Time_Days": 3},
        {"Route_ID": "R003", "Source_ID": "C002", "Destination_ID": "H001", "Distance_KM": 72.8, "Travel_Time_Minutes": 140, "Road_Condition": "Poor", "Accessibility_Level": "Difficult", "Seasonal_Access": "Seasonal (Monsoon Boat Only)", "Transport_Available": "Yes", "Average_Delivery_Time_Days": 6},
        {"Route_ID": "R004", "Source_ID": "C003", "Destination_ID": "H001", "Distance_KM": 38.1, "Travel_Time_Minutes": 45, "Road_Condition": "Good", "Accessibility_Level": "Easy", "Seasonal_Access": "All-season", "Transport_Available": "Yes", "Average_Delivery_Time_Days": 1},
        {"Route_ID": "R005", "Source_ID": "C004", "Destination_ID": "H001", "Distance_KM": 78.5, "Travel_Time_Minutes": 155, "Road_Condition": "Poor", "Accessibility_Level": "Difficult", "Seasonal_Access": "Seasonal", "Transport_Available": "No", "Average_Delivery_Time_Days": 7},
        {"Route_ID": "R006", "Source_ID": "C005", "Destination_ID": "H001", "Distance_KM": 48.0, "Travel_Time_Minutes": 65, "Road_Condition": "Moderate", "Accessibility_Level": "Moderate", "Seasonal_Access": "All-season", "Transport_Available": "Yes", "Average_Delivery_Time_Days": 3}
    ]
    with open(os.path.join(output_dir, "accessibility.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(accessibility[0].keys()))
        writer.writeheader()
        writer.writerows(accessibility)

    # -------------------------------------------------------------
    # 5. INVENTORY DATASET (inventory.csv)
    # -------------------------------------------------------------
    inventory_rows = []
    inv_id_counter = 1
    clinic_ids = [c["Clinic_ID"] for c in clinics]
    medicine_ids = [m["Medicine_ID"] for m in medicines]
    
    for cid in clinic_ids:
        for mid in medicine_ids:
            med_info = next(m for m in medicines if m["Medicine_ID"] == mid)
            crit = med_info["Criticality_Level"]
            
            base_capacity = 1000 if cid in ["C001", "C003"] else 400
            current_stock = int(base_capacity * random.uniform(0.15, 0.75))
            min_stock = int(base_capacity * 0.25)
            max_capacity = base_capacity
            
            lead_time = random.choice([2, 3, 5, 7, 10]) if cid != "C004" else random.choice([7, 10, 14])
            unit_cost = round(random.uniform(5.0, 150.0), 2)
            
            inventory_rows.append({
                "Inventory_ID": f"INV{inv_id_counter:03d}",
                "Clinic_ID": cid,
                "Medicine_ID": mid,
                "Current_Stock": current_stock,
                "Minimum_Stock_Level": min_stock,
                "Maximum_Stock_Capacity": max_capacity,
                "Unit_Cost": unit_cost,
                "Lead_Time_Days": lead_time,
                "Minimum_Order_Quantity": 50,
                "Expiry_Date": (datetime.now() + timedelta(days=random.randint(60, 500))).strftime("%Y-%m-%d"),
                "Batch_ID": f"BAT-{random.randint(1000, 9999)}",
                "Last_Restock_Date": (datetime.now() - timedelta(days=random.randint(10, 60))).strftime("%Y-%m-%d"),
                "Supplier_ID": f"SUP-{random.randint(1, 4):02d}",
                "Criticality_Level": crit
            })
            inv_id_counter += 1
            
    with open(os.path.join(output_dir, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inventory_rows)

    # -------------------------------------------------------------
    # 6. MEDICAL DEMAND DATASET (medical_demand.csv) -> 210 Rows (150+ Requirement)
    # -------------------------------------------------------------
    demand_rows = []
    dates = [f"2026-{m:02d}" for m in range(1, 13)]
    core_meds = [m for m in medicines if m["Medicine_ID"] in ["M001", "M002", "M003", "M004", "M005"]]
    
    for date_str in dates:
        month_num = int(date_str.split("-")[1])
        season = "Summer" if month_num in [4, 5, 6] else ("Monsoon" if month_num in [7, 8, 9, 10] else "Winter")
        
        for c in clinics:
            cid = c["Clinic_ID"]
            c_scale = c["Population_Served"] / 10000.0
            
            for m in core_meds:
                mid = m["Medicine_ID"]
                mname = m["Medicine_Name"]
                mcat = m["Category"]
                
                base_demand = 80 * c_scale
                if mcat == "Rehydration" and season == "Summer":
                    base_demand *= 1.4
                elif mcat == "Antibiotic" and season == "Monsoon":
                    base_demand *= 1.3
                elif mcat == "IV Fluid" and season in ["Summer", "Monsoon"]:
                    base_demand *= 1.35
                    
                noise = random.gauss(0, base_demand * 0.1)
                qty_consumed = max(10, int(base_demand + noise))
                patient_count = int(qty_consumed * random.uniform(2.0, 3.2))
                
                condition_cat = "Gastroenteritis / Dehydration" if mcat == "Rehydration" else (
                    "Respiratory / Bacterial Infection" if mcat == "Antibiotic" else (
                        "Acute Fluid Loss / Shock" if mcat == "IV Fluid" else "General Trauma / Wound"
                    )
                )
                
                prev_stockout = 1 if (cid in ["C002", "C004"] and month_num in [6, 7, 8]) else 0
                stockout_days = random.randint(2, 8) if prev_stockout == 1 else 0
                
                demand_rows.append({
                    "Date": date_str,
                    "Clinic_ID": cid,
                    "Medicine_ID": mid,
                    "Medicine_Name": mname,
                    "Medicine_Category": mcat,
                    "Quantity_Consumed": qty_consumed,
                    "Patient_Count": patient_count,
                    "Condition_Category": condition_cat,
                    "Season": season,
                    "Day_of_Week": "Monthly_Aggregate",
                    "Month": month_num,
                    "Previous_Stockout": prev_stockout,
                    "Stockout_Days": stockout_days
                })
                
    with open(os.path.join(output_dir, "medical_demand.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(demand_rows[0].keys()))
        writer.writeheader()
        writer.writerows(demand_rows)

    # -------------------------------------------------------------
    # 7. PATIENT / REFERRAL CASE DATASET (referral_cases.csv)
    # -------------------------------------------------------------
    referral_rows = []
    case_counter = 1
    services = ["Surgery", "ICU Care", "Advanced Diagnostics", "Basic Care", "Antivenom Administration"]
    urgencies = ["Low", "Medium", "High", "Emergency"]
    age_groups = ["Child", "Adult", "Elderly"]
    
    for _ in range(50):
        cid = random.choice(clinic_ids)
        clinic_info = next(c for c in clinics if c["Clinic_ID"] == cid)
        
        urgency = random.choice(urgencies)
        req_service = random.choice(services)
        age = random.choice(age_groups)
        
        if req_service == "Basic Care" or (req_service == "IV Care" and clinic_info["IV_Fluid_Care"] == "Yes"):
            local_cap = "Yes"
            ref_req = "No"
            recommended_hosp = "None (Local Care)"
        else:
            local_cap = "No"
            ref_req = "Yes"
            recommended_hosp = "H001" if urgency != "Emergency" else "H002"
            
        referral_rows.append({
            "Case_ID": f"R{case_counter:03d}",
            "Clinic_ID": cid,
            "Age_Group": age,
            "Condition_Category": "Acute Trauma / Critical Care" if req_service in ["Surgery", "ICU Care"] else "General Medical Condition",
            "Urgency_Level": urgency,
            "Required_Service": req_service,
            "Required_Diagnostic": "CT Scan / Bio-Lab" if req_service == "Advanced Diagnostics" else "Basic X-Ray",
            "Local_Capability": local_cap,
            "Referral_Required": ref_req,
            "Referral_Date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "Recommended_Hospital_ID": recommended_hosp,
            "Referral_Outcome": "Successful Transfer" if ref_req == "Yes" else "Treated Locally"
        })
        case_counter += 1
        
    with open(os.path.join(output_dir, "referral_cases.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(referral_rows[0].keys()))
        writer.writeheader()
        writer.writerows(referral_rows)
        
    print(f"Generated 7 relational datasets (200+ records) in '{output_dir}'.")


if __name__ == "__main__":
    generate_full_system_datasets(".")
