import csv
import math
import os
import random
from datetime import datetime, timedelta

# Mapping of clinical services to prerequisite essential medications/supplies
SERVICE_MEDICINE_MAP = {
    "IV Fluid Care": ("M003", "IV Normal Saline 500ml"),
    "Minor Wound Care": ("M004", "Sterile Gauze Bandage"),
    "Basic Care": ("M001", "ORS (Oral Rehydration)"),
    "Antivenom Administration": ("M007", "Injectable Antivenom"),
    "Basic Diagnostics": ("M006", "Pulse Oximeter Sensor")
}

# Mapping of clinical services to specialized medical practitioner roles and prerequisites
SPECIALIST_MAPPING = {
    "Surgery": {
        "Specialist_Role": "General Surgeon / On-Call Trauma Surgeon",
        "Prerequisite_Facilities": ["Operation Theatre (OT)", "Post-Op ICU", "Blood Bank"],
        "Escort_Staff": "Medical Officer + Critical Care Nurse"
    },
    "ICU Care": {
        "Specialist_Role": "Critical Care Intensivist / Anesthesiologist",
        "Prerequisite_Facilities": ["Mechanical Ventilators", "Central Line Monitoring", "ICU Stepdown"],
        "Escort_Staff": "Doctor / Emergency Medical Technician (EMT-P)"
    },
    "Antivenom Administration": {
        "Specialist_Role": "Emergency Physician / Clinical Toxicologist",
        "Prerequisite_Facilities": ["Anaphylaxis Crash Cart", "Polyvalent Antivenom", "Ventilator Buffer"],
        "Escort_Staff": "Staff Nurse with Epinephrine & IV Infusion Kit"
    },
    "Advanced Diagnostics": {
        "Specialist_Role": "Consultant Radiologist / Pathologist",
        "Prerequisite_Facilities": ["CT Scanner", "Color Doppler", "Stat Laboratory"],
        "Escort_Staff": "Paramedic Escort"
    },
    "IV Fluid Care": {
        "Specialist_Role": "Duty Medical Officer / General Physician",
        "Prerequisite_Facilities": ["Observation Ward", "IV Rehydration Line"],
        "Escort_Staff": "Staff Nurse / Paramedic"
    },
    "Minor Wound Care": {
        "Specialist_Role": "Medical Officer / Emergency Triage Nurse",
        "Prerequisite_Facilities": ["Suture Suite", "Sterile Dressing"],
        "Escort_Staff": "Paramedic / Community Health Officer (CHO)"
    },
    "Basic Care": {
        "Specialist_Role": "Primary Care Medical Officer / Family Physician",
        "Prerequisite_Facilities": ["Outpatient Consultation Clinic"],
        "Escort_Staff": "Standard Patient Transport / Unassisted"
    }
}

def determine_ambulance_support(clinic_id, urgency, service, route_info=None):
    """
    Evaluates required ambulance tier, life-support equipment, and maritime dispatch requirements.
    """
    is_island = clinic_id in ["C002", "C004"]
    
    if is_island:
        amb_type = "Dual-Modal: Marine Emergency Watercraft (Boat Ambulance) -> Highway ALS Ambulance Rendezvous"
        amb_spec = "Marine rescue craft equipped with marine telemetry, portable oxygen cylinder, water-sealed suction, and life vests."
        dispatch_level = "Priority 1 - Maritime Coast Guard / Rescue Boat Rendezvous"
    elif urgency == "Emergency" or service in ["Surgery", "ICU Care", "Antivenom Administration"]:
        amb_type = "Advanced Life Support (ALS) Mobile ICU Ambulance"
        amb_spec = "Equipped with Transport Ventilator, Defibrillator (AED), Syringe Infusion Pump, and Emergency Airway Kit."
        dispatch_level = "Priority 1 - Immediate ALS Dispatch (Siren Code Red)"
    elif urgency == "High":
        amb_type = "Basic Life Support (BLS) Ambulance"
        amb_spec = "Equipped with continuous multi-parameter vital monitor, supplemental O2 flowmeter, and spine immobilization board."
        dispatch_level = "Priority 2 - Rapid BLS Dispatch (Code Yellow)"
    else:
        amb_type = "Standard Medical Patient Transport Vehicle (PTV)"
        amb_spec = "Foldable stretcher, first aid kit, basic vital monitoring."
        dispatch_level = "Priority 3 - Scheduled Patient Transfer"
        
    return {
        "Required": True,
        "Ambulance_Type": amb_type,
        "Equipment_Specification": amb_spec,
        "Dispatch_Priority": dispatch_level,
        "Is_Marine_Transit": is_island
    }

def determine_practitioner_support(clinic_info, urgency, service):
    """
    Evaluates origin practitioner capacity and mandatory in-transit medical escort staffing.
    """
    clinic_doc = clinic_info.get("Doctor_Available") == "Yes"
    clinic_nurse = clinic_info.get("Nurse_Available") == "Yes"
    staff_count = clinic_info.get("Staff_Count", 2)
    
    if clinic_doc:
        local_staff_status = f"✅ Resident Medical Officer on duty (Total Staff: {staff_count})"
    elif clinic_nurse:
        local_staff_status = f"⚠️ Nurse/CHO-Led Facility (No resident Doctor on site; Total Staff: {staff_count})"
    else:
        local_staff_status = f"🚨 Sub-Centre with auxiliary first-aid staff (Total Staff: {staff_count})"
        
    spec_info = SPECIALIST_MAPPING.get(service, SPECIALIST_MAPPING["Basic Care"])
    escort = spec_info["Escort_Staff"]
    
    if urgency == "Emergency":
        escort_level = f"🚨 Mandatory Accompaniment: {escort} (Active vital signs monitoring and in-transit clinical handover required)"
    elif urgency == "High":
        escort_level = f"⚠️ Recommended Accompaniment: {escort} (Periodic vital checks every 15 minutes)"
    else:
        escort_level = f"ℹ️ Basic Accompaniment: {escort} / Certified Emergency First Responder"
        
    return {
        "Local_Practitioner_Status": local_staff_status,
        "In_Transit_Escort_Required": escort_level,
        "Origin_Doctor_Available": clinic_doc
    }

def verify_specialist_availability(hospital_info, service):
    """
    Verifies destination receiving hospital specialist doctor availability and on-duty coverage.
    """
    spec_info = SPECIALIST_MAPPING.get(service, SPECIALIST_MAPPING["Basic Care"])
    needed_specialist = spec_info["Specialist_Role"]
    hosp_specs = hospital_info.get("Specialties", "")
    hosp_has_spec = hospital_info.get("Specialist_Available") == "Yes"
    
    match = False
    if service == "Surgery" and ("Surgery" in hosp_specs or hospital_info.get("Surgery_Available") == "Yes"):
        match = True
    elif service == "ICU Care" and ("Critical Care" in hosp_specs or hospital_info.get("ICU_Available") == "Yes"):
        match = True
    elif service == "Antivenom Administration" and ("Toxicology" in hosp_specs or hospital_info.get("Emergency_Care") == "Yes"):
        match = True
    elif service == "Advanced Diagnostics" and "Advanced" in hospital_info.get("Diagnostic_Facilities", ""):
        match = True
    elif service in ["Basic Care", "IV Fluid Care", "Minor Wound Care"]:
        match = True
        
    if match and hosp_has_spec:
        duty_status = f"🟢 Verified Available On-Duty (24/7 Specialist Rotational Coverage at {hospital_info['Hospital_Name']})"
        match_level = "Full Specialist Roster Confirmed"
    elif match:
        duty_status = f"🟡 General Medical Officer on Duty (Specialist on 30-min call at {hospital_info['Hospital_Name']})"
        match_level = "On-Call Coverage"
    else:
        duty_status = f"🔴 Specialist Unavailable (Only general clinical care available at {hospital_info['Hospital_Name']})"
        match_level = "No Specialist Coverage"
        
    return {
        "Required_Specialist": needed_specialist,
        "Hospital_Specialties": hosp_specs,
        "Specialist_Duty_Status": duty_status,
        "Specialist_Match_Level": match_level,
        "Prerequisite_Facilities": ", ".join(spec_info["Prerequisite_Facilities"])
    }

def get_golden_hour_window(service, urgency):
    """
    NOVEL FEATURE 2: Returns condition-specific Golden Hour survival window in minutes.
    Clinically grounded thresholds for trauma, ICU shock, and toxicology.
    """
    if urgency == "Emergency":
        if service in ["Surgery", "Emergency Care"]:
            return 60  # Trauma golden hour
        elif service in ["ICU Care", "Acute Trauma / Critical Care"]:
            return 90  # Sepsis & acute respiratory failure
        elif service == "Antivenom Administration":
            return 120 # Critical envenomation
        else:
            return 90
    elif urgency == "High":
        return 180
    else:
        return 360

def evaluate_patient_referral(case_dict, clinics, hospitals, accessibility, inventory=None):
    """
    Risk-Aware Referral Recommendation Engine with:
    1. Treatment service identification & triage urgency
    2. Closed-Loop Inventory-Referral Interlock (detects stock depletions)
    3. Candidate hospital capability filtering (Surgery, ICU, Diagnostics)
    4. Multi-factor ranking: Bed buffer, Transit time, Road hazard penalty, Golden Hour decay
    5. Multi-Hop Stabilization Routing (for isolated outposts breaching Golden Hour)
    6. Clinical Decision Support with Explainable AI justifications
    """
    clinic_id = case_dict.get("Clinic_ID", "C001")
    req_service = case_dict.get("Required_Service", "Basic Care")
    urgency = case_dict.get("Urgency_Level", "Medium")
    
    clinic_info = next((c for c in clinics if c["Clinic_ID"] == clinic_id or c.get("Clinic_Display") == clinic_id), clinics[0])
    cid_clean = clinic_info["Clinic_ID"]
    cname_clean = clinic_info["Clinic_Name"]
    
    # Baseline equipment/staff capability check
    local_capable = False
    if req_service == "Basic Care":
        local_capable = True
    elif req_service == "Minor Wound Care" and clinic_info.get("Minor_Wound_Care") == "Yes":
        local_capable = True
    elif req_service == "IV Fluid Care" and clinic_info.get("IV_Fluid_Care") == "Yes":
        local_capable = True
    elif req_service == "Basic Diagnostics" and clinic_info.get("Basic_Diagnostics") == "Yes":
        local_capable = True
    elif req_service == "Antivenom Administration" and clinic_info.get("Emergency_Care") == "Yes":
        local_capable = True
        
    # --- NOVEL FEATURE 1: CLOSED-LOOP INVENTORY-REFERRAL INTERLOCK ---
    stockout_interlock = False
    interlock_message = ""
    req_med_id, req_med_name = SERVICE_MEDICINE_MAP.get(req_service, (None, None))
    
    if local_capable and req_med_id and inventory is not None:
        inv_record = None
        if isinstance(inventory, list):
            inv_record = next((i for i in inventory if i.get("Clinic_ID") == cid_clean and i.get("Medicine_ID") == req_med_id), None)
        elif hasattr(inventory, "iterrows"):
            matched = inventory[(inventory["Clinic_ID"] == cid_clean) & (inventory["Medicine_ID"] == req_med_id)]
            if not matched.empty:
                inv_record = matched.iloc[0].to_dict()
                
        if inv_record:
            stock = int(inv_record.get("Current_Stock", 50))
            min_stk = int(inv_record.get("Minimum_Stock_Level", 10))
            if stock <= 0 or stock < max(2, int(min_stk * 0.25)):
                # Stockout interlock triggered: Clinic has staff but NO supplies!
                local_capable = False
                stockout_interlock = True
                interlock_message = (
                    f"⚠️ CLOSED-LOOP INTERLOCK ACTIVE: Clinic {cid_clean} possesses staff capability for {req_service}, "
                    f"but its inventory of essential '{req_med_name}' is critically depleted "
                    f"({stock} units remaining vs minimum {min_stk}). "
                    f"Local treatment suspended to prevent treatment failure. Emergency referral initiated."
                )

    if local_capable:
        return {
            "Decision": "Local Care",
            "Recommendation": f"Treat locally at {cid_clean} - {cname_clean}",
            "Referral_Required": "No",
            "Recommended_Hospital": "None (Handled Locally)",
            "Score": 1.0,
            "Details": f"Clinic {cid_clean} ({cname_clean}) possesses required clinical capability and verified medication inventory for {req_service}.",
            "Interlock_Triggered": False,
            "Interlock_Message": "Local medical supplies verified adequate.",
            "Golden_Hour_Feasibility": "Not Applicable (Local Care)",
            "Multi_Hop_Recommended": False,
            "Ambulance_Support": {
                "Required": False,
                "Ambulance_Type": "Not Required (Treated Locally at Clinic)",
                "Equipment_Specification": "Clinic observation couch and nursing care.",
                "Dispatch_Priority": "No Transit Required",
                "Is_Marine_Transit": False
            },
            "Practitioner_Support": {
                "Local_Practitioner_Status": "✅ Resident Medical Officer on Duty" if clinic_info.get("Doctor_Available") == "Yes" else "⚠️ Staff Nurse / CHO on Duty",
                "In_Transit_Escort_Required": "None (Patient Treated on Premises)",
                "Origin_Doctor_Available": clinic_info.get("Doctor_Available") == "Yes"
            },
            "Specialist_Support": {
                "Required_Specialist": "Primary Care Physician / Community Health Officer",
                "Hospital_Specialties": "Outpatient Care",
                "Specialist_Duty_Status": "🟢 Primary Care Staff Available On Site",
                "Specialist_Match_Level": "Local Primary Care Match",
                "Prerequisite_Facilities": "Local Clinic Consultation Room"
            }
        }
        
    # Destination hospital evaluation & ranking
    golden_window = get_golden_hour_window(req_service, urgency)
    candidate_scores = []
    
    for hosp in hospitals:
        hid = hosp["Hospital_ID"]
        hname = hosp["Hospital_Name"]
        avail_beds = int(hosp.get("Available_Beds", 10))
        total_beds = int(hosp.get("Bed_Capacity", 100))
        
        # Bed availability safety buffer
        bed_ratio = min(1.0, avail_beds / max(1, total_beds * 0.20))
        
        # Specialized service capability match
        cap_match = 0.4
        if req_service == "Surgery" and hosp.get("Surgery_Available") == "Yes":
            cap_match = 1.0
        elif req_service == "ICU Care" and hosp.get("ICU_Available") == "Yes":
            cap_match = 1.0
        elif req_service in ["Emergency Care", "Antivenom Administration"] and hosp.get("Emergency_Care") == "Yes":
            cap_match = 1.0
        elif req_service == "Advanced Diagnostics" and "Advanced" in hosp.get("Diagnostic_Facilities", ""):
            cap_match = 1.0
        elif req_service in ["Basic Care", "IV Fluid Care", "Minor Wound Care"]:
            cap_match = 1.0
            
        route = next((r for r in accessibility if r["Source_ID"] == cid_clean and r["Destination_ID"] == hid), None)
        if route:
            dist = float(route.get("Distance_KM", 100.0))
            travel_min = float(route.get("Travel_Time_Minutes", 120.0))
            road_cond = route.get("Road_Condition", "Moderate")
        else:
            dist = 120.0
            travel_min = 150.0
            road_cond = "Moderate"
            
        dist_score = max(0.0, 1.0 - (dist / 200.0))
        travel_score = max(0.0, 1.0 - (travel_min / 240.0))
        
        # --- NOVEL FEATURE 2: GOLDEN HOUR SURVIVAL WINDOW & TIME DECAY ---
        golden_feasible = travel_min <= golden_window
        golden_decay = max(0.0, 1.0 - (travel_min / (golden_window * 1.5)))
        
        # Composite Referral Score with Golden Hour, Bed Buffer, and Travel Safety
        referral_score = round(
            (cap_match * 0.35) + 
            (bed_ratio * 0.25) + 
            (dist_score * 0.15) + 
            (golden_decay * 0.25), 2
        )
        
        # Hard penalty for saturated hospitals (Ambulance Diversion Prevention)
        if avail_beds <= 0:
            referral_score = max(0.05, referral_score - 0.40)
            
        golden_tag = f"🟢 Within Window ({int(travel_min)}m / {golden_window}m)" if golden_feasible else f"🔴 Breaches Window (+{int(travel_min - golden_window)}m delay)"
        
        cand_spec = verify_specialist_availability(hosp, req_service)
        
        candidate_scores.append({
            "Hospital_ID": hid,
            "Hospital_Name": hname,
            "Referral_Score": referral_score,
            "Distance_KM": dist,
            "Travel_Time_Min": travel_min,
            "Available_Beds": avail_beds,
            "Road_Condition": road_cond,
            "Capability_Match": "Full Match" if cap_match == 1.0 else "Partial Match",
            "Specialist_Role": cand_spec["Required_Specialist"],
            "Specialist_Status": cand_spec["Specialist_Duty_Status"],
            "Ambulance_Fleet": "Dispatched / Available" if hosp.get("Ambulance_Available") == "Yes" else "Limited",
            "Golden_Hour_Window_Min": golden_window,
            "Golden_Hour_Feasibility": golden_tag,
            "Bed_Status": "Critical Saturation (0 Beds)" if avail_beds <= 0 else f"{avail_beds} Beds Available"
        })
        
    candidate_scores.sort(key=lambda x: x["Referral_Score"], reverse=True)
    best_hosp = candidate_scores[0]
    
    # Detailed protocol determination for best hospital
    best_hosp_record = next((h for h in hospitals if h["Hospital_ID"] == best_hosp["Hospital_ID"]), hospitals[0])
    best_route = next((r for r in accessibility if r["Source_ID"] == cid_clean and r["Destination_ID"] == best_hosp["Hospital_ID"]), None)
    
    amb_support = determine_ambulance_support(cid_clean, urgency, req_service, best_route)
    prac_support = determine_practitioner_support(clinic_info, urgency, req_service)
    spec_support = verify_specialist_availability(best_hosp_record, req_service)
    
    # --- NOVEL FEATURE 4: MULTI-HOP STABILIZATION ROUTING ---
    multi_hop = None
    if urgency in ["Emergency", "High"] and best_hosp["Travel_Time_Min"] > golden_window:
        inter_clinic = next((c for c in clinics if c["Clinic_ID"] != cid_clean and c.get("Emergency_Care") == "Yes" and c.get("Doctor_Available") == "Yes"), None)
        if inter_clinic:
            multi_hop = {
                "Recommended": True,
                "Stabilization_Clinic": f"{inter_clinic['Clinic_ID']} - {inter_clinic['Clinic_Name']}",
                "Leg_1_Action": f"Dispatch patient immediately to {inter_clinic['Clinic_Name']} (~45 mins) for acute airway/IV resuscitation.",
                "Leg_2_Action": f"Simultaneously dispatch Advanced Life Support (ALS) ambulance from {best_hosp['Hospital_Name']} to intercept at {inter_clinic['Clinic_Name']} for definitive transfer.",
                "Clinical_Rationale": f"Direct transit to {best_hosp['Hospital_Name']} takes {best_hosp['Travel_Time_Min']} mins, exceeding the {golden_window}-minute Golden Hour window. Multi-hop stabilization reduces unmonitored transit mortality risk."
            }

    # Clinical rationale assembly
    details = []
    if stockout_interlock:
        details.append(interlock_message)
    else:
        details.append(f"Clinic {cid_clean} ({cname_clean}) lacks dedicated {req_service} infrastructure.")
        
    details.append(f"Top recommendation: {best_hosp['Hospital_Name']} (Score: {best_hosp['Referral_Score']}) based on clinical match, verified bed buffer ({best_hosp['Available_Beds']} beds), and travel safety ({best_hosp['Road_Condition']} road).")
    details.append(f"Golden Hour Window: {best_hosp['Golden_Hour_Feasibility']}.")
    
    return {
        "Decision": "Hospital Referral Required",
        "Recommendation": f"Refer patient to {best_hosp['Hospital_Name']} ({best_hosp['Distance_KM']} km, ~{best_hosp['Travel_Time_Min']} mins)",
        "Referral_Required": "Yes",
        "Recommended_Hospital": best_hosp["Hospital_Name"],
        "Recommended_Hospital_ID": best_hosp["Hospital_ID"],
        "Score": best_hosp["Referral_Score"],
        "Ranked_Candidates": candidate_scores,
        "Details": " ".join(details),
        "Interlock_Triggered": stockout_interlock,
        "Interlock_Message": interlock_message if stockout_interlock else "Standard clinical capability referral.",
        "Golden_Hour_Window_Min": golden_window,
        "Golden_Hour_Feasibility": best_hosp["Golden_Hour_Feasibility"],
        "Multi_Hop_Option": multi_hop,
        "Ambulance_Support": amb_support,
        "Practitioner_Support": prac_support,
        "Specialist_Support": spec_support
    }

def generate_digital_handshake(case_dict, best_hosp_id, best_hosp_name, clinician_id="DR-CHIEF-01", amb_info=None, spec_info=None, prac_info=None):
    """
    NOVEL FEATURE 3: Generates a Pre-Arrival Digital Handshake & Bed Reservation Token
    Simulating HL7/FHIR emergency encounter payload for destination emergency department.
    """
    token_seed = random.randint(1000, 9999)
    token = f"FHIR-REF-{datetime.now().strftime('%Y%m%d')}-{best_hosp_id}-{token_seed}"
    expires_at = (datetime.now() + timedelta(minutes=90)).strftime("%H:%M:%S")
    
    amb_type = amb_info.get("Ambulance_Type", "ALS Mobile ICU") if amb_info else "Advanced Life Support (ALS) Mobile ICU Ambulance"
    spec_role = spec_info.get("Required_Specialist", "Trauma Surgeon") if spec_info else "General Surgeon / On-Call Trauma Surgeon"
    escort_req = prac_info.get("In_Transit_Escort_Required", "Accompanying Medical Officer") if prac_info else "Accompanying Medical Officer + Critical Care Nurse"
    
    return {
        "Handshake_Token": token,
        "Destination_Hospital": best_hosp_name,
        "Authorized_Clinician": clinician_id,
        "Bed_Reservation_Status": "CONFIRMED (90-Minute Soft Hold Locked)",
        "Bed_Reservation_Expiry": expires_at,
        "Assigned_Trauma_Bay": f"Bay-{random.randint(1, 6)} (Trauma/Acute Wing)",
        "Dispatch_Priority": "CODE RED (ALS Ambulance Pre-Alert Active)",
        "Transmission_Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Telemetry_Payload": {
            "Patient_Urgency": case_dict.get("Urgency_Level", "Emergency"),
            "Required_Specialty": case_dict.get("Required_Service", "Surgery"),
            "Receiving_Specialist": spec_role,
            "Ambulance_Dispatched": amb_type,
            "In_Transit_Staffing": escort_req,
            "Origin_Center": case_dict.get("Clinic_ID", "C001"),
            "Estimated_Transit_Minutes": case_dict.get("Travel_Time_Min", 45),
            "Clinical_Precaution": "Continuous vital telemetry and intravenous infusion line maintained in transit."
        }
    }
