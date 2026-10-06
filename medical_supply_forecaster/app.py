import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from generate_all_datasets import generate_full_system_datasets
from model import load_data, compare_models_and_predict
from inventory_engine import load_relational_datasets, compute_inventory_metrics
from referral_engine import evaluate_patient_referral
from allocation import calculate_allocations, get_default_partner_inventory

# Page configuration
st.set_page_config(
    page_title="AI Healthcare Supply Chain & Patient Referral System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #334155;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Ensure 7 relational datasets exist
if not os.path.exists("medical_demand.csv") or not os.path.exists("inventory.csv"):
    generate_full_system_datasets(".")

@st.cache_data
def load_all_data():
    return load_relational_datasets(".")

@st.cache_data
def get_forecasting_results():
    df_demand = load_data("medical_demand.csv")
    return compare_models_and_predict(df_demand)

# Load data
df_demand, df_inventory, df_clinics, df_hospitals, df_referrals, df_access, df_medicines = load_all_data()
df_comp, df_risk = get_forecasting_results()
df_inv_opt = compute_inventory_metrics(df_demand, df_inventory, df_access, df_medicines)

# Sidebar Navigation & Filters
st.sidebar.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=70)
st.sidebar.title("Healthcare AI Operations")
st.sidebar.caption("Supply Chain & Referral Decision System")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Interactive Filters")

all_clinics = sorted(df_clinics["Clinic_ID"].unique())
selected_clinic = st.sidebar.selectbox("Select Clinic", ["All Clinics"] + all_clinics)

all_meds = sorted(df_medicines["Medicine_Name"].unique())
selected_med = st.sidebar.selectbox("Select Medicine", ["All Medicines"] + all_meds)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Architecture Modules**:\n"
    "1. 📈 Demand Forecasting (ARIMA/Prophet/XGBoost)\n"
    "2. 📦 Safety Stock, ROP & EOQ Optimization\n"
    "3. 🚨 Risk-Aware Priority Engine\n"
    "4. 🚑 Patient Referral & Local Capability Matching\n"
    "5. 🏥 Hospital Capability & Accessibility Scoring"
)

# Header Section
st.title("🏥 AI-Driven Remote Healthcare & Supply Chain System")
st.markdown("Integrated decision platform combining **Multi-Model Demand Forecasting**, **Inventory Optimization (Safety Stock / ROP / EOQ)**, and **Patient Referral Matching**.")

# Top KPI Summary Cards
total_pred_demand = df_risk["predicted_demand"].sum()
total_urgent_rep = len(df_inv_opt[df_inv_opt["Priority_Badge"].str.contains("URGENT")])
avg_stockout_risk = df_inv_opt["Stockout_Risk"].mean() * 100
total_referral_cases = len(df_referrals)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Forecasted Demand</div>
        <div class="metric-value">{total_pred_demand:,}</div>
        <div class="metric-sub">Units needed next month</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Urgent Replenishments 🔴</div>
        <div class="metric-value" style="color: #fb7185;">{total_urgent_rep}</div>
        <div class="metric-sub">Items below Reorder Point</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Stockout Risk</div>
        <div class="metric-value" style="color: #f87171;">{avg_stockout_risk:.1f}%</div>
        <div class="metric-sub">Across all outposts</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Referral Cases Tracked</div>
        <div class="metric-value">{total_referral_cases}</div>
        <div class="metric-sub">Capability checked</div>
    </div>
    """, unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Demand Forecasting & Model Evaluation",
    "📦 Inventory Optimization (Safety Stock, ROP & EOQ)",
    "🚑 Patient Referral & Hospital Matching",
    "🗺️ Route Accessibility Matrix",
    "📋 System Relational Datasets"
])

# ----------------------------------------------------
# TAB 1: DEMAND FORECASTING (ARIMA vs PROPHET vs XGBOOST)
# ----------------------------------------------------
with tab1:
    st.markdown("<div class='section-header'>Workflow Part 1: Demand Forecasting (ARIMA / Prophet / XGBoost)</div>", unsafe_allow_html=True)
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.subheader("🏆 Model Victory Breakdown")
        wins = df_risk["selected_model"].value_counts().reset_index()
        wins.columns = ["Model", "Series Won"]
        st.dataframe(wins, use_container_width=True)
        
        fig_pie = px.pie(wins, names="Model", values="Series Won", title="Best Model Win Ratio", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_m2:
        st.subheader("📊 Model MAE & RMSE Error Benchmarks")
        avg_err = df_comp.groupby("model")[["mae", "rmse"]].mean().reset_index()
        fig_err = px.bar(avg_err, x="model", y=["mae", "rmse"], barmode="group", title="Average Error Rate (Lower is Better)", template="plotly_dark")
        st.plotly_chart(fig_err, use_container_width=True)
        
    st.markdown("<div class='section-header'>Historical Demand Trends & Next Month Forecast</div>", unsafe_allow_html=True)
    fig_line = px.line(df_demand, x="Date", y="Quantity_Consumed", color="Medicine_Name", facet_col="Clinic_ID", title="Monthly Consumption Trends per Clinic", template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

# ----------------------------------------------------
# TAB 2: INVENTORY OPTIMIZATION (SAFETY STOCK, ROP & EOQ)
# ----------------------------------------------------
with tab2:
    st.markdown("<div class='section-header'>Workflow Part 2: Stockout Risk, Safety Stock, ROP & EOQ Engine</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **Operations Management Formulas Implemented**:
    - **Safety Stock ($SS$)**: $Z \cdot \sigma_d \cdot \sqrt{L_{\text{months}}}$
    - **Reorder Point ($ROP$)**: $(\text{Avg Demand} \cdot L_{\text{months}}) + SS$
    - **Economic Order Quantity ($EOQ$)**: $\sqrt{\\frac{2 \cdot D \cdot S}{H}}$
    - **Replenishment Priority Score**: $0.40(\text{Stockout Risk}) + 0.30(\text{Criticality}) + 0.15(\text{Lead Time}) + 0.15(\text{Accessibility Risk})$
    """)
    
    filtered_inv = df_inv_opt.copy()
    if selected_clinic != "All Clinics":
        filtered_inv = filtered_inv[filtered_inv["Clinic_ID"] == selected_clinic]
    if selected_med != "All Medicines":
        filtered_inv = filtered_inv[filtered_inv["Medicine_Name"] == selected_med]
        
    display_cols = [
        "Clinic_ID", "Medicine_Name", "Current_Stock", "Safety_Stock", "Reorder_Point", "EOQ",
        "Stockout_Risk", "Accessibility_Risk", "Replenishment_Priority_Score", "Priority_Badge"
    ]
    
    st.dataframe(
        filtered_inv[display_cols],
        use_container_width=True,
        column_config={
            "Replenishment_Priority_Score": st.column_config.ProgressColumn(
                "Priority Score", min_value=0.0, max_value=1.0, format="%.2f"
            )
        }
    )
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.subheader("Current Stock vs Reorder Point (ROP)")
        fig_rop = px.bar(filtered_inv.head(10), x="Medicine_Name", y=["Current_Stock", "Reorder_Point", "Safety_Stock"], barmode="group", template="plotly_dark", title="Stock Position relative to ROP & Safety Stock")
        st.plotly_chart(fig_rop, use_container_width=True)
        
    with col_i2:
        st.subheader("EOQ (Economic Order Quantity) per Item")
        fig_eoq = px.bar(filtered_inv.head(10), x="Medicine_Name", y="EOQ", color="Criticality_Level", template="plotly_dark", title="Optimal Order Quantity (EOQ)")
        st.plotly_chart(fig_eoq, use_container_width=True)

# ----------------------------------------------------
# TAB 3: PATIENT REFERRAL & LOCAL CAPABILITY MATCHING
# ----------------------------------------------------
with tab3:
    st.markdown("<div class='section-header'>Workflow Part 3: Patient Requirement & Tele-Triage Decision Tree</div>", unsafe_allow_html=True)
    
    st.info("Implements bottom workflow of architecture diagram: Patient Requirement → Local Capability Check → If No → Hospital Capability + Distance Scoring → Referral Recommendation.")
    
    col_ref1, col_ref2 = st.columns([1, 1])
    
    with col_ref1:
        st.subheader("📥 Interactive Patient Triage Input")
        test_clinic = st.selectbox("Patient Origin Clinic", all_clinics, index=1)
        test_service = st.selectbox("Required Treatment Service", ["Surgery", "ICU Care", "Advanced Diagnostics", "Basic Care", "IV Fluid Care", "Minor Wound Care"])
        test_urgency = st.selectbox("Urgency Level", ["Emergency", "High", "Medium", "Low"])
        
        test_case = {
            "Clinic_ID": test_clinic,
            "Required_Service": test_service,
            "Urgency_Level": test_urgency
        }
        
        result = evaluate_patient_referral(test_case, df_clinics.to_dict("records"), df_hospitals.to_dict("records"), df_access.to_dict("records"))
        
        st.markdown("---")
        st.subheader("📋 Decision Output")
        if result["Decision"] == "Local Care":
            st.success(f"✅ **DECISION**: {result['Decision']}")
        else:
            st.warning(f"🚨 **DECISION**: {result['Decision']}")
            
        st.write(f"**Recommendation**: {result['Recommendation']}")
        st.write(f"**Details**: {result['Details']}")
        
    with col_ref2:
        st.subheader("🏥 Hospital Capability & Accessibility Scoring")
        if "Ranked_Candidates" in result:
            cand_df = pd.DataFrame(result["Ranked_Candidates"])
            st.dataframe(cand_df[["Hospital_ID", "Hospital_Name", "Referral_Score", "Distance_KM", "Travel_Time_Min", "Available_Beds", "Capability_Match"]], use_container_width=True)
            
            fig_cand = px.bar(cand_df, x="Hospital_Name", y="Referral_Score", color="Capability_Match", title="Hospital Suitability Score", template="plotly_dark")
            st.plotly_chart(fig_cand, use_container_width=True)
        else:
            st.success("Local clinic has full capability to treat patient. No referral required!")
            
    st.markdown("<div class='section-header'>Historical Referral Cases Log</div>", unsafe_allow_html=True)
    st.dataframe(df_referrals, use_container_width=True)

# ----------------------------------------------------
# TAB 4: ROUTE ACCESSIBILITY MATRIX
# ----------------------------------------------------
with tab4:
    st.markdown("<div class='section-header'>Route Accessibility & Transportation Matrix</div>", unsafe_allow_html=True)
    st.dataframe(df_access, use_container_width=True)
    
    fig_route = px.scatter(df_access, x="Distance_KM", y="Travel_Time_Minutes", size="Average_Delivery_Time_Days", color="Road_Condition", hover_name="Route_ID", title="Transit Distance vs Travel Time & Delivery Delay", template="plotly_dark")
    st.plotly_chart(fig_route, use_container_width=True)

# ----------------------------------------------------
# TAB 5: SYSTEM RELATIONAL DATASETS EXPLORER
# ----------------------------------------------------
with tab5:
    st.markdown("<div class='section-header'>Complete Relational Database System (7 CSV Datasets)</div>", unsafe_allow_html=True)
    
    dataset_choice = st.selectbox("Select Relational Dataset to View", [
        "medical_demand.csv (Demand Dataset)",
        "inventory.csv (Inventory Dataset)",
        "clinics.csv (Remote Clinics Dataset)",
        "hospitals.csv (Hospitals Dataset)",
        "referral_cases.csv (Patient Referral Dataset)",
        "accessibility.csv (Accessibility Dataset)",
        "medicines.csv (Medicines Master Dataset)"
    ])
    
    ds_map = {
        "medical_demand.csv (Demand Dataset)": df_demand,
        "inventory.csv (Inventory Dataset)": df_inventory,
        "clinics.csv (Remote Clinics Dataset)": df_clinics,
        "hospitals.csv (Hospitals Dataset)": df_hospitals,
        "referral_cases.csv (Patient Referral Dataset)": df_referrals,
        "accessibility.csv (Accessibility Dataset)": df_access,
        "medicines.csv (Medicines Master Dataset)": df_medicines
    }
    
    target_df = ds_map[dataset_choice]
    st.write(f"Displaying {len(target_df)} rows:")
    st.dataframe(target_df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("AI Healthcare Supply Chain & Referral System • Complete Architecture Implementation")
