import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from generate_dataset import generate_medical_dataset
from model import load_data, train_and_predict_next_month, HAS_XGBOOST
from allocation import calculate_allocations, get_default_partner_inventory

# Page configuration
st.set_page_config(
    page_title="Medical Supply Demand Forecaster & Risk Allocator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium UI design
st.markdown("""
<style>
    /* Dark glassmorphism card styling */
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
    .badge-red {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-yellow {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fde047;
        border: 1px solid #f59e0b;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid #22c55e;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
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

DATA_FILE = "medical_supply_data.csv"

# Ensure data file exists
if not os.path.exists(DATA_FILE):
    generate_medical_dataset(DATA_FILE)

@st.cache_data
def get_historical_data():
    return load_data(DATA_FILE)

@st.cache_data
def get_predictions_and_risk():
    df_hist = load_data(DATA_FILE)
    return train_and_predict_next_month(df_hist)

# Load data
df_hist = get_historical_data()
df_risk = get_predictions_and_risk()

# Sidebar Navigation & Filters
st.sidebar.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=70)
st.sidebar.title("Healthcare Supply AI")
st.sidebar.caption("Proof of Concept • Demand & Risk System")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Interactive Filters")

all_centres = sorted(df_hist["centre"].unique())
selected_centre = st.sidebar.selectbox("Select Healthcare Centre", ["All Centres"] + all_centres)

all_items = sorted(df_hist["item"].unique())
selected_item = st.sidebar.selectbox("Select Medical Item", ["All Items"] + all_items)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model Engine**: {'XGBoost Regressor' if HAS_XGBOOST else 'Random Forest Regressor'}\n\n"
    f"**Data Scope**: 5 Centres × 5 Items × 12 Months\n\n"
    f"**Forecast Target**: Month 13 (2027-01)"
)

# Header Section
st.title("🏥 Medical Supply Demand & Risk Allocator")
st.markdown("Predict future demand, evaluate shortage risk scores, and optimize central hospital stock allocations across healthcare centres.")

# Top KPI Summary Cards
total_pred_demand = df_risk["predicted_demand"].sum()
total_stock = df_risk["current_stock"].sum()
total_shortage = df_risk["expected_shortage"].sum()
high_risk_count = len(df_risk[df_risk["status"].str.contains("High")])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Demand</div>
        <div class="metric-value">{total_pred_demand:,}</div>
        <div class="metric-sub">Units for next month</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Stock</div>
        <div class="metric-value">{total_stock:,}</div>
        <div class="metric-sub">Units across all centres</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Expected Shortage</div>
        <div class="metric-value" style="color: #f87171;">{total_shortage:,}</div>
        <div class="metric-sub">Deficit units needed</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High Risk Items 🔴</div>
        <div class="metric-value" style="color: #fb7185;">{high_risk_count}</div>
        <div class="metric-sub">Requires immediate allocation</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Step 1 & 2: Historical Demand",
    "🤖 Step 3-6: ML Forecast & Risk Ranking",
    "🏥 Step 7: Partner Hospital Allocation",
    "📋 Complete Dataset View"
])

# ----------------------------------------------------
# TAB 1: HISTORICAL DEMAND VISUALIZATION
# ----------------------------------------------------
with tab1:
    st.markdown("<div class='section-header'>Step 1 & 2: Historical Demand Analysis</div>", unsafe_allow_html=True)
    
    filtered_hist = df_hist.copy()
    if selected_centre != "All Centres":
        filtered_hist = filtered_hist[filtered_hist["centre"] == selected_centre]
    if selected_item != "All Items":
        filtered_hist = filtered_hist[filtered_hist["item"] == selected_item]
        
    st.write(f"Showing historical demand data ({len(filtered_hist)} records):")
    
    # Plotly Trend Chart
    fig_hist = px.line(
        filtered_hist,
        x="date",
        y="demand",
        color="item" if selected_centre != "All Centres" else "centre",
        line_group="item",
        symbol="item",
        markers=True,
        title=f"Historical Demand Trend (2026-01 to 2026-12) - {selected_centre} / {selected_item}",
        labels={"date": "Month", "demand": "Demand (Units)", "centre": "Centre", "item": "Item"},
        template="plotly_dark"
    )
    fig_hist.update_layout(height=450, hovermode="x unified", font=dict(family="Inter, sans-serif"))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Demand Distribution by Item")
        fig_box = px.box(
            filtered_hist, x="item", y="demand", color="item",
            template="plotly_dark", title="Demand Variance per Item"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_b:
        st.subheader("Demand vs Current Stock")
        avg_df = filtered_hist.groupby("item")[["demand", "stock"]].mean().reset_index()
        fig_bar = px.bar(
            avg_df, x="item", y=["demand", "stock"], barmode="group",
            template="plotly_dark", title="Average Monthly Demand vs Average Stock Pool"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------------------------------
# TAB 2: ML FORECAST & RISK SCORING
# ----------------------------------------------------
with tab2:
    st.markdown("<div class='section-header'>Step 3 - 6: ML Prediction & Item Risk Ranking</div>", unsafe_allow_html=True)
    
    st.markdown("""
    **Risk Score Calculation Pipeline:**
    $$\\text{Predicted Demand} \\rightarrow \\text{Current Stock} \\rightarrow \\text{Expected Shortage} \\rightarrow \\text{Criticality (1-5)} \\rightarrow \\text{Lead Time} \\rightarrow \\text{Risk Score (0-1)}$$
    """)
    
    filtered_risk = df_risk.copy()
    if selected_centre != "All Centres":
        filtered_risk = filtered_risk[filtered_risk["centre"] == selected_centre]
    if selected_item != "All Items":
        filtered_risk = filtered_risk[filtered_risk["item"] == selected_item]
        
    # Grouped display per Centre (as requested in Step 6 format)
    centres_to_show = [selected_centre] if selected_centre != "All Centres" else sorted(filtered_risk["centre"].unique())
    
    for c in centres_to_show:
        c_df = filtered_risk[filtered_risk["centre"] == c].sort_values("risk_score", ascending=False)
        
        st.subheader(f"📍 {c} - Item Risk Ranking")
        
        # Display nicely formatted table matching Step 6 format
        table_rows = []
        for idx, row in c_df.reset_index(drop=True).iterrows():
            table_rows.append({
                "Rank": f"#{idx+1}",
                "Medical Item": row["item"],
                "Predicted Demand": row["predicted_demand"],
                "Current Stock": row["current_stock"],
                "Expected Shortage": row["expected_shortage"],
                "Criticality": f"{row['criticality']}/5",
                "Lead Time": f"{row['lead_time']} days",
                "Risk Score": f"{row['risk_score']:.2f}",
                "Status": row["status"]
            })
            
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    help="Calculated risk level",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                )
            }
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
    # Risk Matrix Scatter Plot
    st.markdown("<div class='section-header'>Risk Matrix: Shortage vs Lead Time & Criticality</div>", unsafe_allow_html=True)
    fig_scatter = px.scatter(
        filtered_risk,
        x="expected_shortage",
        y="risk_score",
        size="criticality",
        color="status",
        hover_name="item",
        facet_col="centre" if selected_centre == "All Centres" else None,
        color_discrete_map={"🔴 High": "#ef4444", "🟡 Medium": "#f59e0b", "🟢 Low": "#22c55e"},
        title="Inventory Risk Matrix",
        template="plotly_dark"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------------------------------------------
# TAB 3: PARTNER HOSPITAL ALLOCATION
# ----------------------------------------------------
with tab3:
    st.markdown("<div class='section-header'>Step 7: Partner Hospital Allocation Engine</div>", unsafe_allow_html=True)
    
    st.info("Partner Hospital A operates a central reserve stock pool. The allocation algorithm automatically assigns inventory to healthcare centres based on predicted shortages and risk rank.")
    
    # Sidebar or top controls for central depot stock customization
    st.subheader("📦 Central Depot Reserve Stock (Partner Hospital A)")
    
    default_inv = get_default_partner_inventory()
    inv_cols = st.columns(len(default_inv))
    
    custom_inventory = {}
    for i, (item_k, default_v) in enumerate(default_inv.items()):
        with inv_cols[i]:
            custom_inventory[item_k] = st.number_input(
                f"{item_k}", min_value=0, max_value=1000, value=default_v, step=10
            )
            
    # Calculate allocations
    alloc_df, remaining_stock = calculate_allocations(df_risk, custom_inventory)
    
    col_alloc1, col_alloc2 = st.columns([3, 2])
    
    with col_alloc1:
        st.subheader("💡 Recommended Allocation Directives")
        if len(alloc_df) > 0:
            # Display step-7 style recommendation text
            rec_texts = []
            for _, r in alloc_df.iterrows():
                rec_texts.append(f"**Partner Hospital A** → **{r['to']}** : {r['allocated_quantity']} units of `{r['item']}` ({r['risk_status']} Risk, Shortage: {r['expected_shortage']})")
                
            for txt in rec_texts:
                st.markdown(f"👉 {txt}")
                
            st.markdown("---")
            st.dataframe(
                alloc_df[["from", "to", "item", "expected_shortage", "allocated_quantity", "remaining_shortage", "risk_status"]],
                use_container_width=True
            )
        else:
            st.success("No shortages detected! All healthcare centres have sufficient current stock.")
            
    with col_alloc2:
        st.subheader("📊 Remaining Central Pool")
        rem_data = pd.DataFrame([
            {"Item": k, "Initial Stock": custom_inventory[k], "Remaining Pool": v, "Allocated": custom_inventory[k] - v}
            for k, v in remaining_stock.items()
        ])
        st.dataframe(rem_data, use_container_width=True)
        
        fig_alloc_pie = px.bar(
            rem_data, x="Item", y=["Allocated", "Remaining Pool"],
            title="Central Stock Utilization", barmode="stack", template="plotly_dark"
        )
        st.plotly_chart(fig_alloc_pie, use_container_width=True)

# ----------------------------------------------------
# TAB 4: COMPLETE DATASET
# ----------------------------------------------------
with tab4:
    st.markdown("<div class='section-header'>Complete Historical Dataset (5 Centres × 5 Items × 12 Months)</div>", unsafe_allow_html=True)
    st.dataframe(df_hist, use_container_width=True)
    
    st.download_button(
        label="📥 Download CSV Dataset",
        data=df_hist.to_csv(index=False),
        file_name="medical_supply_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("Proof of Concept Application • Built with Streamlit, Pandas, XGBoost & Plotly")
