import pandas as pd
import numpy as np
import os

def generate_medical_dataset(output_path="medical_supply_data.csv"):
    np.random.seed(42)
    
    centres = ["Centre A", "Centre B", "Centre C", "Centre D", "Centre E"]
    items = ["ORS", "Antibiotic", "Diagnostic Kit", "Bandage", "Pulse Oximeter"]
    
    # Base characteristics per item: (base_demand, seasonality_factor, default_lead_time, criticality_score)
    item_metadata = {
        "ORS": {"base": 90, "seasonality": 25, "lead_time": 5, "criticality": 5},
        "Antibiotic": {"base": 60, "seasonality": 15, "lead_time": 7, "criticality": 5},
        "Diagnostic Kit": {"base": 40, "seasonality": 10, "lead_time": 10, "criticality": 4},
        "Bandage": {"base": 120, "seasonality": 15, "lead_time": 3, "criticality": 2},
        "Pulse Oximeter": {"base": 15, "seasonality": 5, "lead_time": 14, "criticality": 4}
    }
    
    # Centre multiplier (demand scale per centre)
    centre_multipliers = {
        "Centre A": 1.2,
        "Centre B": 1.0,
        "Centre C": 0.8,
        "Centre D": 1.1,
        "Centre E": 0.7
    }
    
    records = []
    
    # 12 months in 2026: 2026-01 to 2026-12
    dates = [f"2026-{m:02d}" for m in range(1, 13)]
    
    for c in centres:
        c_mult = centre_multipliers[c]
        for item_name, meta in item_metadata.items():
            base_dem = meta["base"] * c_mult
            season_amp = meta["seasonality"] * c_mult
            lead_t = meta["lead_time"]
            crit = meta["criticality"]
            
            # Stock policy simulation per item
            current_stock = int(base_dem * np.random.uniform(0.7, 1.3))
            
            for m_idx, date_str in enumerate(dates):
                # Add seasonal trend (e.g. higher demand in summer/monsoon months 5-8)
                seasonal_wave = np.sin((m_idx / 12.0) * 2 * np.pi) * season_amp
                noise = np.random.normal(0, base_dem * 0.08)
                
                demand = max(10, int(base_dem + seasonal_wave + noise))
                
                # Stock varies over time; in recent months some items might be low on stock
                if m_idx == 11: # Latest month (2026-12)
                    # Introduce realistic variance for current stock
                    stock_ratio = np.random.choice([0.6, 0.8, 1.1, 0.4, 1.3])
                    stock = max(5, int(demand * stock_ratio))
                else:
                    stock = max(10, int(demand * np.random.uniform(0.8, 1.4)))
                
                records.append({
                    "date": date_str,
                    "centre": c,
                    "item": item_name,
                    "demand": demand,
                    "stock": stock,
                    "lead_time": lead_t,
                    "criticality": crit
                })
                
    df = pd.DataFrame(records)
    
    # Ensure parent directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_medical_dataset()
