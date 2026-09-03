import pandas as pd

def get_default_partner_inventory():
    """Default inventory pool available at partner hospital / central depot."""
    return {
        "ORS": 200,
        "Antibiotic": 120,
        "Diagnostic Kit": 50,
        "Bandage": 300,
        "Pulse Oximeter": 10
    }

def calculate_allocations(risk_df, central_inventory=None):
    """
    Step 7: Reallocate central partner hospital inventory to health centres based on risk and expected shortages.
    Prioritizes items with higher expected shortages and risk scores.
    """
    if central_inventory is None:
        central_inventory = get_default_partner_inventory()
        
    available_stock = central_inventory.copy()
    
    # Filter centres that have expected shortages
    shortage_df = risk_df[risk_df["expected_shortage"] > 0].copy()
    
    # Sort by risk_score (descending) and criticality (descending)
    shortage_df = shortage_df.sort_values(by=["item", "risk_score", "criticality"], ascending=[True, False, False])
    
    allocations = []
    
    for item_name in available_stock.keys():
        item_shortages = shortage_df[shortage_df["item"] == item_name]
        stock_pool = available_stock[item_name]
        
        for _, row in item_shortages.iterrows():
            if stock_pool <= 0:
                break
                
            centre = row["centre"]
            shortage = row["expected_shortage"]
            
            # Allocate up to the full expected shortage or remaining stock pool
            allocated_qty = min(shortage, stock_pool)
            stock_pool -= allocated_qty
            
            allocations.append({
                "from": "Partner Hospital A",
                "to": centre,
                "item": item_name,
                "expected_shortage": shortage,
                "allocated_quantity": allocated_qty,
                "remaining_shortage": shortage - allocated_qty,
                "risk_status": row["status"]
            })
            
    alloc_df = pd.DataFrame(allocations)
    return alloc_df, available_stock
