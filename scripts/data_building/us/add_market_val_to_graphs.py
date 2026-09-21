#!/usr/bin/env python3
"""
Add market_val attribute to existing graphs without full rebuild.
"""
import pickle
import pandas as pd
import torch
from datetime import datetime

GRAPH_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"
CSV_PATH = "/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/Final_data_v3_cleaned.csv"
OUTPUT_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"

print("="*80)
print("ADDING market_val TO EXISTING GRAPHS")
print("="*80)

# Load existing graphs
print("\n[1/4] Loading existing graphs...")
with open(GRAPH_PATH, 'rb') as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graph snapshots")
dates = sorted(graphs.keys())
print(f"Date range: {dates[0]} to {dates[-1]}")

# Load CSV data
print("\n[2/4] Loading CRSP data...")
df = pd.read_csv(CSV_PATH, parse_dates=['caldt'], low_memory=False)
df_stocks = df[df['security_type'] == 'Common stock'].copy()
print(f"Loaded {len(df_stocks):,} stock holdings")

# Create date-indexed lookup for faster access
print("\n[3/4] Creating date-indexed lookup...")
df_stocks_by_date = {date: group for date, group in df_stocks.groupby('caldt')}
print(f"Indexed {len(df_stocks_by_date)} unique dates")

# Add market_val to each graph
print("\n[4/4] Adding market_val to graph edges...")
etype = ('fund', 'holds_stock', 'stock')

for i, (date, g) in enumerate(sorted(graphs.items()), 1):
    print(f"  Processing {date} ({i}/{len(graphs)})...", end='\r')
    
    if etype not in g.edge_types:
        continue
    
    # Get corresponding CSV data
    if date not in df_stocks_by_date:
        print(f"\n  ⚠️ No CSV data for {date}, skipping...")
        continue
    
    date_df = df_stocks_by_date[date]
    
    # Deduplicate to portfolio level (as in graph building)
    date_df_dedup = date_df.drop_duplicates(['crsp_portno', 'permno'])
    
    # Get edge information from graph
    edge_index = g[etype].edge_index
    num_edges = edge_index.size(1)
    
    # Get fund and stock ID mappings
    fund_id_idx = g['fund'].id_idx.tolist()
    stock_id_idx = g['stock'].id_idx.tolist()
    
    # Create reverse mappings (local_id -> global_id)
    fund_local_to_global = {i: gid for i, gid in enumerate(fund_id_idx)}
    stock_local_to_global = {i: gid for i, gid in enumerate(stock_id_idx)}
    
    # Create lookup: (crsp_portno, permno) -> market_val
    market_val_lookup = {}
    for _, row in date_df_dedup.iterrows():
        key = (row['crsp_portno'], row['permno'])
        mv = row['market_val']
        market_val_lookup[key] = 0.0 if pd.isna(mv) else float(mv)
    
    # Build market_val tensor for edges
    market_vals = []
    for j in range(num_edges):
        fund_local_id = int(edge_index[0, j])
        stock_local_id = int(edge_index[1, j])
        
        fund_global_id = fund_local_to_global[fund_local_id]
        stock_global_id = stock_local_to_global[stock_local_id]
        
        # Look up market_val
        key = (fund_global_id, stock_global_id)
        mv = market_val_lookup.get(key, 0.0)
        market_vals.append(mv)
    
    # Add to graph
    g[etype].market_val = torch.tensor(market_vals, dtype=torch.float)
    
    # Verify
    if j % 10 == 0:  # Check every 10th snapshot
        n_nonzero = (g[etype].market_val > 0).sum().item()
        pct_nonzero = 100 * n_nonzero / num_edges
        print(f"\n  ✓ {date}: {num_edges:,} edges, {n_nonzero:,} ({pct_nonzero:.1f}%) with market_val > 0")

print(f"\n\n✓ Processed all {len(graphs)} snapshots")

# Save updated graphs
print(f"\nSaving updated graphs to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(graphs, f)

print("✓ Saved successfully!")

# Verify one snapshot
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

sample_date = dates[10]
g = graphs[sample_date]

print(f"\nSample snapshot: {sample_date}")
if etype in g.edge_types:
    print(f"Edges: {g[etype].edge_index.size(1):,}")
    if hasattr(g[etype], 'market_val'):
        mv = g[etype].market_val
        print(f"\n✓ market_val attribute present!")
        print(f"  Shape: {mv.shape}")
        print(f"  Min: ${mv.min().item():,.2f}")
        print(f"  Max: ${mv.max().item():,.2f}")
        print(f"  Mean: ${mv.mean().item():,.2f}")
        print(f"  Median: ${mv.median().item():,.2f}")
        print(f"  Non-zero: {(mv > 0).sum().item():,} ({100*(mv > 0).sum().item()/len(mv):.1f}%)")
    else:
        print("✗ market_val attribute NOT found!")

print("="*80)
print("COMPLETE!")
print("="*80)
