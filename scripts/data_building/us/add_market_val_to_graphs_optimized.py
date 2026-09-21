#!/usr/bin/env python3
"""
Add market_val attribute to existing graphs (memory-optimized).
"""
import pickle
import pandas as pd
import torch
import gc

GRAPH_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"
CSV_PATH = "/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/Final_data_v3_cleaned.csv"
OUTPUT_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"

print("="*80)
print("ADDING market_val TO EXISTING GRAPHS (MEMORY-OPTIMIZED)")
print("="*80)

# Load existing graphs
print("\n[1/3] Loading existing graphs...")
with open(GRAPH_PATH, 'rb') as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graph snapshots")
dates = sorted(graphs.keys())
print(f"Date range: {dates[0]} to {dates[-1]}")

# Process each date one at a time to save memory
print("\n[2/3] Adding market_val to each snapshot...")
etype = ('fund', 'holds_stock', 'stock')

for i, (date, g) in enumerate(sorted(graphs.items()), 1):
    print(f"  [{i}/{len(graphs)}] Processing {date}...", end='')
    
    if etype not in g.edge_types:
        print(" no edges, skipping")
        continue
    
    # Load ONLY this date's data from CSV
    date_str = str(date).split()[0]  # Convert to 'YYYY-MM-DD'
    
    # Read CSV in chunks and filter for this date
    date_df_list = []
    for chunk in pd.read_csv(CSV_PATH, chunksize=500000, parse_dates=['caldt'], low_memory=False):
        chunk_date = chunk[(chunk['caldt'] == date) & (chunk['security_type'] == 'Common stock')]
        if len(chunk_date) > 0:
            date_df_list.append(chunk_date)
    
    if not date_df_list:
        print(" no CSV data, filling with zeros")
        num_edges = g[etype].edge_index.size(1)
        g[etype].market_val = torch.zeros(num_edges, dtype=torch.float)
        continue
    
    date_df = pd.concat(date_df_list, ignore_index=True)
    date_df_dedup = date_df.drop_duplicates(['crsp_portno', 'permno'])
    
    # Get edge information
    edge_index = g[etype].edge_index
    num_edges = edge_index.size(1)
    
    # Get ID mappings
    fund_id_idx = g['fund'].id_idx.tolist()
    stock_id_idx = g['stock'].id_idx.tolist()
    
    fund_local_to_global = {i: gid for i, gid in enumerate(fund_id_idx)}
    stock_local_to_global = {i: gid for i, gid in enumerate(stock_id_idx)}
    
    # Create lookup
    market_val_lookup = {}
    for _, row in date_df_dedup.iterrows():
        key = (row['crsp_portno'], row['permno'])
        mv = row['market_val']
        market_val_lookup[key] = 0.0 if pd.isna(mv) else float(mv)
    
    # Build market_val tensor
    market_vals = []
    for j in range(num_edges):
        fund_local_id = int(edge_index[0, j])
        stock_local_id = int(edge_index[1, j])
        
        fund_global_id = fund_local_to_global[fund_local_id]
        stock_global_id = stock_local_to_global[stock_local_id]
        
        key = (fund_global_id, stock_global_id)
        mv = market_val_lookup.get(key, 0.0)
        market_vals.append(mv)
    
    g[etype].market_val = torch.tensor(market_vals, dtype=torch.float)
    
    n_nonzero = (g[etype].market_val > 0).sum().item()
    pct_nonzero = 100 * n_nonzero / num_edges
    print(f" ✓ {num_edges:,} edges, {n_nonzero:,} ({pct_nonzero:.1f}%) > 0")
    
    # Clean up
    del date_df, date_df_dedup, market_val_lookup, market_vals
    gc.collect()

print(f"\n✓ Processed all {len(graphs)} snapshots")

# Save
print(f"\n[3/3] Saving updated graphs...")
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(graphs, f)
print("✓ Saved successfully!")

# Verify
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

sample_date = dates[10]
g = graphs[sample_date]

print(f"\nSample snapshot: {sample_date}")
if etype in g.edge_types and hasattr(g[etype], 'market_val'):
    mv = g[etype].market_val
    print(f"✓ market_val present: {mv.shape}")
    print(f"  Min: ${mv.min().item():,.2f}")
    print(f"  Max: ${mv.max().item():,.2f}")
    print(f"  Mean: ${mv.mean().item():,.2f}")
    print(f"  Non-zero: {(mv > 0).sum().item():,} ({100*(mv > 0).sum().item()/len(mv):.1f}%)")
else:
    print("✗ market_val NOT found!")

print("="*80)
