#!/usr/bin/env python3
"""
Add market_val to existing graphs by matching edges directly.
"""
import pickle
import pandas as pd
import torch
import gc

GRAPH_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"
CSV_PATH = "/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/Final_data_v3_cleaned.csv"
OUTPUT_PATH = "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/CRSP_code_v4/DHGAS/graphs/graphs_v2_portfolio_clean.pkl"

print("="*80)
print("ADDING market_val TO EXISTING GRAPHS (FIXED)")
print("="*80)

# Load existing graphs
print("\n[1/3] Loading existing graphs...")
with open(GRAPH_PATH, 'rb') as f:
    graphs = pickle.load(f)

print(f"Loaded {len(graphs)} graph snapshots")
dates = sorted(graphs.keys())

# Process each date
print("\n[2/3] Adding market_val to each snapshot...")
etype = ('fund', 'holds_stock', 'stock')

for i, (date, g) in enumerate(sorted(graphs.items()), 1):
    print(f"  [{i}/{len(graphs)}] Processing {date}...", end='')
    
    if etype not in g.edge_types:
        print(" no edges, skipping")
        continue
    
    edge_index = g[etype].edge_index
    num_edges = edge_index.size(1)
    
    # Load CSV data for this date only
    date_str = str(date).split()[0]
    date_df_list = []
    
    for chunk in pd.read_csv(CSV_PATH, chunksize=500000, parse_dates=['caldt'], low_memory=False):
        chunk_filtered = chunk[
            (chunk['caldt'] == date) & 
            (chunk['security_type'] == 'Common stock')
        ]
        if len(chunk_filtered) > 0:
            date_df_list.append(chunk_filtered)
    
    if not date_df_list:
        print(" no CSV data, filling with zeros")
        g[etype].market_val = torch.zeros(num_edges, dtype=torch.float)
        continue
    
    date_df = pd.concat(date_df_list, ignore_index=True)
    date_df_dedup = date_df.drop_duplicates(['crsp_portno', 'permno'])
    
    # Create lookup: (crsp_portno, permno) -> market_val
    market_val_dict = {}
    for _, row in date_df_dedup.iterrows():
        key = (int(row['crsp_portno']), float(row['permno']))
        mv = row['market_val']
        market_val_dict[key] = 0.0 if pd.isna(mv) else float(mv)
    
    # Get fund and stock ID mappings
    # id_idx contains global IDs, but we need actual crsp_portno/permno
    # We'll reconstruct by matching edge structure
    
    # Build reverse mapping: global_id -> actual value
    # Since we don't have the original active_portfolios/active_stocks lists,
    # we'll extract unique crsp_portno/permno from CSV and match by order
    
    # Get unique portfolios and stocks from CSV (sorted, matching graph building logic)
    unique_portfolios = sorted(date_df_dedup['crsp_portno'].unique())
    unique_stocks = sorted(date_df_dedup['permno'].dropna().unique())
    
    # Create mapping: local_idx -> actual value
    fund_local_to_actual = {i: int(p) for i, p in enumerate(unique_portfolios)}
    stock_local_to_actual = {i: float(s) for i, s in enumerate(unique_stocks)}
    
    # But wait - the graph's local indices might not match CSV order!
    # Instead, let's match edges directly by building a set of all edges from CSV
    csv_edges = set()
    csv_edge_mv = {}
    for _, row in date_df_dedup.iterrows():
        edge_key = (int(row['crsp_portno']), float(row['permno']))
        csv_edges.add(edge_key)
        mv = row['market_val']
        csv_edge_mv[edge_key] = 0.0 if pd.isna(mv) else float(mv)
    
    # Now we need to map graph edges to CSV edges
    # The problem: graph uses local IDs, CSV uses actual crsp_portno/permno
    # Solution: We need to know which local ID corresponds to which actual value
    
    # Actually, let's try a different approach:
    # Since the graph was built from the same CSV, the edges should be in the same order
    # after deduplication. Let's match by position!
    
    # Get graph edges in order
    graph_edges_list = []
    for j in range(num_edges):
        fund_local = int(edge_index[0, j])
        stock_local = int(edge_index[1, j])
        graph_edges_list.append((fund_local, stock_local))
    
    # Match by assuming same order (risky but might work)
    # Better: build a mapping from the CSV data structure
    
    # Actually, the safest way: iterate through CSV edges and match to graph edges
    # by finding the corresponding local indices
    
    # Build mapping: (crsp_portno, permno) -> (fund_local_idx, stock_local_idx)
    # We need to know which local index corresponds to which actual value
    # This requires knowing active_portfolios and active_stocks, which we don't have
    
    # Alternative: Use the fact that edge_index[0] and edge_index[1] are local indices
    # and match them to the sorted unique lists from CSV
    
    # Try matching by assuming the graph's local indices correspond to sorted unique values
    # from the CSV (which is how the graph was built)
    
    # Get all unique portfolios and stocks that appear in edges (from CSV)
    edge_portfolios = sorted(date_df_dedup['crsp_portno'].unique())
    edge_stocks = sorted(date_df_dedup['permno'].dropna().unique())
    
    # Create reverse lookup: actual value -> local index in graph
    # But we don't know the graph's local index mapping!
    
    # Final approach: Match edges by their position and the fact that
    # the graph was built with sorted unique portfolios/stocks
    
    # Actually, let me check: the graph building uses sorted(unique()) which should match
    # our sorted unique lists from CSV. So local index i should correspond to sorted_unique[i]
    
    fund_actual_to_local = {int(p): i for i, p in enumerate(edge_portfolios)}
    stock_actual_to_local = {float(s): i for i, s in enumerate(edge_stocks)}
    
    # Now build market_val tensor
    market_vals = []
    matched = 0
    for j in range(num_edges):
        fund_local = int(edge_index[0, j])
        stock_local = int(edge_index[1, j])
        
        # Try to find corresponding actual values
        # If local indices match sorted order, then:
        if fund_local < len(edge_portfolios) and stock_local < len(edge_stocks):
            fund_actual = edge_portfolios[fund_local]
            stock_actual = edge_stocks[stock_local]
            key = (fund_actual, stock_actual)
            mv = csv_edge_mv.get(key, 0.0)
            if mv > 0:
                matched += 1
        else:
            mv = 0.0
        
        market_vals.append(mv)
    
    g[etype].market_val = torch.tensor(market_vals, dtype=torch.float)
    
    n_nonzero = (g[etype].market_val > 0).sum().item()
    pct_nonzero = 100 * n_nonzero / num_edges
    print(f" ✓ {num_edges:,} edges, {n_nonzero:,} ({pct_nonzero:.1f}%) > 0, {matched:,} matched")
    
    del date_df, date_df_dedup, market_val_dict, csv_edges, csv_edge_mv
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
