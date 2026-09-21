"""
Build the correct id_idx -> crsp_portno -> crsp_fundno mapping for
graphs_v2_portfolio_clean_with_ids_since_2010Q3.pkl.

The graph was built from Final_data_v3_preprocessed.csv using setdefault
per chronological snapshot:
    fund_global_id_map = {}
    for date in sorted(dates):
        for portno in sorted(active_portnos_at_date):
            fund_global_id_map.setdefault(portno, len(fund_global_id_map))

This script replays that logic, then verifies against the actual graph
by checking that portfolio-level features (num_funds_w, turn_ratio_w,
family_age_w) in the graph match the CSV row for the assigned portno
at the same date.
"""
import pandas as pd
import pickle
import os
import sys

PREPROCESSED_CSV = (
    "/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/"
    "Final_data_v3_preprocessed.csv"
)
GRAPH_PKL = (
    "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
    "CRSP_code_v4/DHGAS/graphs/"
    "graphs_v2_portfolio_clean_with_ids_since_2010Q3.pkl"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "fund_mapping_since2010Q3.csv")
CHUNK_SIZE = 1_000_000


def replay_setdefault():
    """Replay setdefault on preprocessed CSV to get id_idx -> crsp_portno."""
    print("Step 1: Replaying setdefault on preprocessed CSV...")

    date_portnos = {}
    for chunk in pd.read_csv(
        PREPROCESSED_CSV,
        usecols=["crsp_portno", "caldt"],
        dtype={"crsp_portno": int},
        chunksize=CHUNK_SIZE,
    ):
        for caldt, grp in chunk.groupby("caldt"):
            if caldt not in date_portnos:
                date_portnos[caldt] = set()
            date_portnos[caldt].update(grp["crsp_portno"].unique())

    fund_map = {}
    for d in sorted(date_portnos.keys()):
        for p in sorted(date_portnos[d]):
            fund_map.setdefault(p, len(fund_map))

    idx_to_portno = {v: k for k, v in fund_map.items()}
    print(f"  {len(fund_map)} funds mapped (id_idx 0-{max(idx_to_portno)})")
    return idx_to_portno


def collect_fundnos():
    """Collect crsp_portno -> list of crsp_fundno from preprocessed CSV."""
    print("\nStep 2: Collecting crsp_fundno per portno...")
    portno_to_fundnos = {}
    for chunk in pd.read_csv(
        PREPROCESSED_CSV,
        usecols=["crsp_portno", "crsp_fundno"],
        dtype={"crsp_portno": int, "crsp_fundno": int},
        chunksize=CHUNK_SIZE,
    ):
        for portno, grp in chunk.groupby("crsp_portno"):
            fns = set(grp["crsp_fundno"].unique())
            if portno in portno_to_fundnos:
                portno_to_fundnos[portno].update(fns)
            else:
                portno_to_fundnos[portno] = fns

    portno_to_fundnos = {k: sorted(v) for k, v in portno_to_fundnos.items()}
    print(f"  {len(portno_to_fundnos)} portnos with fundno data")
    return portno_to_fundnos


def verify_against_graph(idx_to_portno):
    """
    Verify mapping by sampling id_idx values from the graph, finding
    the portno our replay assigned, and checking that portfolio-level
    features in the graph match the CSV row for that portno at the
    same date.
    """
    print("\nStep 3: Verifying against graph features...")
    with open(GRAPH_PKL, "rb") as f:
        graphs = pickle.load(f)

    dates = sorted(graphs.keys())
    all_graph_ids = set()
    for g in graphs.values():
        all_graph_ids.update(g["fund"]["id_idx"].tolist())

    print(f"  Graph: {len(all_graph_ids)} unique id_idx ({min(all_graph_ids)}-{max(all_graph_ids)})")

    # Sample every 100th id_idx, cap at 25
    test_ids = sorted(all_graph_ids)[::100][:25]

    # For each sampled id_idx, find its first graph snapshot and record features
    test_cases = {}
    for tid in test_ids:
        portno = idx_to_portno.get(tid)
        if portno is None:
            continue
        for d in dates:
            fids = graphs[d]["fund"]["id_idx"].tolist()
            if tid in fids:
                lp = fids.index(tid)
                gf = graphs[d]["fund"]["x"][lp].tolist()
                test_cases[portno] = (tid, str(d)[:10], gf)
                break

    del graphs

    # Single pass through CSV to check features
    print(f"  Checking {len(test_cases)} sample funds...")
    needed = set(test_cases.keys())
    verified = 0
    mismatches = 0

    for chunk in pd.read_csv(
        PREPROCESSED_CSV,
        usecols=["crsp_portno", "caldt", "num_funds_w", "turn_ratio_w", "family_age_w"],
        dtype={"crsp_portno": int},
        chunksize=CHUNK_SIZE,
    ):
        for portno in list(needed):
            tid, date_str, gf = test_cases[portno]
            sub = chunk[(chunk["caldt"] == date_str) & (chunk["crsp_portno"] == portno)]
            if sub.empty:
                continue

            r = sub.iloc[0]
            nf_ok = abs(gf[4] - r["num_funds_w"]) < 0.5
            tr_ok = abs(gf[5] - r["turn_ratio_w"]) < 0.005
            fa_ok = abs(gf[6] - r["family_age_w"]) < 0.005

            if nf_ok and tr_ok and fa_ok:
                verified += 1
            else:
                mismatches += 1
                print(
                    f"    MISMATCH id={tid} portno={portno} at {date_str}: "
                    f"graph(nf={gf[4]:.0f},tr={gf[5]:.2f},fa={gf[6]:.4f}) vs "
                    f"csv(nf={r['num_funds_w']:.0f},tr={r['turn_ratio_w']:.2f},fa={r['family_age_w']:.4f})"
                )
            needed.discard(portno)

        if not needed:
            break

    not_found = len(needed)
    print(f"  Verified: {verified}/{len(test_cases)}, mismatches: {mismatches}, not found in CSV: {not_found}")

    if mismatches > 0 or not_found > 0:
        print("  WARNING: some verifications failed!")
        return False
    print("  All samples verified OK!")
    return True


def main():
    idx_to_portno = replay_setdefault()
    portno_to_fundnos = collect_fundnos()
    ok = verify_against_graph(idx_to_portno)
    if not ok:
        print("\nMapping verification failed. Aborting.")
        sys.exit(1)

    # Build and save
    print("\nStep 4: Saving mapping...")
    rows = []
    for idx in sorted(idx_to_portno.keys()):
        portno = idx_to_portno[idx]
        fundnos = portno_to_fundnos.get(portno, [])
        if not fundnos:
            rows.append({"id_idx": idx, "crsp_portno": portno, "crsp_fundno": None})
        else:
            for fno in fundnos:
                rows.append({"id_idx": idx, "crsp_portno": portno, "crsp_fundno": int(fno)})

    df = pd.DataFrame(rows)
    for col in ["id_idx", "crsp_portno"]:
        df[col] = df[col].astype(int)
    df["crsp_fundno"] = df["crsp_fundno"].astype("Int64")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved to {OUTPUT_CSV}")
    print(f"  Total id_idx: {df['id_idx'].nunique()}")
    print(f"  Total rows: {len(df)}")
    print(f"  id_idx range: {df['id_idx'].min()}-{df['id_idx'].max()}")


if __name__ == "__main__":
    main()
