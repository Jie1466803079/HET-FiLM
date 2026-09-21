"""
Build series_id -> fund_node_id (id_idx) mapping via WRDS CRSP.

Uses crsp.crsp_cik_map which contains:
  crsp_fundno, comp_cik, series_cik, contract_cik

Joins with fund_mapping_since2010Q3.csv (id_idx, crsp_portno, crsp_fundno)
to produce: series_cik -> id_idx -> crsp_portno

Usage:
    python build_series_mapping.py
"""

import os
import sys

import pandas as pd
import wrds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_CSV = os.path.join(SCRIPT_DIR, "fund_mapping_since2010Q3.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "series_id_to_fund_node.csv")

WRDS_USERNAME = "jieliu1001"


def main():
    # Load existing fund mapping
    print(f"Loading {MAPPING_CSV}...")
    fund_map = pd.read_csv(MAPPING_CSV)
    print(f"  {len(fund_map)} rows, {fund_map['id_idx'].nunique()} unique id_idx, "
          f"{fund_map['crsp_portno'].nunique()} unique crsp_portno, "
          f"{fund_map['crsp_fundno'].nunique()} unique crsp_fundno")

    all_fundnos = sorted(fund_map["crsp_fundno"].dropna().astype(int).unique())

    # Connect to WRDS and query crsp_cik_map for series_cik
    print(f"\nConnecting to WRDS as {WRDS_USERNAME}...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    print("Connected.")

    try:
        print(f"\nQuerying crsp.crsp_cik_map for {len(all_fundnos)} crsp_fundno values...")

        # Query in chunks to avoid SQL tuple-size limits
        chunk_size = 5000
        results = []
        for i in range(0, len(all_fundnos), chunk_size):
            chunk = all_fundnos[i:i + chunk_size]
            fund_tuple = tuple(chunk) if len(chunk) > 1 else f"({chunk[0]})"
            query = f"""
                SELECT DISTINCT crsp_fundno, series_cik
                FROM crsp.crsp_cik_map
                WHERE crsp_fundno IN {fund_tuple}
                  AND series_cik IS NOT NULL
            """
            df_chunk = db.raw_sql(query)
            results.append(df_chunk)
            print(f"  Chunk {i // chunk_size + 1}: {len(df_chunk)} rows")

    finally:
        db.close()
        print("WRDS connection closed.")

    if not results or all(df.empty for df in results):
        print("\nERROR: No series_cik mappings found!")
        sys.exit(1)

    cik_map = pd.concat(results, ignore_index=True).drop_duplicates()
    cik_map["crsp_fundno"] = cik_map["crsp_fundno"].astype(int)
    print(f"\nTotal from crsp_cik_map: {len(cik_map)} rows")
    print(f"  Unique crsp_fundno: {cik_map['crsp_fundno'].nunique()}")
    print(f"  Unique series_cik: {cik_map['series_cik'].nunique()}")
    print(f"\n  Sample:")
    print(cik_map.head(10).to_string(index=False))

    # Join with fund mapping: crsp_fundno -> (id_idx, crsp_portno)
    merged = cik_map.merge(fund_map, on="crsp_fundno", how="inner")
    print(f"\nAfter joining with fund_map: {len(merged)} rows")

    # Deduplicate to unique (series_cik, id_idx, crsp_portno) triples
    final = (
        merged[["series_cik", "id_idx", "crsp_portno"]]
        .drop_duplicates()
        .sort_values(["series_cik", "id_idx"])
        .reset_index(drop=True)
    )

    print(f"\nFinal mapping:")
    print(f"  {len(final)} rows")
    print(f"  Unique series_cik: {final['series_cik'].nunique()}")
    print(f"  Unique id_idx: {final['id_idx'].nunique()}")
    print(f"  Unique crsp_portno: {final['crsp_portno'].nunique()}")

    total_idx = fund_map["id_idx"].nunique()
    covered_idx = final["id_idx"].nunique()
    print(f"\n  Coverage: {covered_idx}/{total_idx} id_idx values "
          f"({100 * covered_idx / total_idx:.1f}%)")

    # Save
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"\n  Sample:")
    print(final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
