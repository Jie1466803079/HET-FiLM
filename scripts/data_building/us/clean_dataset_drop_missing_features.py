#!/usr/bin/env python3
"""
Clean CRSP Dataset v3 by dropping records with missing stock features
Removes 444 records (0.11% of common stock holdings) that lack stock features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

DATA_DIR = Path("/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/")
INPUT_FILE = DATA_DIR / "Final_data_v3.csv"
OUTPUT_FILE = DATA_DIR / "Final_data_v3_cleaned.csv"
BACKUP_DIR = DATA_DIR / "backups"

STOCK_FEATURE_COLS = [
    'r2_1', 'r12_2', 'r12_7', 'ST_Rev', 'LT_Rev', 'NI_yj',
    'Beta', 'IdioVol_yj', 'LME', 'LTurnover', 'Rel2High_yj',
    'Resid_Var_yj', 'spread_q_yj', 'SUV', 'Variance_yj'
]

def create_backup():
    """Create backup directory if needed"""
    BACKUP_DIR.mkdir(exist_ok=True)
    print(f"Backup directory: {BACKUP_DIR}")

def clean_data_chunked():
    """Clean data in chunks to handle large file"""
    print("=" * 80)
    print("CLEANING CRSP DATASET v3")
    print("=" * 80)
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    chunk_size = 500000
    total_records = 0
    total_dropped = 0
    total_kept = 0

    # Statistics tracking
    dropped_by_type = {}
    fund_qtrs_before = set()
    fund_qtrs_after = set()

    print(f"\nProcessing in chunks of {chunk_size:,} records...")

    # First pass: count total records
    print("\nPass 1: Counting records...")
    for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunk_size, low_memory=False)):
        total_records += len(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Counted {total_records:,} records...")

    print(f"Total records in source: {total_records:,}")

    # Second pass: filter and write
    print("\nPass 2: Filtering and writing cleaned data...")

    first_chunk = True
    chunks_processed = 0

    for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size, low_memory=False):
        chunks_processed += 1

        # Track fund-quarters before filtering
        chunk['caldt'] = pd.to_datetime(chunk['caldt'])
        chunk['fund_qtr'] = chunk['crsp_fundno'].astype(str) + '_' + chunk['caldt'].dt.to_period('Q').astype(str)
        fund_qtrs_before.update(chunk['fund_qtr'].unique())

        # Identify records to drop
        # Drop common stocks with missing stock features
        is_common_stock = chunk['security_type'] == 'Common stock'
        has_missing_features = chunk[STOCK_FEATURE_COLS].isna().any(axis=1)

        to_drop = is_common_stock & has_missing_features
        to_keep = ~to_drop

        # Statistics
        chunk_dropped = to_drop.sum()
        chunk_kept = to_keep.sum()

        total_dropped += chunk_dropped
        total_kept += chunk_kept

        # Track what we're dropping
        for sec_type in chunk[to_drop]['security_type'].unique():
            dropped_by_type[sec_type] = dropped_by_type.get(sec_type, 0) + (chunk[to_drop]['security_type'] == sec_type).sum()

        # Keep only clean records
        chunk_clean = chunk[to_keep].copy()

        # Track fund-quarters after filtering
        fund_qtrs_after.update(chunk_clean['fund_qtr'].unique())

        # Drop temporary column
        chunk_clean = chunk_clean.drop(columns=['fund_qtr'])

        # Write to output
        if first_chunk:
            chunk_clean.to_csv(OUTPUT_FILE, index=False, mode='w')
            first_chunk = False
        else:
            chunk_clean.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        # Progress update
        print(f"  Chunk {chunks_processed}: Kept {chunk_kept:,} / {len(chunk):,} records "
              f"(dropped {chunk_dropped:,})")

    return total_records, total_dropped, total_kept, dropped_by_type, fund_qtrs_before, fund_qtrs_after

def generate_report(total_records, total_dropped, total_kept, dropped_by_type, fund_qtrs_before, fund_qtrs_after):
    """Generate cleaning report"""
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)

    print(f"\n--- Overall Statistics ---")
    print(f"Total records processed: {total_records:,}")
    print(f"Records dropped: {total_dropped:,} ({total_dropped/total_records*100:.3f}%)")
    print(f"Records kept: {total_kept:,} ({total_kept/total_records*100:.3f}%)")

    print(f"\n--- Dropped Records by Security Type ---")
    for sec_type, count in sorted(dropped_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sec_type}: {count:,}")

    print(f"\n--- Fund-Quarter Coverage ---")
    print(f"Fund-quarters before cleaning: {len(fund_qtrs_before):,}")
    print(f"Fund-quarters after cleaning: {len(fund_qtrs_after):,}")
    print(f"Fund-quarters lost: {len(fund_qtrs_before) - len(fund_qtrs_after):,}")

    if len(fund_qtrs_before) == len(fund_qtrs_after):
        print("✅ SUCCESS: No fund-quarters lost all holdings!")
    else:
        lost_qtrs = fund_qtrs_before - fund_qtrs_after
        print(f"⚠️  WARNING: {len(lost_qtrs)} fund-quarters lost all holdings")
        print(f"Sample lost fund-quarters: {list(lost_qtrs)[:5]}")

    print(f"\n--- Output ---")
    print(f"Cleaned data saved to: {OUTPUT_FILE}")

    # Create log file
    log_file = DATA_DIR / f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write("CRSP Dataset v3 Cleaning Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Input: {INPUT_FILE}\n")
        f.write(f"Output: {OUTPUT_FILE}\n\n")
        f.write(f"Total records: {total_records:,}\n")
        f.write(f"Dropped: {total_dropped:,} ({total_dropped/total_records*100:.3f}%)\n")
        f.write(f"Kept: {total_kept:,} ({total_kept/total_records*100:.3f}%)\n\n")
        f.write("Dropped by type:\n")
        for sec_type, count in sorted(dropped_by_type.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {sec_type}: {count:,}\n")
        f.write(f"\nFund-quarters before: {len(fund_qtrs_before):,}\n")
        f.write(f"Fund-quarters after: {len(fund_qtrs_after):,}\n")
        f.write(f"Fund-quarters lost: {len(fund_qtrs_before) - len(fund_qtrs_after):,}\n")

    print(f"\nLog file saved to: {log_file}")

def verify_cleaned_data():
    """Verify the cleaned dataset"""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    print("\nReading cleaned data sample...")
    df_clean = pd.read_csv(OUTPUT_FILE, nrows=100000, low_memory=False)

    # Check for missing stock features in common stocks
    common_stocks = df_clean[df_clean['security_type'] == 'Common stock']
    has_missing = common_stocks[STOCK_FEATURE_COLS].isna().any(axis=1).sum()

    print(f"\n--- Verification Results ---")
    print(f"Sample size: {len(df_clean):,}")
    print(f"Common stocks in sample: {len(common_stocks):,}")
    print(f"Common stocks with missing features: {has_missing}")

    if has_missing == 0:
        print("✅ VERIFIED: No common stocks with missing features!")
    else:
        print(f"⚠️  WARNING: Found {has_missing} common stocks with missing features")

    # Check data integrity
    print(f"\n--- Data Integrity Checks ---")
    print(f"Total records in cleaned file (sample): {len(df_clean):,}")
    print(f"Columns: {len(df_clean.columns)}")
    print(f"Missing target values: {df_clean['target_abn_next_w'].isna().sum()}")

    # Check unique values
    print(f"\n--- Coverage Statistics ---")
    df_clean['caldt'] = pd.to_datetime(df_clean['caldt'])
    print(f"Date range: {df_clean['caldt'].min()} to {df_clean['caldt'].max()}")
    print(f"Unique funds: {df_clean['crsp_fundno'].nunique():,}")
    print(f"Unique stocks (permno): {df_clean['permno'].nunique():,}")

def main():
    """Main cleaning process"""
    print("\n" + "=" * 80)
    print("CRSP DATASET v3 CLEANING SCRIPT")
    print("Dropping records with missing stock features")
    print("=" * 80)

    try:
        # Create backup directory
        create_backup()

        # Clean data
        total_records, total_dropped, total_kept, dropped_by_type, fund_qtrs_before, fund_qtrs_after = clean_data_chunked()

        # Generate report
        generate_report(total_records, total_dropped, total_kept, dropped_by_type, fund_qtrs_before, fund_qtrs_after)

        # Verify cleaned data
        verify_cleaned_data()

        print("\n" + "=" * 80)
        print("CLEANING COMPLETE")
        print("=" * 80)
        print(f"\n✅ Cleaned dataset saved to: {OUTPUT_FILE}")
        print(f"✅ Ready for graph construction")

        return 0

    except Exception as e:
        print(f"\n❌ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
