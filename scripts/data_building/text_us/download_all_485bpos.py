"""
Download 485BPOS filings for all funds in the since_2010Q3 dataset.

1. Reads fund_mapping_since2010Q3.csv (built by build_fund_mapping.py)
2. Filters to portnos active at caldt >= 2010-09-30
3. Queries WRDS for crsp_fundno -> CIK
4. Saves CIK mapping to disk
5. Downloads 485BPOS from SEC EDGAR, skipping already-downloaded CIKs
"""
import pandas as pd
import wrds
import os
import time
import logging

# ========================= Config =========================
# SEC EDGAR asks every requester to declare a contact in the User-Agent.
USER_IDENTITY = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")

PREPROCESSED_CSV = (
    "/srv/scratch/dbgcse/jieliu/Dataset/CRSP_Dataset_v3/"
    "Final_data_v3_preprocessed.csv"
)
DOWNLOAD_FOLDER = (
    "/srv/scratch/dbgcse/jieliu/mutual_fund_prediction/"
    "sec_data/sec-edgar-filings"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_CSV = os.path.join(SCRIPT_DIR, "fund_mapping_since2010Q3.csv")
CIK_MAPPING_CSV = os.path.join(SCRIPT_DIR, "crsp_fundno_to_cik.csv")
LOG_FILE = os.path.join(SCRIPT_DIR, "download_485bpos.log")

START_DATE = "2010-01-01"
END_DATE = "2021-12-31"
MIN_CALDT = "2010-09-30"
CHUNK_SIZE = 1_000_000
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def get_fundnos_since_2010q3():
    """Get all crsp_fundno for portnos active since 2010Q3."""
    log.info("Getting fundnos for portnos active since %s...", MIN_CALDT)

    # Portnos active since 2010Q3
    active_portnos = set()
    for chunk in pd.read_csv(
        PREPROCESSED_CSV,
        usecols=["crsp_portno", "caldt"],
        dtype={"crsp_portno": int},
        chunksize=CHUNK_SIZE,
    ):
        sub = chunk[chunk["caldt"] >= MIN_CALDT]
        active_portnos.update(sub["crsp_portno"].unique())

    log.info("  Active portnos: %d", len(active_portnos))

    # Get fundnos from mapping
    mapping = pd.read_csv(MAPPING_CSV)
    filtered = mapping[mapping["crsp_portno"].isin(active_portnos)]
    fundnos = sorted(filtered["crsp_fundno"].dropna().astype(int).unique())
    log.info("  Fundnos: %d", len(fundnos))
    return fundnos


def get_ciks_from_wrds(fundnos):
    """Query WRDS crsp.crsp_cik_map for crsp_fundno -> comp_cik."""
    log.info("Connecting to WRDS for CIK mapping (%d fundnos)...", len(fundnos))

    db = wrds.Connection(wrds_username="jieliu1001")
    fund_list = tuple(fundnos)

    query = f"""
        SELECT DISTINCT crsp_fundno, comp_cik AS cik
        FROM crsp.crsp_cik_map
        WHERE crsp_fundno IN {fund_list}
        AND comp_cik IS NOT NULL
    """
    df = db.raw_sql(query)
    db.close()

    if df.empty:
        log.warning("No CIK mappings found!")
        return []

    df["cik"] = (
        pd.to_numeric(df["cik"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(10)
    )

    df.to_csv(CIK_MAPPING_CSV, index=False)
    log.info("  Saved CIK mapping to %s (%d rows)", CIK_MAPPING_CSV, len(df))

    unique_ciks = sorted(df["cik"].unique())
    log.info("  %d fundnos -> %d unique CIKs", df["crsp_fundno"].nunique(), len(unique_ciks))
    return unique_ciks


def download_filings(cik_list):
    """Download 485BPOS for each CIK, skipping already-downloaded ones."""
    from sec_edgar_downloader import Downloader

    existing = set()
    if os.path.isdir(DOWNLOAD_FOLDER):
        existing = set(os.listdir(DOWNLOAD_FOLDER))

    to_download = [c for c in cik_list if c not in existing]
    log.info(
        "CIKs: %d total, %d already downloaded, %d to download",
        len(cik_list), len(existing & set(cik_list)), len(to_download),
    )

    if not to_download:
        log.info("Nothing to download!")
        return

    dl = Downloader(USER_IDENTITY, DOWNLOAD_FOLDER)
    total = len(to_download)

    for i, cik in enumerate(to_download, 1):
        log.info("[%d/%d] CIK %s ...", i, total, cik)
        try:
            count = dl.get("485BPOS", cik, limit=None, after=START_DATE, before=END_DATE)
            log.info("  -> %s filings", count if count else "0")
        except Exception as e:
            log.error("  -> FAILED: %s", e)
        time.sleep(0.5)


def main():
    log.info("=" * 60)
    log.info("Download 485BPOS for since_2010Q3 dataset")
    log.info("=" * 60)

    # Reuse saved CIK mapping if it exists
    if os.path.exists(CIK_MAPPING_CSV):
        log.info("Loading existing CIK mapping from %s", CIK_MAPPING_CSV)
        df = pd.read_csv(CIK_MAPPING_CSV)
        df["cik"] = df["cik"].astype(str).str.zfill(10)
        cik_list = sorted(df["cik"].unique())
        log.info("  %d unique CIKs", len(cik_list))
    else:
        fundnos = get_fundnos_since_2010q3()
        cik_list = get_ciks_from_wrds(fundnos)
        if not cik_list:
            log.error("No CIKs. Exiting.")
            return

    download_filings(cik_list)

    log.info("=" * 60)
    log.info("DONE. Files in %s", DOWNLOAD_FOLDER)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
