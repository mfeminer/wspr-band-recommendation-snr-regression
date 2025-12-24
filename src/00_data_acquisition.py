import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"00_data_acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _parse_bands(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _make_cache_name(start_dt: datetime, end_dt: datetime) -> str:
    s = start_dt.strftime("%Y-%m-%d_%H-%M")
    e = end_dt.strftime("%Y-%m-%d_%H-%M")
    return f"spots_{s}_to_{e}.parquet"


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df["band"] = df["band"].astype("int16")
    df["distance"] = df["distance"].astype("int32")
    df["snr"] = df["snr"].astype("int8")
    df["tx_lat"] = df["tx_lat"].astype("float32")
    df["tx_lon"] = df["tx_lon"].astype("float32")
    df["rx_lat"] = df["rx_lat"].astype("float32")
    df["rx_lon"] = df["rx_lon"].astype("float32")
    return df


def fetch_spots(start: str, end: str, bands: List[int], limit: Optional[int] = None, max_retries: int = 5) -> pd.DataFrame:
    base_url = "https://db1.wspr.live/"
    band_list = ",".join(str(b) for b in bands)

    where_parts = [
        f"time >= toDateTime('{start}')",
        f"time < toDateTime('{end}')",
        f"band IN ({band_list})",
    ]
    where_sql = " AND ".join(where_parts)

    cols = [
        "time",
        "band",
        "distance",
        "snr",
        "tx_lat",
        "tx_lon",
        "rx_lat",
        "rx_lon",
    ]
    select_sql = f"SELECT {', '.join(cols)} FROM wspr.rx WHERE {where_sql}"
    if limit is not None:
        select_sql += f" LIMIT {int(limit)}"
    select_sql += " FORMAT JSON"

    url = base_url + "?query=" + requests.utils.quote(select_sql, safe="")
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            payload = r.json()
            df = pd.DataFrame(payload["data"])
            return _optimize_dtypes(df)
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"ERROR: {e}")
                print(f"  Retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"ERROR: {e}")
                print(f"  Retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    
    return pd.DataFrame()


def _download_batch(start_dt: datetime, end_dt: datetime, bands: List[int], 
                    cache_file: Path, limit: Optional[int], sleep_sec: float) -> Tuple[Path, int, str, float]:
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    start_time = time.time()
    
    try:
        df_batch = fetch_spots(start_str, end_str, bands, limit=limit)
        row_count = len(df_batch)
        
        if not df_batch.empty:
            df_batch["time"] = pd.to_datetime(df_batch["time"], utc=True)
            df_batch.to_parquet(cache_file, index=False)
        
        elapsed = time.time() - start_time
        time.sleep(sleep_sec)
        return cache_file, row_count, "", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        return cache_file, 0, str(e), elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start time (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--end", required=True, help="End time (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--bands", required=True, help="Comma-separated bands, e.g. 7,10,14,21")
    p.add_argument("--limit", type=int, default=None, help="Row limit per batch (optional)")
    p.add_argument("--out", required=True, help="Output file path (.parquet)")
    p.add_argument("--batch-hours", type=int, default=6, help="Batch size in hours (default: 6)")
    p.add_argument("--cache-dir", default="data/cache", help="Cache directory for batch files")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    p.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    args = p.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    bands = _parse_bands(args.bands)

    batch_delta = timedelta(hours=args.batch_hours)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batches_to_download = []
    cache_files_to_merge = []
    current = start_dt
    overwrite_all = None

    while current < end_dt:
        next_time = min(current + batch_delta, end_dt)
        cache_file = cache_dir / _make_cache_name(current, next_time)

        if cache_file.exists():
            if overwrite_all is None:
                ans = input(f"{cache_file.name} already exists. Overwrite? (y/n/all/none): ").strip().lower()
                if ans == "all":
                    overwrite_all = True
                    batches_to_download.append((current, next_time, cache_file))
                elif ans == "none":
                    overwrite_all = False
                    print(f"  → Using cached: {cache_file.name}")
                    cache_files_to_merge.append(cache_file)
                elif ans == "y":
                    batches_to_download.append((current, next_time, cache_file))
                else:
                    print(f"  → Using cached: {cache_file.name}")
                    cache_files_to_merge.append(cache_file)
            elif overwrite_all:
                batches_to_download.append((current, next_time, cache_file))
            else:
                print(f"  → Using cached: {cache_file.name}")
                cache_files_to_merge.append(cache_file)
        else:
            batches_to_download.append((current, next_time, cache_file))

        current = next_time

    if batches_to_download:
        total_batches = len(batches_to_download)
        print(f"\nDownloading {total_batches} batches with {args.workers} workers...")
        print(f"Starting parallel downloads... (each worker will process multiple batches)\n")
        
        completed_count = 0
        total_rows = 0
        start_time_overall = time.time()
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_download_batch, start_dt, end_dt, bands, cache_file, args.limit, args.delay): (start_dt, end_dt, cache_file)
                for start_dt, end_dt, cache_file in batches_to_download
            }
            
            for future in as_completed(futures):
                start_dt, end_dt, cache_file = futures[future]
                cache_file_result, row_count, error, elapsed = future.result()
                
                completed_count += 1
                progress_pct = (completed_count / total_batches) * 100
                
                start_str = start_dt.strftime("%Y-%m-%d %H:%M")
                end_str = end_dt.strftime("%Y-%m-%d %H:%M")
                
                elapsed_overall = time.time() - start_time_overall
                eta_seconds = (elapsed_overall / completed_count) * (total_batches - completed_count) if completed_count > 0 else 0
                eta_minutes = eta_seconds / 60
                avg_speed = total_rows / elapsed_overall if elapsed_overall > 0 else 0
                
                if error:
                    print(f"[{completed_count:3d}/{total_batches}] ✗ {start_str} → {end_str} | ERROR: {error}")
                else:
                    total_rows += row_count
                    print(f"[{completed_count:3d}/{total_batches}] ✓ {start_str} → {end_str} | {row_count:>8,} rows | {elapsed:4.1f}s | Progress: {progress_pct:5.1f}% | Total: {total_rows:>10,} | ETA: {eta_minutes:4.1f}m | Speed: {avg_speed:>8,.0f} rows/s", flush=True)
                    if row_count > 0:
                        cache_files_to_merge.append(cache_file_result)

    all_cache_files = sorted(set(cache_files_to_merge), key=lambda p: p.name)

    if not all_cache_files:
        print("No data to merge.")
        return

    print(f"\nMerging {len(all_cache_files)} cache files (streaming)...")
    
    writer = None
    total_rows = 0
    
    for i, cache_file in enumerate(all_cache_files):
        df_chunk = pd.read_parquet(cache_file)
        df_chunk = _optimize_dtypes(df_chunk)
        total_rows += len(df_chunk)
        
        table = pa.Table.from_pandas(df_chunk)
        
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        
        writer.write_table(table)
        del df_chunk, table
        
        if (i + 1) % 100 == 0:
            print(f"  Merged {i + 1}/{len(all_cache_files)} files...")
    
    if writer:
        writer.close()
    
    print(f"Wrote {total_rows:,} rows to {out_path}")


if __name__ == "__main__":
    main()

