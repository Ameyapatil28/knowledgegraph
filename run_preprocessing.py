"""Script to run TMDB 5000 preprocessing pipeline."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    MOVIES_RAW_PATH,
    CREDITS_RAW_PATH,
    PROCESSED_MOVIES_PARQUET,
    PROCESSED_MOVIES_JSONL,
)
from src.preprocessing import preprocess_datasets, save_processed_dataset


def main():
    print("=" * 60)
    print("Starting TMDB 5000 Dataset Preprocessing Pipeline")
    print("=" * 60)

    start_time = time.time()

    # Verify input files exist
    if not MOVIES_RAW_PATH.exists():
        print(f"Error: Movies raw file not found at {MOVIES_RAW_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CREDITS_RAW_PATH.exists():
        print(f"Error: Credits raw file not found at {CREDITS_RAW_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Raw Movies file:  {MOVIES_RAW_PATH}")
    print(f"Raw Credits file: {CREDITS_RAW_PATH}")

    # Run preprocessing
    processed_df = preprocess_datasets(
        movies_path=MOVIES_RAW_PATH,
        credits_path=CREDITS_RAW_PATH,
    )

    # Save to disk
    save_processed_dataset(
        df=processed_df,
        parquet_path=PROCESSED_MOVIES_PARQUET,
        jsonl_path=PROCESSED_MOVIES_JSONL,
    )

    elapsed = time.time() - start_time

    # Display dataset summary metrics
    print("\n" + "=" * 60)
    print("Preprocessing Pipeline Finished Successfully!")
    print("=" * 60)
    print(f"Execution Time:          {elapsed:.2f} seconds")
    print(f"Total Movies Processed:  {len(processed_df):,}")
    print(f"Movies with valid Budget:{processed_df['has_valid_budget'].sum():,} "
          f"({processed_df['budget'].isna().sum():,} unrecorded)")
    print(f"Movies with valid Revenue:{processed_df['has_valid_revenue'].sum():,} "
          f"({processed_df['revenue'].isna().sum():,} unrecorded)")
    print(f"Duplicate Titles Count:  {processed_df['is_duplicate_title'].sum():,}")

    dup_sample = processed_df[processed_df["is_duplicate_title"]][["tmdb_id", "title", "disambiguated_title", "release_year"]]
    print("\nSample Disambiguated Duplicate Titles:")
    print(dup_sample.to_string(index=False))

    parquet_size_mb = PROCESSED_MOVIES_PARQUET.stat().st_size / (1024 * 1024)
    jsonl_size_mb = PROCESSED_MOVIES_JSONL.stat().st_size / (1024 * 1024)
    print(f"\nSaved Files:")
    print(f"  - Parquet: {PROCESSED_MOVIES_PARQUET} ({parquet_size_mb:.2f} MB)")
    print(f"  - JSONL:   {PROCESSED_MOVIES_JSONL} ({jsonl_size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
