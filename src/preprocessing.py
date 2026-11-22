"""TMDB 5000 Dataset Validation and Preprocessing Pipeline."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# JSON columns present in the raw TMDB datasets
JSON_COLUMNS_MOVIES = [
    "genres",
    "keywords",
    "production_companies",
    "production_countries",
    "spoken_languages",
]
JSON_COLUMNS_CREDITS = ["cast", "crew"]
ALL_JSON_COLUMNS = JSON_COLUMNS_MOVIES + JSON_COLUMNS_CREDITS


def parse_json_field(val: Any) -> list:
    """Safely parse a JSON-encoded string column into a Python list/dict.

    Returns an empty list for NaN, None, or parsing errors.
    """
    if val is None or pd.isna(val):
        return []
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str in ("[]", "{}"):
            return []
        try:
            return json.loads(val_str)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Failed to parse JSON string: {val_str[:50]}... Error: {exc}")
            return []
    return []


def validate_keys(movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate primary and foreign keys between movies and credits.

    Ensures uniqueness of 'id' in movies and 'movie_id' in credits, and checks
    for any missing or orphaned records.
    """
    if "id" not in movies_df.columns:
        raise KeyError("movies_df is missing required primary key column 'id'")
    if "movie_id" not in credits_df.columns:
        raise KeyError("credits_df is missing required foreign key column 'movie_id'")

    movies_count = len(movies_df)
    credits_count = len(credits_df)

    movies_unique_ids = movies_df["id"].nunique()
    credits_unique_ids = credits_df["movie_id"].nunique()

    if movies_unique_ids != movies_count:
        dup_ids = movies_df[movies_df["id"].duplicated()]["id"].tolist()
        raise ValueError(f"Duplicate primary key 'id' in movies_df: {dup_ids}")

    if credits_unique_ids != credits_count:
        dup_credits = credits_df[credits_df["movie_id"].duplicated()]["movie_id"].tolist()
        raise ValueError(f"Duplicate foreign key 'movie_id' in credits_df: {dup_credits}")

    movies_id_set = set(movies_df["id"])
    credits_id_set = set(credits_df["movie_id"])

    missing_in_credits = movies_id_set - credits_id_set
    missing_in_movies = credits_id_set - movies_id_set

    if missing_in_credits or missing_in_movies:
        raise ValueError(
            f"Key mismatch between movies and credits: "
            f"{len(missing_in_credits)} movies missing in credits, "
            f"{len(missing_in_movies)} credits missing in movies."
        )

    logger.info(
        f"Key validation passed: {movies_count} movies match exactly {credits_count} credits records."
    )
    return {
        "movies_count": movies_count,
        "credits_count": credits_count,
        "keys_match": True,
        "orphan_count": 0,
    }


def extract_directors(crew_list: list) -> List[Dict[str, Any]]:
    """Extract list of directors with id and name from parsed crew items."""
    directors = []
    seen = set()
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            director_id = member.get("id")
            if director_id and director_id not in seen:
                seen.add(director_id)
                directors.append({
                    "id": director_id,
                    "name": member.get("name", "").strip(),
                })
    return directors


def extract_crew_by_roles(crew_list: list, roles: List[str]) -> List[Dict[str, Any]]:
    """Extract crew members matching specific job roles (e.g. Producer, Screenplay)."""
    matches = []
    seen = set()
    role_set = set(roles)
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") in role_set:
            person_id = member.get("id")
            key = (person_id, member.get("job"))
            if key not in seen:
                seen.add(key)
                matches.append({
                    "id": person_id,
                    "name": member.get("name", "").strip(),
                    "job": member.get("job"),
                    "department": member.get("department", ""),
                })
    return matches


def preprocess_datasets(
    movies_path: str | Path,
    credits_path: str | Path,
) -> pd.DataFrame:
    """Load, validate, clean, and enrich the TMDB 5000 movies and credits datasets."""
    logger.info(f"Loading raw datasets:\n  Movies: {movies_path}\n  Credits: {credits_path}")

    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    # 1. Validate primary and foreign keys
    validate_keys(movies_df, credits_df)

    # 2. Parse JSON columns in movies_df
    logger.info("Parsing JSON columns in movies dataset...")
    for col in JSON_COLUMNS_MOVIES:
        if col in movies_df.columns:
            movies_df[col] = movies_df[col].apply(parse_json_field)

    # 3. Parse JSON columns in credits_df
    logger.info("Parsing JSON columns in credits dataset...")
    for col in JSON_COLUMNS_CREDITS:
        if col in credits_df.columns:
            credits_df[col] = credits_df[col].apply(parse_json_field)

    # 4. Merge datasets on TMDB ID
    logger.info("Merging movies and credits datasets...")
    # Drop 'title' from credits to prevent column collision ('title_x', 'title_y')
    credits_clean = credits_df.drop(columns=["title"])
    merged = pd.merge(
        movies_df,
        credits_clean,
        left_on="id",
        right_on="movie_id",
        how="inner",
    )
    # Ensure canonical column tmdb_id
    merged["tmdb_id"] = merged["id"].astype(int)

    # 5. Handle budget and revenue: preserve unknown figures as missing (np.nan), not 0
    # In TMDB, budget=0 and revenue=0 indicate missing/unrecorded values.
    merged["budget_raw"] = merged["budget"]
    merged["revenue_raw"] = merged["revenue"]
    merged["budget"] = merged["budget"].apply(lambda x: float(x) if pd.notna(x) and x > 0 else np.nan)
    merged["revenue"] = merged["revenue"].apply(lambda x: float(x) if pd.notna(x) and x > 0 else np.nan)
    merged["has_valid_budget"] = merged["budget"].notna()
    merged["has_valid_revenue"] = merged["revenue"].notna()

    # 6. Parse and extract release dates and release year
    merged["release_date_raw"] = merged["release_date"]
    release_dates = pd.to_datetime(merged["release_date"], errors="coerce")
    merged["release_year"] = release_dates.dt.year.astype("Int64")
    merged["release_date"] = release_dates.dt.strftime("%Y-%m-%d").where(release_dates.notna(), None)

    # 7. Normalize missing textual and numerical values
    merged["title"] = merged["title"].fillna("").astype(str).str.strip()
    merged["overview"] = merged["overview"].fillna("").astype(str).str.strip()
    merged["tagline"] = merged["tagline"].fillna("").astype(str).str.strip()
    merged["homepage"] = merged["homepage"].fillna("").astype(str).str.strip()
    merged["runtime"] = merged["runtime"].apply(lambda x: float(x) if pd.notna(x) and x > 0 else np.nan)
    merged["vote_average"] = merged["vote_average"].fillna(0.0).astype(float)
    merged["vote_count"] = merged["vote_count"].fillna(0).astype(int)
    merged["popularity"] = merged["popularity"].fillna(0.0).astype(float)

    # 8. Handle duplicate titles using release year & TMDB ID
    duplicate_titles = merged[merged["title"].duplicated(keep=False)]["title"].unique()
    logger.info(f"Identified duplicate titles across releases: {list(duplicate_titles)}")

    def disambiguate_title(row: pd.Series) -> str:
        t = row["title"]
        if t in duplicate_titles:
            yr = row["release_year"]
            if pd.notna(yr):
                return f"{t} ({yr})"
            return f"{t} (ID: {row['tmdb_id']})"
        return t

    merged["disambiguated_title"] = merged.apply(disambiguate_title, axis=1)
    merged["is_duplicate_title"] = merged["title"].isin(duplicate_titles)

    # 9. Extract directors and relevant crew
    logger.info("Extracting directors and specialized crew roles...")
    merged["directors"] = merged["crew"].apply(extract_directors)
    merged["director_names"] = merged["directors"].apply(lambda dirs: [d["name"] for d in dirs])

    merged["producers"] = merged["crew"].apply(lambda c: extract_crew_by_roles(c, ["Producer", "Executive Producer"]))
    merged["writers"] = merged["crew"].apply(lambda c: extract_crew_by_roles(c, ["Screenplay", "Writer", "Author", "Story"]))

    # 10. Extract convenience string lists for fast entity indexing
    merged["actor_names"] = merged["cast"].apply(
        lambda items: [c.get("name", "").strip() for c in items if isinstance(c, dict) and c.get("name")]
    )
    merged["genre_names"] = merged["genres"].apply(
        lambda items: [g.get("name", "").strip() for g in items if isinstance(g, dict) and g.get("name")]
    )
    merged["keyword_names"] = merged["keywords"].apply(
        lambda items: [k.get("name", "").strip() for k in items if isinstance(k, dict) and k.get("name")]
    )
    merged["company_names"] = merged["production_companies"].apply(
        lambda items: [c.get("name", "").strip() for c in items if isinstance(c, dict) and c.get("name")]
    )
    merged["country_names"] = merged["production_countries"].apply(
        lambda items: [c.get("name", "").strip() for c in items if isinstance(c, dict) and c.get("name")]
    )

    logger.info(f"Preprocessing completed. Processed {len(merged)} movie records.")
    return merged


def save_processed_dataset(
    df: pd.DataFrame,
    parquet_path: str | Path,
    jsonl_path: Optional[str | Path] = None,
) -> None:
    """Save processed DataFrame to Parquet and optional JSONL formats."""
    parquet_p = Path(parquet_path)
    parquet_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed dataset to Parquet: {parquet_p}")
    df.to_parquet(parquet_p, index=False)

    if jsonl_path:
        jsonl_p = Path(jsonl_path)
        jsonl_p.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving processed dataset to JSON Lines: {jsonl_p}")
        df.to_json(jsonl_p, orient="records", lines=True)


def load_processed_dataset(parquet_path: str | Path) -> pd.DataFrame:
    """Load the preprocessed dataset from Parquet."""
    return pd.read_parquet(parquet_path)
