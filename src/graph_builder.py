"""TMDB 5000 Knowledge Graph Construction using NetworkX MultiDiGraph.

Constructs an in-memory Knowledge Graph modeling movies, people (cast & crew),
genres, keywords, production companies, production countries, and languages.
"""

from __future__ import annotations
import logging
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Node prefix definitions
PREFIX_MOVIE = "movie"
PREFIX_PERSON = "person"
PREFIX_GENRE = "genre"
PREFIX_KEYWORD = "keyword"
PREFIX_COMPANY = "company"
PREFIX_COUNTRY = "country"
PREFIX_LANGUAGE = "language"

# Relationship constants
REL_HAS_CAST = "HAS_CAST"
REL_HAS_CREW = "HAS_CREW_MEMBER"
REL_DIRECTED_BY = "DIRECTED_BY"
REL_HAS_GENRE = "HAS_GENRE"
REL_HAS_KEYWORD = "HAS_KEYWORD"
REL_PRODUCED_BY = "PRODUCED_BY"
REL_PRODUCED_IN = "PRODUCED_IN"
REL_HAS_LANGUAGE = "HAS_LANGUAGE"


def _safe_float(val: Any) -> Optional[float]:
    """Convert value to float, returning None if NaN, None, or invalid."""
    if val is None or pd.isna(val):
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert value to int, returning None if NaN, None, or invalid."""
    if val is None or pd.isna(val):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> str:
    """Safely convert value to non-null trimmed string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def build_graph(df: pd.DataFrame) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """Build a NetworkX MultiDiGraph from the preprocessed TMDB DataFrame.

    Entities without reliable IDs are skipped according to deterministic rules.
    Person nodes are unified even when appearing as both cast and crew.
    Accidental duplicate edges are prevented using unique edge keys.

    Returns:
        (graph, stats_dictionary)
    """
    logger.info(f"Starting Knowledge Graph construction from {len(df)} movie records...")
    start_time = time.time()

    G = nx.MultiDiGraph()

    # Track skipped entities for data hygiene reporting
    skipped_counts = Counter()

    # Iterate through each movie row
    for _, row in df.iterrows():
        # --- 1. MOVIE NODE ---
        movie_tmdb_id = _safe_int(row.get("tmdb_id") or row.get("id"))
        if movie_tmdb_id is None:
            skipped_counts["movies_missing_id"] += 1
            continue

        movie_node_id = f"{PREFIX_MOVIE}:{movie_tmdb_id}"

        title = _safe_str(row.get("title"))
        disambiguated_title = _safe_str(row.get("disambiguated_title")) or title
        original_title = _safe_str(row.get("original_title")) or title
        release_date = row.get("release_date")
        if pd.isna(release_date) or not release_date:
            release_date = None
        else:
            release_date = str(release_date).strip()

        release_year = _safe_int(row.get("release_year"))
        runtime = _safe_float(row.get("runtime"))
        budget = _safe_float(row.get("budget"))
        revenue = _safe_float(row.get("revenue"))
        vote_average = float(row.get("vote_average", 0.0) or 0.0)
        vote_count = int(row.get("vote_count", 0) or 0)
        popularity = float(row.get("popularity", 0.0) or 0.0)
        overview = _safe_str(row.get("overview"))
        original_language = _safe_str(row.get("original_language"))

        G.add_node(
            movie_node_id,
            type="Movie",
            tmdb_id=movie_tmdb_id,
            title=title,
            disambiguated_title=disambiguated_title,
            original_title=original_title,
            release_date=release_date,
            release_year=release_year,
            runtime=runtime,
            budget=budget,
            revenue=revenue,
            vote_average=vote_average,
            vote_count=vote_count,
            popularity=popularity,
            overview=overview,
            original_language=original_language,
        )

        # --- 2. CAST (Movie -> Actor: HAS_CAST) ---
        cast_items = row.get("cast")
        if isinstance(cast_items, (list, np.ndarray)):
            for item in cast_items:
                if not isinstance(item, dict):
                    continue
                person_id = _safe_int(item.get("id"))
                if person_id is None or person_id <= 0:
                    skipped_counts["cast_missing_id"] += 1
                    continue

                person_node_id = f"{PREFIX_PERSON}:{person_id}"
                person_name = _safe_str(item.get("name"))
                person_gender = _safe_int(item.get("gender"))

                # Create or update Person node
                if person_node_id not in G:
                    G.add_node(
                        person_node_id,
                        type="Person",
                        tmdb_id=person_id,
                        name=person_name,
                        gender=person_gender,
                    )
                elif person_gender and person_gender > 0 and G.nodes[person_node_id].get("gender") in (None, 0):
                    G.nodes[person_node_id]["gender"] = person_gender

                # Add HAS_CAST edge with deterministic key to avoid accidental duplicates
                credit_id = _safe_str(item.get("credit_id"))
                order = _safe_int(item.get("order"))
                character = _safe_str(item.get("character"))
                cast_id = _safe_int(item.get("cast_id"))
                edge_key = f"cast:{credit_id or f'{person_id}_{order}'}"

                G.add_edge(
                    movie_node_id,
                    person_node_id,
                    key=edge_key,
                    relationship=REL_HAS_CAST,
                    character=character,
                    order=order,
                    credit_id=credit_id,
                    cast_id=cast_id,
                )

        # --- 3. CREW & DIRECTORS (DIRECTED_BY & HAS_CREW_MEMBER) ---
        crew_items = row.get("crew")
        if isinstance(crew_items, (list, np.ndarray)):
            for item in crew_items:
                if not isinstance(item, dict):
                    continue
                person_id = _safe_int(item.get("id"))
                if person_id is None or person_id <= 0:
                    skipped_counts["crew_missing_id"] += 1
                    continue

                person_node_id = f"{PREFIX_PERSON}:{person_id}"
                person_name = _safe_str(item.get("name"))
                person_gender = _safe_int(item.get("gender"))

                # Create or update Person node (shared with cast)
                if person_node_id not in G:
                    G.add_node(
                        person_node_id,
                        type="Person",
                        tmdb_id=person_id,
                        name=person_name,
                        gender=person_gender,
                    )
                elif person_gender and person_gender > 0 and G.nodes[person_node_id].get("gender") in (None, 0):
                    G.nodes[person_node_id]["gender"] = person_gender

                job = _safe_str(item.get("job"))
                department = _safe_str(item.get("department"))
                credit_id = _safe_str(item.get("credit_id"))

                # Check if director (case-insensitive)
                is_director = job.lower() == "director"
                if is_director:
                    rel_type = REL_DIRECTED_BY
                    edge_key = f"director:{credit_id or person_id}"
                else:
                    rel_type = REL_HAS_CREW
                    edge_key = f"crew:{credit_id or f'{person_id}_{job}'}"

                G.add_edge(
                    movie_node_id,
                    person_node_id,
                    key=edge_key,
                    relationship=rel_type,
                    job=job,
                    department=department,
                    credit_id=credit_id,
                )

        # --- 4. GENRES (Movie -> Genre: HAS_GENRE) ---
        genre_items = row.get("genres")
        if isinstance(genre_items, (list, np.ndarray)):
            for item in genre_items:
                if not isinstance(item, dict):
                    continue
                gid = _safe_int(item.get("id"))
                if gid is None:
                    skipped_counts["genres_missing_id"] += 1
                    continue

                gname = _safe_str(item.get("name"))
                genre_node_id = f"{PREFIX_GENRE}:{gid}"
                if genre_node_id not in G:
                    G.add_node(genre_node_id, type="Genre", id=gid, name=gname)

                G.add_edge(
                    movie_node_id,
                    genre_node_id,
                    key=f"genre:{gid}",
                    relationship=REL_HAS_GENRE,
                )

        # --- 5. KEYWORDS (Movie -> Keyword: HAS_KEYWORD) ---
        keyword_items = row.get("keywords")
        if isinstance(keyword_items, (list, np.ndarray)):
            for item in keyword_items:
                if not isinstance(item, dict):
                    continue
                kid = _safe_int(item.get("id"))
                if kid is None:
                    skipped_counts["keywords_missing_id"] += 1
                    continue

                kname = _safe_str(item.get("name"))
                keyword_node_id = f"{PREFIX_KEYWORD}:{kid}"
                if keyword_node_id not in G:
                    G.add_node(keyword_node_id, type="Keyword", id=kid, name=kname)

                G.add_edge(
                    movie_node_id,
                    keyword_node_id,
                    key=f"keyword:{kid}",
                    relationship=REL_HAS_KEYWORD,
                )

        # --- 6. PRODUCTION COMPANIES (Movie -> Company: PRODUCED_BY) ---
        company_items = row.get("production_companies")
        if isinstance(company_items, (list, np.ndarray)):
            for item in company_items:
                if not isinstance(item, dict):
                    continue
                cid = _safe_int(item.get("id"))
                if cid is None:
                    skipped_counts["companies_missing_id"] += 1
                    continue

                cname = _safe_str(item.get("name"))
                company_node_id = f"{PREFIX_COMPANY}:{cid}"
                if company_node_id not in G:
                    G.add_node(company_node_id, type="ProductionCompany", id=cid, name=cname)

                G.add_edge(
                    movie_node_id,
                    company_node_id,
                    key=f"company:{cid}",
                    relationship=REL_PRODUCED_BY,
                )

        # --- 7. PRODUCTION COUNTRIES (Movie -> Country: PRODUCED_IN) ---
        country_items = row.get("production_countries")
        if isinstance(country_items, (list, np.ndarray)):
            for item in country_items:
                if not isinstance(item, dict):
                    continue
                iso_code = _safe_str(item.get("iso_3166_1")).upper()
                cname = _safe_str(item.get("name"))
                if not iso_code:
                    skipped_counts["countries_missing_code"] += 1
                    continue

                country_node_id = f"{PREFIX_COUNTRY}:{iso_code}"
                if country_node_id not in G:
                    G.add_node(country_node_id, type="ProductionCountry", code=iso_code, name=cname)

                G.add_edge(
                    movie_node_id,
                    country_node_id,
                    key=f"country:{iso_code}",
                    relationship=REL_PRODUCED_IN,
                )

        # --- 8. SPOKEN LANGUAGES (Movie -> Language: HAS_LANGUAGE) ---
        lang_items = row.get("spoken_languages")
        if isinstance(lang_items, (list, np.ndarray)):
            for item in lang_items:
                if not isinstance(item, dict):
                    continue
                lang_code = _safe_str(item.get("iso_639_1")).lower()
                lname = _safe_str(item.get("name"))
                if not lang_code:
                    skipped_counts["languages_missing_code"] += 1
                    continue

                lang_node_id = f"{PREFIX_LANGUAGE}:{lang_code}"
                if lang_node_id not in G:
                    G.add_node(lang_node_id, type="Language", code=lang_code, name=lname)

                G.add_edge(
                    movie_node_id,
                    lang_node_id,
                    key=f"language:{lang_code}",
                    relationship=REL_HAS_LANGUAGE,
                )

    construction_time = time.time() - start_time
    logger.info(
        f"Graph construction completed in {construction_time:.2f}s: "
        f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges."
    )

    stats = get_graph_statistics(G, construction_time=construction_time)
    stats["skipped_entities"] = dict(skipped_counts)
    return G, stats


def get_graph_statistics(
    G: nx.MultiDiGraph, construction_time: float = 0.0
) -> Dict[str, Any]:
    """Calculate thorough graph validation metrics directly from the graph instance."""
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()

    # Count nodes grouped by type
    nodes_by_type = Counter()
    for _, data in G.nodes(data=True):
        nodes_by_type[data.get("type", "Unknown")] += 1

    # Count edges grouped by relationship
    edges_by_relationship = Counter()
    for _, _, data in G.edges(data=True):
        edges_by_relationship[data.get("relationship", "Unknown")] += 1

    # Movie-specific connectivity metrics
    movies_with_cast = 0
    movies_with_directors = 0
    movies_with_genres = 0

    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Movie"]
    for m in movie_nodes:
        out_edges = G[m]
        has_cast = False
        has_dir = False
        has_genre = False
        for target, edges_dict in out_edges.items():
            for _, edge_attr in edges_dict.items():
                rel = edge_attr.get("relationship")
                if rel == REL_HAS_CAST:
                    has_cast = True
                elif rel == REL_DIRECTED_BY:
                    has_dir = True
                elif rel == REL_HAS_GENRE:
                    has_genre = True
            if has_cast and has_dir and has_genre:
                break
        if has_cast:
            movies_with_cast += 1
        if has_dir:
            movies_with_directors += 1
        if has_genre:
            movies_with_genres += 1

    isolated_nodes_count = len(list(nx.isolates(G)))

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "nodes_by_type": dict(nodes_by_type),
        "edges_by_relationship": dict(edges_by_relationship),
        "total_movies": len(movie_nodes),
        "movies_with_cast": movies_with_cast,
        "movies_with_directors": movies_with_directors,
        "movies_with_genres": movies_with_genres,
        "isolated_nodes": isolated_nodes_count,
        "construction_time_seconds": round(construction_time, 2),
    }


def save_graph(G: nx.MultiDiGraph, file_path: Union[str, Path]) -> None:
    """Save the graph to disk using Python pickle serialization.

    Format Decision:
      NetworkX graphs containing arbitrary attributes (e.g. None values for
      unrecorded budget/revenue, nested metadata, multi-edges with custom keys)
      cannot be cleanly serialized to GraphML without losing types or failing
      on None values. Python pickle preserves exact native Python types,
      MultiDiGraph keys, and attribute structures with sub-second save/load times.

    Limitations:
      - Python-specific binary format.
      - Requires loading with trusted code (standard pickle safety).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Serializing Knowledge Graph to: {path}")
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Graph successfully saved ({path.stat().st_size / (1024 * 1024):.2f} MB).")


def load_graph(file_path: Union[str, Path]) -> nx.MultiDiGraph:
    """Load a previously serialized MultiDiGraph from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge Graph file not found: {path}")
    logger.info(f"Loading Knowledge Graph from: {path}")
    with open(path, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError(f"Loaded object is not a NetworkX MultiDiGraph: {type(G)}")
    logger.info(f"Loaded graph with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
    return G
