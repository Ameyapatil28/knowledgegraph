"""Knowledge Graph Query Engine for TMDB 5000.

Provides deterministic, structured graph traversals and analytical queries
over the NetworkX MultiDiGraph without requiring an LLM or vector database.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from src.config import GRAPH_PICKLE_PATH
from src.graph_builder import (
    load_graph,
    PREFIX_MOVIE,
    PREFIX_PERSON,
    PREFIX_GENRE,
    PREFIX_KEYWORD,
    PREFIX_COMPANY,
    PREFIX_COUNTRY,
    PREFIX_LANGUAGE,
    REL_HAS_CAST,
    REL_HAS_CREW,
    REL_DIRECTED_BY,
    REL_HAS_GENRE,
    REL_HAS_KEYWORD,
    REL_PRODUCED_BY,
    REL_PRODUCED_IN,
    REL_HAS_LANGUAGE,
)

logger = logging.getLogger(__name__)

# Allowed query operations
SUPPORTED_OPERATIONS = {
    "movie_details",
    "search_movies",
    "movie_cast",
    "movie_crew",
    "movie_directors",
    "movies_by_person",
    "movies_by_genre",
    "movies_by_keyword",
    "movies_by_company",
    "top_movies",
    "compare_movies",
    "common_cast",
    "aggregate_by_genre",
    "movies_by_year_range",
    "count_by_year",
}

# Allowed ranking and analytical metrics
SUPPORTED_METRICS = {
    "vote_average",
    "popularity",
    "revenue",
    "budget",
    "runtime",
    "vote_count",
}

# Allowed roles for person searches
SUPPORTED_ROLES = {
    "all",
    "actor",
    "cast",
    "director",
    "crew",
    "producer",
    "writer",
    "screenplay",
}

# Allowed sort orders
SUPPORTED_SORT_ORDERS = {
    "asc",
    "desc",
    "ascending",
    "descending",
}


def _normalize_sort_order(order: Optional[str], default: str = "desc") -> str:
    """Normalize sort order string to 'asc' or 'desc'."""
    if not order:
        return default
    norm = order.strip().lower()
    if norm in ("asc", "ascending"):
        return "asc"
    elif norm in ("desc", "descending"):
        return "desc"
    return default


# Allowed parameters per operation for strict security & boundary validation
ALLOWED_PARAMETERS_PER_OPERATION = {
    "movie_details": {"movie_id", "title", "query"},
    "search_movies": {"query", "title", "limit"},
    "movie_cast": {"movie_id", "title", "limit"},
    "movie_crew": {"movie_id", "title", "department", "job", "limit"},
    "movie_directors": {"movie_id", "title"},
    "movies_by_person": {"person_name", "person_id", "role", "limit", "sort_order"},
    "movies_by_genre": {"genre_name", "genre_id", "metric", "min_vote_count", "sort_order", "limit"},
    "movies_by_keyword": {"keyword_name", "keyword_id", "metric", "sort_order", "limit"},
    "movies_by_company": {"company_name", "company_id", "metric", "sort_order", "limit"},
    "top_movies": {"metric", "limit", "min_vote_count", "sort_order", "genre", "year_start", "year_end"},
    "compare_movies": {"movies", "movie_titles", "movie_ids"},
    "common_cast": {"movies", "movie_titles", "movie_ids"},
    "aggregate_by_genre": {"metric", "min_vote_count"},
    "movies_by_year_range": {"year_start", "year_end", "sort_order", "limit"},
    "count_by_year": {"year_start", "year_end", "sort_order"},
}


class QueryEngineError(Exception):
    """Custom exception for query engine execution errors."""
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class QuerySpecification(BaseModel):
    """Pydantic model validating structured query specifications."""
    operation: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def validate_spec(self) -> None:
        op = self.operation.strip().lower()
        if op not in SUPPORTED_OPERATIONS:
            raise QueryEngineError(
                code="INVALID_OPERATION",
                message=f"Unsupported operation '{self.operation}'. Allowed: {sorted(SUPPORTED_OPERATIONS)}",
            )

        params = self.parameters

        # Strict validation against unexpected parameters
        allowed = ALLOWED_PARAMETERS_PER_OPERATION.get(op, set())
        unexpected = set(params.keys()) - allowed
        if unexpected:
            raise QueryEngineError(
                code="UNEXPECTED_PARAMETER",
                message=f"Operation '{op}' received unexpected parameter(s): {sorted(unexpected)}. Allowed: {sorted(allowed)}",
            )

        # Validate sort_order if present
        if "sort_order" in params:
            so = str(params["sort_order"]).strip().lower()
            if so not in SUPPORTED_SORT_ORDERS:
                raise QueryEngineError(
                    code="INVALID_PARAMETER",
                    message=f"Unsupported sort_order '{params['sort_order']}'. Allowed: {sorted(SUPPORTED_SORT_ORDERS)}",
                )

        # Validate min_vote_count if present
        if "min_vote_count" in params:
            try:
                mvc = int(params["min_vote_count"])
                if mvc < 0:
                    raise ValueError
            except (ValueError, TypeError):
                raise QueryEngineError(
                    code="INVALID_PARAMETER",
                    message="min_vote_count must be a non-negative integer.",
                )

        # Parameter validation per operation
        if op == "movie_details":
            if not params.get("movie_id") and not params.get("title") and not params.get("query"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movie_details requires 'movie_id', 'title', or 'query'",
                )

        elif op == "search_movies":
            if not params.get("query") and not params.get("title"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="search_movies requires 'query' or 'title'",
                )

        elif op in ("movie_cast", "movie_crew", "movie_directors"):
            if not params.get("movie_id") and not params.get("title"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message=f"{op} requires 'movie_id' or 'title'",
                )

        elif op == "movies_by_person":
            if not params.get("person_name") and not params.get("person_id"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movies_by_person requires 'person_name' or 'person_id'",
                )
            role = params.get("role")
            if role and str(role).strip().lower() not in SUPPORTED_ROLES:
                raise QueryEngineError(
                    code="INVALID_ROLE",
                    message=f"Unsupported role '{role}'. Allowed: {sorted(SUPPORTED_ROLES)}",
                )

        elif op == "movies_by_genre":
            if not params.get("genre_name") and not params.get("genre_id"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movies_by_genre requires 'genre_name' or 'genre_id'",
                )
            metric = params.get("metric", "vote_average")
            if metric not in SUPPORTED_METRICS:
                raise QueryEngineError(
                    code="INVALID_METRIC",
                    message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
                )

        elif op == "movies_by_keyword":
            if not params.get("keyword_name") and not params.get("keyword_id"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movies_by_keyword requires 'keyword_name' or 'keyword_id'",
                )
            metric = params.get("metric", "popularity")
            if metric not in SUPPORTED_METRICS:
                raise QueryEngineError(
                    code="INVALID_METRIC",
                    message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
                )

        elif op == "movies_by_company":
            if not params.get("company_name") and not params.get("company_id"):
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movies_by_company requires 'company_name' or 'company_id'",
                )
            metric = params.get("metric", "revenue")
            if metric not in SUPPORTED_METRICS:
                raise QueryEngineError(
                    code="INVALID_METRIC",
                    message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
                )

        elif op == "top_movies":
            metric = params.get("metric", "vote_average")
            if metric not in SUPPORTED_METRICS:
                raise QueryEngineError(
                    code="INVALID_METRIC",
                    message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
                )

        elif op in ("compare_movies", "common_cast"):
            movies = params.get("movies") or params.get("movie_titles") or params.get("movie_ids")
            if not movies or not isinstance(movies, list) or len(movies) < 2:
                raise QueryEngineError(
                    code="INVALID_PARAMETER",
                    message=f"{op} requires a 'movies' list containing at least 2 movie titles or IDs",
                )

        elif op == "aggregate_by_genre":
            metric = params.get("metric", "vote_average")
            if metric not in ("vote_average", "popularity", "revenue", "budget"):
                raise QueryEngineError(
                    code="INVALID_METRIC",
                    message=f"Metric '{metric}' not supported for aggregate_by_genre. Allowed: vote_average, popularity, revenue, budget",
                )

        elif op == "movies_by_year_range":
            if "year_start" not in params or "year_end" not in params:
                raise QueryEngineError(
                    code="MISSING_PARAMETER",
                    message="movies_by_year_range requires 'year_start' and 'year_end'",
                )


class GraphQueryEngine:
    """High-performance Knowledge Graph Query Engine with in-memory inverted indexes."""

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self._build_indexes()

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Alias property for backward compatibility with external consumers."""
        return self.G

    def _build_indexes(self) -> None:
        """Build in-memory inverted indexes for fast O(1) entity lookups."""
        logger.info("Building in-memory query indexes from Knowledge Graph...")

        self.movie_id_to_node: Dict[int, str] = {}
        self.title_to_movie_ids: Dict[str, List[str]] = {}
        self.disambiguated_to_movie_id: Dict[str, str] = {}

        self.person_name_to_ids: Dict[str, List[str]] = {}
        self.person_id_to_node: Dict[int, str] = {}

        self.genre_name_to_id: Dict[str, str] = {}
        self.keyword_name_to_id: Dict[str, str] = {}
        self.company_name_to_id: Dict[str, str] = {}

        self.movie_records: List[Dict[str, Any]] = []

        for node_id, data in self.G.nodes(data=True):
            ntype = data.get("type")

            if ntype == "Movie":
                tmdb_id = data.get("tmdb_id")
                if tmdb_id is not None:
                    self.movie_id_to_node[int(tmdb_id)] = node_id

                title = data.get("title", "")
                if title:
                    norm_title = title.strip().lower()
                    self.title_to_movie_ids.setdefault(norm_title, []).append(node_id)

                dis_title = data.get("disambiguated_title", "")
                if dis_title:
                    self.disambiguated_to_movie_id[dis_title.strip().lower()] = node_id

                rec = dict(data)
                rec["node_id"] = node_id
                self.movie_records.append(rec)

            elif ntype == "Person":
                tmdb_id = data.get("tmdb_id")
                if tmdb_id is not None:
                    self.person_id_to_node[int(tmdb_id)] = node_id
                name = data.get("name", "")
                if name:
                    norm_name = name.strip().lower()
                    self.person_name_to_ids.setdefault(norm_name, []).append(node_id)

            elif ntype == "Genre":
                gname = data.get("name", "")
                if gname:
                    self.genre_name_to_id[gname.strip().lower()] = node_id

            elif ntype == "Keyword":
                kname = data.get("name", "")
                if kname:
                    self.keyword_name_to_id[kname.strip().lower()] = node_id

            elif ntype == "ProductionCompany":
                cname = data.get("name", "")
                if cname:
                    self.company_name_to_id[cname.strip().lower()] = node_id

        # Convert cached movie records to a Pandas DataFrame for lightning-fast analytics
        self.movies_df = pd.DataFrame(self.movie_records)

        # Precompute movie genres map for fast genre queries and aggregations
        self.movie_genres_map: Dict[str, List[str]] = {}
        for m_node in self.movie_id_to_node.values():
            genres = []
            for _, target, edge_data in self.G.out_edges(m_node, data=True):
                if edge_data.get("relationship") == REL_HAS_GENRE:
                    g_name = self.G.nodes[target].get("name")
                    if g_name:
                        genres.append(g_name)
            self.movie_genres_map[m_node] = genres

        logger.info(
            f"Indexes ready: {len(self.movie_id_to_node):,} movies, "
            f"{len(self.person_name_to_ids):,} people names, "
            f"{len(self.genre_name_to_id)} genres indexed."
        )

    # -------------------------------------------------------------------------
    # Node Resolution Helpers
    # -------------------------------------------------------------------------
    def resolve_movie_nodes(self, identifier: Union[str, int]) -> List[str]:
        """Resolve a movie identifier (TMDB ID, node ID, or title) to matching node IDs."""
        if identifier is None:
            return []

        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            node_id = self.movie_id_to_node.get(int(identifier))
            return [node_id] if node_id else []

        id_str = str(identifier).strip()
        if id_str.startswith(f"{PREFIX_MOVIE}:"):
            return [id_str] if id_str in self.G else []

        norm_id = id_str.lower()
        if norm_id in self.disambiguated_to_movie_id:
            return [self.disambiguated_to_movie_id[norm_id]]

        if norm_id in self.title_to_movie_ids:
            return self.title_to_movie_ids[norm_id]

        return []

    def resolve_person_nodes(self, identifier: Union[str, int]) -> List[str]:
        """Resolve a person identifier (TMDB ID, node ID, or name) to matching node IDs."""
        if identifier is None:
            return []

        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            node_id = self.person_id_to_node.get(int(identifier))
            return [node_id] if node_id else []

        id_str = str(identifier).strip()
        if id_str.startswith(f"{PREFIX_PERSON}:"):
            return [id_str] if id_str in self.G else []

        norm_name = id_str.lower()
        return self.person_name_to_ids.get(norm_name, [])

    def resolve_genre_node(self, identifier: Union[str, int]) -> Optional[str]:
        """Resolve genre name or ID to genre node ID."""
        if identifier is None:
            return None
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            node_id = f"{PREFIX_GENRE}:{identifier}"
            return node_id if node_id in self.G else None

        id_str = str(identifier).strip()
        if id_str.startswith(f"{PREFIX_GENRE}:"):
            return id_str if id_str in self.G else None

        return self.genre_name_to_id.get(id_str.lower())

    def resolve_keyword_node(self, identifier: Union[str, int]) -> Optional[str]:
        """Resolve keyword name or ID to keyword node ID."""
        if identifier is None:
            return None
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            node_id = f"{PREFIX_KEYWORD}:{identifier}"
            return node_id if node_id in self.G else None

        id_str = str(identifier).strip()
        if id_str.startswith(f"{PREFIX_KEYWORD}:"):
            return id_str if id_str in self.G else None

        return self.keyword_name_to_id.get(id_str.lower())

    def resolve_company_node(self, identifier: Union[str, int]) -> Optional[str]:
        """Resolve company name or ID to company node ID."""
        if identifier is None:
            return None
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            node_id = f"{PREFIX_COMPANY}:{identifier}"
            return node_id if node_id in self.G else None

        id_str = str(identifier).strip()
        if id_str.startswith(f"{PREFIX_COMPANY}:"):
            return id_str if id_str in self.G else None

        return self.company_name_to_id.get(id_str.lower())

    # -------------------------------------------------------------------------
    # Core Graph Retrieval Operations
    # -------------------------------------------------------------------------
    def get_movie_details(self, movie_id_or_title: Union[str, int]) -> Dict[str, Any]:
        """Retrieve full movie details, genres, directors, companies, and metadata."""
        node_ids = self.resolve_movie_nodes(movie_id_or_title)
        if not node_ids:
            raise QueryEngineError(
                code="MOVIE_NOT_FOUND",
                message=f"Movie '{movie_id_or_title}' not found in Knowledge Graph.",
            )

        movies_details = []
        for node_id in node_ids:
            node_data = dict(self.G.nodes[node_id])

            genres = []
            keywords = []
            companies = []
            countries = []
            languages = []
            directors = []
            actors_summary = []

            for _, target, edge_data in self.G.out_edges(node_id, data=True):
                rel = edge_data.get("relationship")
                tdata = self.G.nodes[target]

                if rel == REL_HAS_GENRE:
                    genres.append({"id": tdata.get("id"), "name": tdata.get("name")})
                elif rel == REL_HAS_KEYWORD:
                    keywords.append({"id": tdata.get("id"), "name": tdata.get("name")})
                elif rel == REL_PRODUCED_BY:
                    companies.append({"id": tdata.get("id"), "name": tdata.get("name")})
                elif rel == REL_PRODUCED_IN:
                    countries.append({"code": tdata.get("code"), "name": tdata.get("name")})
                elif rel == REL_HAS_LANGUAGE:
                    languages.append({"code": tdata.get("code"), "name": tdata.get("name")})
                elif rel == REL_DIRECTED_BY:
                    directors.append({
                        "tmdb_id": tdata.get("tmdb_id"),
                        "name": tdata.get("name"),
                        "job": edge_data.get("job", "Director"),
                    })
                elif rel == REL_HAS_CAST:
                    if len(actors_summary) < 5:
                        actors_summary.append({
                            "tmdb_id": tdata.get("tmdb_id"),
                            "name": tdata.get("name"),
                            "character": edge_data.get("character"),
                            "order": edge_data.get("order"),
                        })

            actors_summary.sort(key=lambda x: (x.get("order") if x.get("order") is not None else 999))

            movie_dict = {
                "tmdb_id": node_data.get("tmdb_id"),
                "title": node_data.get("title"),
                "disambiguated_title": node_data.get("disambiguated_title"),
                "original_title": node_data.get("original_title"),
                "release_date": node_data.get("release_date"),
                "release_year": node_data.get("release_year"),
                "runtime": node_data.get("runtime"),
                "budget": node_data.get("budget"),
                "revenue": node_data.get("revenue"),
                "vote_average": node_data.get("vote_average"),
                "vote_count": node_data.get("vote_count"),
                "popularity": node_data.get("popularity"),
                "overview": node_data.get("overview"),
                "original_language": node_data.get("original_language"),
                "directors": directors,
                "genres": genres,
                "keywords": keywords,
                "production_companies": companies,
                "production_countries": countries,
                "spoken_languages": languages,
                "top_cast_preview": actors_summary,
            }
            movies_details.append(movie_dict)

        if len(movies_details) == 1:
            return movies_details[0]
        return {"ambiguous": True, "count": len(movies_details), "movies": movies_details}

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title using case-insensitive matching with deterministic tie-breaking."""
        if not query or not query.strip():
            return []

        q = query.strip().lower()
        limit = min(max(1, limit), 100)

        matches = []
        for rec in self.movie_records:
            title = rec.get("title", "").lower()
            dis_title = rec.get("disambiguated_title", "").lower()
            orig_title = rec.get("original_title", "").lower()

            if q in title or q in dis_title or q in orig_title:
                matches.append({
                    "tmdb_id": rec.get("tmdb_id"),
                    "title": rec.get("title"),
                    "disambiguated_title": rec.get("disambiguated_title"),
                    "release_year": rec.get("release_year"),
                    "vote_average": rec.get("vote_average"),
                    "vote_count": rec.get("vote_count"),
                    "popularity": rec.get("popularity"),
                })
                if len(matches) >= limit * 3:
                    break

        # Deterministic ranking: exact match (0) -> prefix match (1) -> substring (2)
        # Tiebreak on popularity desc, vote_count desc, title asc, tmdb_id asc
        matches.sort(
            key=lambda m: (
                0 if m["title"].lower() == q else (1 if m["title"].lower().startswith(q) else 2),
                -float(m.get("popularity") or 0.0),
                -int(m.get("vote_count") or 0),
                m["title"].lower(),
                m.get("tmdb_id"),
            )
        )
        return matches[:limit]

    def get_movie_cast(
        self, movie_id_or_title: Union[str, int], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve cast members for a movie sorted by cast order."""
        node_ids = self.resolve_movie_nodes(movie_id_or_title)
        if not node_ids:
            raise QueryEngineError(
                code="MOVIE_NOT_FOUND",
                message=f"Movie '{movie_id_or_title}' not found in Knowledge Graph.",
            )

        cast_list = []
        for movie_node in node_ids:
            for _, person_node, edge_data in self.G.out_edges(movie_node, data=True):
                if edge_data.get("relationship") == REL_HAS_CAST:
                    pdata = self.G.nodes[person_node]
                    order_val = edge_data.get("order")
                    cast_list.append({
                        "movie_id": self.G.nodes[movie_node].get("tmdb_id"),
                        "movie_title": self.G.nodes[movie_node].get("title"),
                        "person_id": pdata.get("tmdb_id"),
                        "person_name": pdata.get("name"),
                        "character": edge_data.get("character"),
                        "order": order_val if order_val is not None else 9999,
                        "gender": pdata.get("gender"),
                        "credit_id": edge_data.get("credit_id"),
                    })

        cast_list.sort(key=lambda c: (c.get("order", 9999), c.get("person_name", "")))
        if limit:
            return cast_list[: min(limit, 200)]
        return cast_list

    def get_movie_crew(
        self,
        movie_id_or_title: Union[str, int],
        department: Optional[str] = None,
        job: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve crew members (including directors) with optional department/job filtering."""
        node_ids = self.resolve_movie_nodes(movie_id_or_title)
        if not node_ids:
            raise QueryEngineError(
                code="MOVIE_NOT_FOUND",
                message=f"Movie '{movie_id_or_title}' not found in Knowledge Graph.",
            )

        dep_filter = department.strip().lower() if department else None
        job_filter = job.strip().lower() if job else None

        crew_list = []
        for movie_node in node_ids:
            for _, person_node, edge_data in self.G.out_edges(movie_node, data=True):
                rel = edge_data.get("relationship")
                if rel in (REL_HAS_CREW, REL_DIRECTED_BY):
                    c_job = edge_data.get("job", "")
                    c_dep = edge_data.get("department", "")

                    if dep_filter and c_dep.lower() != dep_filter:
                        continue
                    if job_filter and c_job.lower() != job_filter:
                        continue

                    pdata = self.G.nodes[person_node]
                    crew_list.append({
                        "movie_id": self.G.nodes[movie_node].get("tmdb_id"),
                        "movie_title": self.G.nodes[movie_node].get("title"),
                        "person_id": pdata.get("tmdb_id"),
                        "person_name": pdata.get("name"),
                        "job": c_job,
                        "department": c_dep,
                        "relationship": rel,
                        "credit_id": edge_data.get("credit_id"),
                    })

        crew_list.sort(key=lambda x: (x.get("department", ""), x.get("job", ""), x.get("person_name", "")))
        if limit:
            return crew_list[: min(limit, 200)]
        return crew_list

    def get_movie_directors(self, movie_id_or_title: Union[str, int]) -> List[Dict[str, Any]]:
        """Retrieve directors for a movie."""
        node_ids = self.resolve_movie_nodes(movie_id_or_title)
        if not node_ids:
            raise QueryEngineError(
                code="MOVIE_NOT_FOUND",
                message=f"Movie '{movie_id_or_title}' not found in Knowledge Graph.",
            )

        directors = []
        seen = set()
        for movie_node in node_ids:
            m_title = self.G.nodes[movie_node].get("title")
            m_id = self.G.nodes[movie_node].get("tmdb_id")
            for _, person_node, edge_data in self.G.out_edges(movie_node, data=True):
                if edge_data.get("relationship") == REL_DIRECTED_BY:
                    pdata = self.G.nodes[person_node]
                    pid = pdata.get("tmdb_id")
                    if (m_id, pid) not in seen:
                        seen.add((m_id, pid))
                        directors.append({
                            "movie_id": m_id,
                            "movie_title": m_title,
                            "person_id": pid,
                            "person_name": pdata.get("name"),
                            "job": edge_data.get("job", "Director"),
                            "department": edge_data.get("department", "Directing"),
                            "credit_id": edge_data.get("credit_id"),
                        })
        return directors

    def get_movies_by_person(
        self,
        person_name_or_id: Union[str, int],
        role: Optional[str] = None,
        limit: int = 50,
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Find movies associated with a person, optionally filtered by role."""
        person_nodes = self.resolve_person_nodes(person_name_or_id)
        if not person_nodes:
            raise QueryEngineError(
                code="PERSON_NOT_FOUND",
                message=f"Person '{person_name_or_id}' not found in Knowledge Graph.",
            )

        norm_role = role.strip().lower() if role else "all"
        if norm_role not in SUPPORTED_ROLES:
            raise QueryEngineError(
                code="INVALID_ROLE",
                message=f"Unsupported role '{role}'. Allowed: {sorted(SUPPORTED_ROLES)}",
            )

        results = []
        seen_keys = set()

        for p_node in person_nodes:
            p_data = self.G.nodes[p_node]
            p_id = p_data.get("tmdb_id")
            p_name = p_data.get("name")

            for movie_node, _, edge_data in self.G.in_edges(p_node, data=True):
                rel = edge_data.get("relationship")
                job = edge_data.get("job", "")
                norm_job = job.lower()

                match = False
                if norm_role in ("all", None):
                    match = True
                elif norm_role in ("actor", "cast") and rel == REL_HAS_CAST:
                    match = True
                elif norm_role == "director" and rel == REL_DIRECTED_BY:
                    match = True
                elif norm_role == "crew" and rel in (REL_HAS_CREW, REL_DIRECTED_BY):
                    match = True
                elif norm_role == "producer" and norm_job in ("producer", "executive producer"):
                    match = True
                elif norm_role in ("writer", "screenplay") and norm_job in ("writer", "screenplay", "author", "story"):
                    match = True

                if match:
                    m_data = self.G.nodes[movie_node]
                    m_id = m_data.get("tmdb_id")
                    dedup_key = (m_id, p_id, rel, job, edge_data.get("character"))
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        results.append({
                            "movie_id": m_id,
                            "movie_title": m_data.get("title"),
                            "disambiguated_title": m_data.get("disambiguated_title"),
                            "release_year": m_data.get("release_year"),
                            "vote_average": m_data.get("vote_average"),
                            "vote_count": m_data.get("vote_count"),
                            "popularity": m_data.get("popularity"),
                            "person_id": p_id,
                            "person_name": p_name,
                            "relationship": rel,
                            "role_description": edge_data.get("character") if rel == REL_HAS_CAST else job,
                            "character": edge_data.get("character") if rel == REL_HAS_CAST else None,
                            "job": job if rel in (REL_HAS_CREW, REL_DIRECTED_BY) else None,
                        })

        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_desc = norm_sort == "desc"

        # Deterministic sort with tiebreakers
        results.sort(
            key=lambda r: (
                r.get("release_year") is None,
                -(r.get("release_year") or 0) if is_desc else (r.get("release_year") or 9999),
                -(r.get("vote_average") or 0.0),
                -(r.get("vote_count") or 0),
                (r.get("movie_title") or "").lower(),
                r.get("movie_id") or 0,
            )
        )
        return results[: min(limit, 200)]

    def get_movies_by_genre(
        self,
        genre_name_or_id: Union[str, int],
        metric: str = "vote_average",
        min_vote_count: int = 0,
        sort_order: str = "desc",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find movies belonging to a genre with strict metric ordering and tie-breaking."""
        g_node = self.resolve_genre_node(genre_name_or_id)
        if not g_node:
            raise QueryEngineError(
                code="GENRE_NOT_FOUND",
                message=f"Genre '{genre_name_or_id}' not found in Knowledge Graph.",
            )

        if metric not in SUPPORTED_METRICS:
            raise QueryEngineError(
                code="INVALID_METRIC",
                message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
            )

        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_desc = norm_sort == "desc"

        g_data = self.G.nodes[g_node]
        results = []
        for movie_node, _, _ in self.G.in_edges(g_node, data=True):
            m = self.G.nodes[movie_node]
            v_count = m.get("vote_count", 0) or 0
            if min_vote_count > 0 and v_count < min_vote_count:
                continue

            metric_val = m.get(metric)
            if metric in ("budget", "revenue") and (metric_val is None or metric_val <= 0):
                continue

            results.append({
                "movie_id": m.get("tmdb_id"),
                "title": m.get("title"),
                "release_year": m.get("release_year"),
                "release_date": m.get("release_date"),
                "vote_average": float(m.get("vote_average") or 0.0),
                "vote_count": int(v_count),
                "popularity": float(m.get("popularity") or 0.0),
                "revenue": float(m.get("revenue")) if m.get("revenue") is not None else None,
                "budget": float(m.get("budget")) if m.get("budget") is not None else None,
                "genre_name": g_data.get("name"),
            })

        # Deterministic sorting:
        # Primary: requested metric (descending or ascending)
        # Secondary: vote_count descending
        # Tertiary: title ascending
        # Quaternary: movie_id ascending
        def sort_key(item: Dict[str, Any]):
            val = item.get(metric)
            is_none = val is None
            if is_desc:
                num_val = -float(val) if not is_none else float("inf")
            else:
                num_val = float(val) if not is_none else float("inf")

            return (
                is_none,
                num_val,
                -int(item.get("vote_count", 0)),
                item["title"].lower(),
                item["movie_id"],
            )

        results.sort(key=sort_key)
        return results[: min(limit, 200)]

    def get_movies_by_keyword(
        self,
        keyword_name_or_id: Union[str, int],
        metric: str = "popularity",
        sort_order: str = "desc",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find movies associated with a keyword with deterministic sorting."""
        k_node = self.resolve_keyword_node(keyword_name_or_id)
        if not k_node:
            raise QueryEngineError(
                code="KEYWORD_NOT_FOUND",
                message=f"Keyword '{keyword_name_or_id}' not found in Knowledge Graph.",
            )

        if metric not in SUPPORTED_METRICS:
            raise QueryEngineError(
                code="INVALID_METRIC",
                message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
            )

        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_desc = norm_sort == "desc"

        k_data = self.G.nodes[k_node]
        results = []
        for movie_node, _, _ in self.G.in_edges(k_node, data=True):
            m = self.G.nodes[movie_node]
            results.append({
                "movie_id": m.get("tmdb_id"),
                "title": m.get("title"),
                "release_year": m.get("release_year"),
                "vote_average": float(m.get("vote_average") or 0.0),
                "vote_count": int(m.get("vote_count") or 0),
                "popularity": float(m.get("popularity") or 0.0),
                "revenue": float(m.get("revenue")) if m.get("revenue") is not None else None,
                "budget": float(m.get("budget")) if m.get("budget") is not None else None,
                "keyword_name": k_data.get("name"),
            })

        def sort_key(item: Dict[str, Any]):
            val = item.get(metric)
            is_none = val is None
            num_val = -float(val) if is_desc else float(val) if not is_none else float("inf")
            return (
                is_none,
                num_val,
                -float(item.get("vote_average") or 0.0),
                -int(item.get("vote_count") or 0),
                item["title"].lower(),
                item["movie_id"],
            )

        results.sort(key=sort_key)
        return results[: min(limit, 200)]

    def get_movies_by_company(
        self,
        company_name_or_id: Union[str, int],
        metric: str = "revenue",
        sort_order: str = "desc",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find movies produced by a company with deterministic sorting."""
        c_node = self.resolve_company_node(company_name_or_id)
        if not c_node:
            raise QueryEngineError(
                code="COMPANY_NOT_FOUND",
                message=f"Production company '{company_name_or_id}' not found in Knowledge Graph.",
            )

        if metric not in SUPPORTED_METRICS:
            raise QueryEngineError(
                code="INVALID_METRIC",
                message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
            )

        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_desc = norm_sort == "desc"

        c_data = self.G.nodes[c_node]
        results = []
        for movie_node, _, _ in self.G.in_edges(c_node, data=True):
            m = self.G.nodes[movie_node]
            results.append({
                "movie_id": m.get("tmdb_id"),
                "title": m.get("title"),
                "release_year": m.get("release_year"),
                "revenue": float(m.get("revenue")) if m.get("revenue") is not None else None,
                "budget": float(m.get("budget")) if m.get("budget") is not None else None,
                "vote_average": float(m.get("vote_average") or 0.0),
                "vote_count": int(m.get("vote_count") or 0),
                "popularity": float(m.get("popularity") or 0.0),
                "company_name": c_data.get("name"),
            })

        def sort_key(item: Dict[str, Any]):
            val = item.get(metric)
            is_none = val is None
            num_val = -float(val) if is_desc else float(val) if not is_none else float("inf")
            return (
                is_none,
                num_val,
                -float(item.get("vote_average") or 0.0),
                -int(item.get("vote_count") or 0),
                item["title"].lower(),
                item["movie_id"],
            )

        results.sort(key=sort_key)
        return results[: min(limit, 200)]

    # -------------------------------------------------------------------------
    # Ranking & Analytical Operations
    # -------------------------------------------------------------------------
    def get_top_movies(
        self,
        metric: str = "vote_average",
        limit: int = 10,
        min_vote_count: int = 0,
        sort_order: str = "desc",
        genre: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top ranked movies by metric with vote thresholds and genre/year filters."""
        if metric not in SUPPORTED_METRICS:
            raise QueryEngineError(
                code="INVALID_METRIC",
                message=f"Unsupported metric '{metric}'. Allowed: {sorted(SUPPORTED_METRICS)}",
            )

        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_desc = norm_sort == "desc"

        df = self.movies_df.copy()

        # Filter out NaN/None/zero for financial/numerical metrics
        if metric in ("budget", "revenue"):
            df = df[df[metric].notna() & (df[metric] > 0)]
        elif metric in ("vote_average", "popularity", "runtime", "vote_count"):
            df = df[df[metric].notna()]
            if metric == "vote_average" and min_vote_count > 0:
                pass  # threshold applied below
            elif metric in ("popularity", "runtime"):
                df = df[df[metric] > 0]

        # Apply min_vote_count threshold
        if min_vote_count > 0:
            df = df[df["vote_count"] >= min_vote_count]

        # Apply release year filters
        if year_start is not None:
            df = df[df["release_year"] >= year_start]
        if year_end is not None:
            df = df[df["release_year"] <= year_end]

        # Apply genre filter if requested
        if genre:
            genre_norm = genre.strip().lower()
            matching_nodes = set()
            for m_node, g_list in self.movie_genres_map.items():
                if any(g.lower() == genre_norm for g in g_list):
                    matching_nodes.add(m_node)
            df = df[df["node_id"].isin(matching_nodes)]

        if df.empty:
            return []

        # Deterministic sort: metric according to sort_order, then vote_count desc, title asc, tmdb_id asc
        df = df.sort_values(
            by=[metric, "vote_count", "title", "tmdb_id"],
            ascending=[not is_desc, False, True, True],
        )

        limit = min(max(1, limit), 100)
        top_df = df.head(limit)

        results = []
        for _, row in top_df.iterrows():
            results.append({
                "tmdb_id": int(row["tmdb_id"]),
                "title": row["title"],
                "release_year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
                metric: float(row[metric]) if metric in ("vote_average", "popularity", "revenue", "budget", "runtime") else int(row[metric]),
                "vote_average": float(row["vote_average"]),
                "vote_count": int(row["vote_count"]),
                "popularity": float(row["popularity"]),
                "revenue": float(row["revenue"]) if pd.notna(row["revenue"]) else None,
                "budget": float(row["budget"]) if pd.notna(row["budget"]) else None,
            })
        return results

    def compare_movies(self, movie_identifiers: List[Union[str, int]]) -> Dict[str, Any]:
        """Compare two or more movies side-by-side across all key metrics and cast/directors."""
        if len(movie_identifiers) < 2:
            raise QueryEngineError(
                code="INVALID_PARAMETER",
                message="compare_movies requires at least 2 movies to compare.",
            )

        movies_data = []
        for ident in movie_identifiers:
            details = self.get_movie_details(ident)
            if details.get("ambiguous"):
                details = details["movies"][0]
            movies_data.append(details)

        comparison_table = []
        for m in movies_data:
            genres_str = ", ".join(g["name"] for g in m.get("genres", []))
            dirs_str = ", ".join(d["name"] for d in m.get("directors", []))
            top_cast_str = ", ".join(c["name"] for c in m.get("top_cast_preview", [])[:4])

            comparison_table.append({
                "tmdb_id": m.get("tmdb_id"),
                "title": m.get("title"),
                "release_year": m.get("release_year"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "popularity": m.get("popularity"),
                "budget": m.get("budget"),
                "revenue": m.get("revenue"),
                "runtime": m.get("runtime"),
                "genres": genres_str,
                "directors": dirs_str,
                "top_cast": top_cast_str,
            })

        return {
            "comparison_count": len(comparison_table),
            "movies": comparison_table,
        }

    def get_common_cast(self, movie_identifiers: List[Union[str, int]]) -> List[Dict[str, Any]]:
        """Find actors who appeared in ALL of the specified movies."""
        if len(movie_identifiers) < 2:
            raise QueryEngineError(
                code="INVALID_PARAMETER",
                message="common_cast requires at least 2 movies.",
            )

        movie_node_sets = []
        resolved_movies = []

        for ident in movie_identifiers:
            node_ids = self.resolve_movie_nodes(ident)
            if not node_ids:
                raise QueryEngineError(
                    code="MOVIE_NOT_FOUND",
                    message=f"Movie '{ident}' not found in Knowledge Graph.",
                )
            m_node = node_ids[0]
            resolved_movies.append(m_node)

            movie_cast = {}
            for _, p_node, edge_data in self.G.out_edges(m_node, data=True):
                if edge_data.get("relationship") == REL_HAS_CAST:
                    movie_cast[p_node] = edge_data.get("character")
            movie_node_sets.append(movie_cast)

        common_person_nodes = set(movie_node_sets[0].keys())
        for m_cast in movie_node_sets[1:]:
            common_person_nodes.intersection_update(m_cast.keys())

        results = []
        for p_node in sorted(common_person_nodes, key=lambda p: self.G.nodes[p].get("name", "")):
            p_data = self.G.nodes[p_node]
            roles = {}
            for idx, m_node in enumerate(resolved_movies):
                m_title = self.G.nodes[m_node].get("title")
                roles[m_title] = movie_node_sets[idx].get(p_node)

            results.append({
                "person_id": p_data.get("tmdb_id"),
                "person_name": p_data.get("name"),
                "roles_by_movie": roles,
            })

        return results

    def aggregate_by_genre(
        self,
        metric: str = "vote_average",
        min_vote_count: int = 1,
        return_metadata: bool = False,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Calculate analytical statistics (count, mean, min, max) grouped by genre.

        By default (min_vote_count=1), movies with 0 recorded votes are excluded
        to prevent placeholder 0.0 ratings from artificially skewing genre minimums.
        If min_vote_count=0 is passed, all movies are included.
        """
        if metric not in ("vote_average", "popularity", "revenue", "budget"):
            raise QueryEngineError(
                code="INVALID_METRIC",
                message=f"Unsupported metric '{metric}' for genre aggregation.",
            )

        genre_records = []
        total_included_set = set()
        total_excluded_set = set()

        for g_name, g_node in self.genre_name_to_id.items():
            g_data = self.G.nodes[g_node]
            actual_name = g_data.get("name")
            values = []
            excluded_in_genre = 0

            for m_node, _, _ in self.G.in_edges(g_node, data=True):
                m = self.G.nodes[m_node]
                val = m.get(metric)
                vc = int(m.get("vote_count", 0) or 0)

                # Check vote count threshold
                if min_vote_count > 0 and vc < min_vote_count:
                    excluded_in_genre += 1
                    total_excluded_set.add(m_node)
                    continue

                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    if metric in ("budget", "revenue") and val <= 0:
                        excluded_in_genre += 1
                        total_excluded_set.add(m_node)
                        continue
                    values.append(float(val))
                    total_included_set.add(m_node)
                else:
                    excluded_in_genre += 1
                    total_excluded_set.add(m_node)

            if values:
                genre_records.append({
                    "genre_id": g_data.get("id"),
                    "genre_name": actual_name,
                    "movie_count": len(values),
                    "excluded_movie_count": excluded_in_genre,
                    f"avg_{metric}": round(float(np.mean(values)), 2),
                    f"min_{metric}": round(float(np.min(values)), 2),
                    f"max_{metric}": round(float(np.max(values)), 2),
                })

        sort_key = f"avg_{metric}"
        genre_records.sort(key=lambda x: (-x[sort_key], -x["movie_count"], x["genre_name"]))

        meta = {
            "metric": metric,
            "min_vote_count": min_vote_count,
            "total_movies_in_dataset": len(self.movies_df),
            "total_included_movies": len(total_included_set),
            "total_excluded_movies": len(total_excluded_set - total_included_set),
            "aggregation_methodology": (
                f"Aggregated by genre using movies with vote_count >= {min_vote_count}. "
                f"Movies with fewer than {min_vote_count} votes were excluded to eliminate unrated placeholder ratings."
                if min_vote_count > 0
                else "Aggregated across all movies including zero-vote records."
            ),
        }

        if return_metadata:
            return genre_records, meta
        return genre_records

    def get_movies_by_year_range(
        self,
        year_start: int,
        year_end: int,
        sort_order: str = "asc",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find movies released within a given year range in strictly sorted order.

        Defaults to ascending chronological order ('asc') so earlier years appear first.
        Supports 'desc' for reverse chronological order.
        """
        norm_sort = _normalize_sort_order(sort_order, default="asc")
        is_asc = norm_sort == "asc"

        df = self.movies_df.copy()
        df = df[df["release_year"].notna()]
        filtered = df[(df["release_year"] >= year_start) & (df["release_year"] <= year_end)].copy()

        # Sort consistently by release_date (or release_year if date missing)
        # For asc: earlier date first, tiebreak on vote_average desc, title asc
        # For desc: later date first, tiebreak on vote_average desc, title asc
        filtered["sort_date"] = filtered["release_date"].fillna(
            filtered["release_year"].apply(lambda y: f"{int(y)}-12-31" if pd.notna(y) else "")
        )

        filtered = filtered.sort_values(
            by=["sort_date", "vote_average", "vote_count", "title", "tmdb_id"],
            ascending=[is_asc, False, False, True, True],
            na_position="last",
        )

        limit = min(max(1, limit), 200)
        results = []
        for _, row in filtered.head(limit).iterrows():
            results.append({
                "tmdb_id": int(row["tmdb_id"]),
                "title": row["title"],
                "release_year": int(row["release_year"]),
                "release_date": str(row["release_date"]) if pd.notna(row["release_date"]) else None,
                "vote_average": float(row["vote_average"]),
                "vote_count": int(row["vote_count"]),
                "popularity": float(row["popularity"]),
            })
        return results

    def count_movies_by_year(
        self,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Count movies released per year with configurable sort order."""
        norm_sort = _normalize_sort_order(sort_order, default="desc")
        is_asc = norm_sort == "asc"

        df = self.movies_df[self.movies_df["release_year"].notna()].copy()
        if year_start is not None:
            df = df[df["release_year"] >= year_start]
        if year_end is not None:
            df = df[df["release_year"] <= year_end]

        counts = df["release_year"].value_counts().sort_index(ascending=is_asc)
        return [{"year": int(year), "movie_count": int(count)} for year, count in counts.items()]

    # -------------------------------------------------------------------------
    # Structured Execution Entry Point
    # -------------------------------------------------------------------------
    def execute_query(self, query_spec: Union[Dict[str, Any], QuerySpecification]) -> Dict[str, Any]:
        """Execute a controlled structured query specification safely."""
        try:
            if isinstance(query_spec, dict):
                spec = QuerySpecification(**query_spec)
            else:
                spec = query_spec
            spec.validate_spec()
        except ValidationError as val_err:
            return {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(val_err),
                },
            }
        except QueryEngineError as q_err:
            return {
                "success": False,
                "error": {
                    "code": q_err.code,
                    "message": q_err.message,
                },
            }

        op = spec.operation.strip().lower()
        p = spec.parameters

        try:
            if op == "movie_details":
                ident = p.get("movie_id") or p.get("title") or p.get("query")
                res = self.get_movie_details(ident)
                count = res.get("count", 1) if isinstance(res, dict) and res.get("ambiguous") else 1
                return {
                    "success": True,
                    "operation": op,
                    "result_count": count,
                    "results": res,
                    "metadata": {"identifier": ident},
                }

            elif op == "search_movies":
                query_str = p.get("query") or p.get("title")
                limit = p.get("limit", 10)
                res = self.search_movies(query_str, limit=limit)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"query": query_str, "limit": limit},
                }

            elif op == "movie_cast":
                ident = p.get("movie_id") or p.get("title")
                limit = p.get("limit")
                res = self.get_movie_cast(ident, limit=limit)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"movie": ident, "limit": limit},
                }

            elif op == "movie_crew":
                ident = p.get("movie_id") or p.get("title")
                dep = p.get("department")
                job = p.get("job")
                limit = p.get("limit")
                res = self.get_movie_crew(ident, department=dep, job=job, limit=limit)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"movie": ident, "department": dep, "job": job, "limit": limit},
                }

            elif op == "movie_directors":
                ident = p.get("movie_id") or p.get("title")
                res = self.get_movie_directors(ident)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"movie": ident},
                }

            elif op == "movies_by_person":
                ident = p.get("person_name") or p.get("person_id")
                role = p.get("role")
                limit = p.get("limit", 50)
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                res = self.get_movies_by_person(ident, role=role, limit=limit, sort_order=sort_order)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"person": ident, "role": role, "sort_order": sort_order, "limit": limit},
                }

            elif op == "movies_by_genre":
                ident = p.get("genre_name") or p.get("genre_id")
                limit = p.get("limit", 50)
                metric = p.get("metric", "vote_average")
                min_votes = p.get("min_vote_count", 0)
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                res = self.get_movies_by_genre(
                    ident,
                    metric=metric,
                    min_vote_count=min_votes,
                    sort_order=sort_order,
                    limit=limit,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {
                        "genre": ident,
                        "metric": metric,
                        "min_vote_count": min_votes,
                        "sort_order": sort_order,
                        "limit": limit,
                    },
                }

            elif op == "movies_by_keyword":
                ident = p.get("keyword_name") or p.get("keyword_id")
                limit = p.get("limit", 50)
                metric = p.get("metric", "popularity")
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                res = self.get_movies_by_keyword(
                    ident,
                    metric=metric,
                    sort_order=sort_order,
                    limit=limit,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {
                        "keyword": ident,
                        "metric": metric,
                        "sort_order": sort_order,
                        "limit": limit,
                    },
                }

            elif op == "movies_by_company":
                ident = p.get("company_name") or p.get("company_id")
                limit = p.get("limit", 50)
                metric = p.get("metric", "revenue")
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                res = self.get_movies_by_company(
                    ident,
                    metric=metric,
                    sort_order=sort_order,
                    limit=limit,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {
                        "company": ident,
                        "metric": metric,
                        "sort_order": sort_order,
                        "limit": limit,
                    },
                }

            elif op == "top_movies":
                metric = p.get("metric", "vote_average")
                limit = p.get("limit", 10)
                min_votes = p.get("min_vote_count", 0)
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                genre = p.get("genre")
                y_start = p.get("year_start")
                y_end = p.get("year_end")
                res = self.get_top_movies(
                    metric=metric,
                    limit=limit,
                    min_vote_count=min_votes,
                    sort_order=sort_order,
                    genre=genre,
                    year_start=y_start,
                    year_end=y_end,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {
                        "metric": metric,
                        "sort_order": sort_order,
                        "limit": limit,
                        "min_vote_count": min_votes,
                        "genre": genre,
                    },
                }

            elif op == "compare_movies":
                movies = p.get("movies") or p.get("movie_titles") or p.get("movie_ids")
                res = self.compare_movies(movies)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": res.get("comparison_count", 0),
                    "results": res.get("movies", []),
                    "metadata": {"compared_movies": movies},
                }

            elif op == "common_cast":
                movies = p.get("movies") or p.get("movie_titles") or p.get("movie_ids")
                res = self.get_common_cast(movies)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"movies": movies},
                }

            elif op == "aggregate_by_genre":
                metric = p.get("metric", "vote_average")
                min_votes = int(p.get("min_vote_count", 1))
                res, agg_meta = self.aggregate_by_genre(
                    metric=metric,
                    min_vote_count=min_votes,
                    return_metadata=True,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": agg_meta,
                }

            elif op == "movies_by_year_range":
                y_start = int(p["year_start"])
                y_end = int(p["year_end"])
                sort_order = _normalize_sort_order(p.get("sort_order"), default="asc")
                limit = p.get("limit", 50)
                res = self.get_movies_by_year_range(
                    y_start,
                    y_end,
                    sort_order=sort_order,
                    limit=limit,
                )
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {
                        "year_start": y_start,
                        "year_end": y_end,
                        "sort_order": sort_order,
                        "limit": limit,
                    },
                }

            elif op == "count_by_year":
                y_start = p.get("year_start")
                y_end = p.get("year_end")
                sort_order = _normalize_sort_order(p.get("sort_order"), default="desc")
                res = self.count_movies_by_year(y_start, y_end, sort_order=sort_order)
                return {
                    "success": True,
                    "operation": op,
                    "result_count": len(res),
                    "results": res,
                    "metadata": {"year_start": y_start, "year_end": y_end, "sort_order": sort_order},
                }

            else:
                return {
                    "success": False,
                    "error": {
                        "code": "INVALID_OPERATION",
                        "message": f"Unsupported operation '{op}'",
                    },
                }

        except QueryEngineError as q_err:
            return {
                "success": False,
                "error": {
                    "code": q_err.code,
                    "message": q_err.message,
                },
            }
        except Exception as exc:
            logger.exception(f"Unexpected error executing query {op}: {exc}")
            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_EXECUTION_ERROR",
                    "message": str(exc),
                },
            }


# -----------------------------------------------------------------------------
# Module Singleton Management
# -----------------------------------------------------------------------------
_ENGINE_INSTANCE: Optional[GraphQueryEngine] = None


def load_query_engine(
    graph_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> GraphQueryEngine:
    """Load or retrieve the cached singleton instance of GraphQueryEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None or force_reload:
        path = graph_path or GRAPH_PICKLE_PATH
        logger.info(f"Initializing GraphQueryEngine from: {path}")
        G = load_graph(path)
        _ENGINE_INSTANCE = GraphQueryEngine(G)
    return _ENGINE_INSTANCE


def execute_query(query_spec: Union[Dict[str, Any], QuerySpecification]) -> Dict[str, Any]:
    """Execute a query specification using the global loaded query engine."""
    engine = load_query_engine()
    return engine.execute_query(query_spec)
