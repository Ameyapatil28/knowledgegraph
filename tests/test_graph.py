"""Unit and integration tests for Knowledge Graph construction and persistence."""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.graph_builder import (
    build_graph,
    save_graph,
    load_graph,
    get_graph_statistics,
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
from src.config import GRAPH_PICKLE_PATH


@pytest.fixture
def synthetic_movie_df():
    """Create a realistic mini dataset covering all entity types and edge cases."""
    records = [
        {
            "tmdb_id": 101,
            "title": "Inception",
            "original_title": "Inception",
            "disambiguated_title": "Inception",
            "release_date": "2010-07-16",
            "release_year": 2010,
            "runtime": 148.0,
            "budget": 160000000.0,
            "revenue": 825532764.0,
            "vote_average": 8.3,
            "vote_count": 14000,
            "popularity": 167.5,
            "overview": "Cobb steals information from subconscious minds.",
            "original_language": "en",
            # Cast: Leonardo DiCaprio (id 6193), Joseph Gordon-Levitt (id 24045)
            "cast": [
                {"id": 6193, "name": "Leonardo DiCaprio", "character": "Dom Cobb", "order": 0, "gender": 2, "credit_id": "c1"},
                {"id": 24045, "name": "Joseph Gordon-Levitt", "character": "Arthur", "order": 1, "gender": 2, "credit_id": "c2"},
            ],
            # Crew: Christopher Nolan (id 525) is both Director and Producer
            "crew": [
                {"id": 525, "name": "Christopher Nolan", "job": "Director", "department": "Directing", "gender": 2, "credit_id": "cr1"},
                {"id": 525, "name": "Christopher Nolan", "job": "Producer", "department": "Production", "gender": 2, "credit_id": "cr2"},
                {"id": 947, "name": "Hans Zimmer", "job": "Original Music Composer", "department": "Sound", "gender": 2, "credit_id": "cr3"},
            ],
            "genres": [
                {"id": 28, "name": "Action"},
                {"id": 878, "name": "Science Fiction"},
            ],
            "keywords": [
                {"id": 1009, "name": "dream"},
                {"id": 1563, "name": "subconscious"},
            ],
            "production_companies": [
                {"id": 923, "name": "Legendary Pictures"},
                {"id": 9996, "name": "Syncopy"},
            ],
            "production_countries": [
                {"iso_3166_1": "US", "name": "United States of America"},
                {"iso_3166_1": "GB", "name": "United Kingdom"},
            ],
            "spoken_languages": [
                {"iso_639_1": "en", "name": "English"},
                {"iso_639_1": "ja", "name": "日本語"},
            ],
        },
        {
            # Edge case movie: Quentin Tarantino is Director, Writer, and Actor
            "tmdb_id": 102,
            "title": "Pulp Fiction",
            "original_title": "Pulp Fiction",
            "disambiguated_title": "Pulp Fiction",
            "release_date": "1994-09-10",
            "release_year": 1994,
            "runtime": 154.0,
            "budget": 8000000.0,
            "revenue": 213928762.0,
            "vote_average": 8.5,
            "vote_count": 8428,
            "popularity": 121.4,
            "overview": "Stories of crime in Los Angeles.",
            "original_language": "en",
            "cast": [
                {"id": 138, "name": "Quentin Tarantino", "character": "Jimmie Dimmick", "order": 8, "gender": 2, "credit_id": "c3"},
                # Missing / invalid person id test
                {"id": None, "name": "Unknown Extra", "character": "Background", "order": 99, "gender": 0},
            ],
            "crew": [
                {"id": 138, "name": "Quentin Tarantino", "job": "Director", "department": "Directing", "gender": 2, "credit_id": "cr4"},
                {"id": 138, "name": "Quentin Tarantino", "job": "Writer", "department": "Writing", "gender": 2, "credit_id": "cr5"},
            ],
            "genres": [
                {"id": 53, "name": "Thriller"},
                {"id": 80, "name": "Crime"},
            ],
            "keywords": [
                {"id": 1009, "name": "dream"},  # Shared keyword with movie 101
            ],
            "production_companies": [
                {"id": 14, "name": "Miramax Films"},
            ],
            "production_countries": [
                {"iso_3166_1": "US", "name": "United States of America"},
            ],
            "spoken_languages": [
                {"iso_639_1": "en", "name": "English"},
                {"iso_639_1": "fr", "name": "Français"},
            ],
        },
        {
            # Movie with missing budget and revenue (unrecorded)
            "tmdb_id": 103,
            "title": "Indie Film",
            "original_title": "Indie Film",
            "disambiguated_title": "Indie Film",
            "release_date": None,
            "release_year": None,
            "runtime": None,
            "budget": None,
            "revenue": None,
            "vote_average": 0.0,
            "vote_count": 0,
            "popularity": 0.1,
            "overview": "",
            "original_language": "en",
            "cast": [],
            "crew": [],
            "genres": [],
            "keywords": [],
            "production_companies": [],
            "production_countries": [],
            "spoken_languages": [],
        },
    ]
    return pd.DataFrame(records)


class TestGraphConstruction:
    def test_movie_node_creation(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_node_id = f"{PREFIX_MOVIE}:101"

        assert movie_node_id in G
        data = G.nodes[movie_node_id]
        assert data["type"] == "Movie"
        assert data["title"] == "Inception"
        assert data["release_year"] == 2010
        assert data["budget"] == 160000000.0
        assert data["revenue"] == 825532764.0
        assert data["vote_average"] == 8.3

        # Test unrecorded budget/revenue preserves None
        indie_node = G.nodes[f"{PREFIX_MOVIE}:103"]
        assert indie_node["budget"] is None
        assert indie_node["revenue"] is None

    def test_person_node_creation(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        person_node_id = f"{PREFIX_PERSON}:6193"

        assert person_node_id in G
        data = G.nodes[person_node_id]
        assert data["type"] == "Person"
        assert data["name"] == "Leonardo DiCaprio"
        assert data["gender"] == 2

    def test_actor_relationship(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        actor_id = f"{PREFIX_PERSON}:6193"

        assert G.has_edge(movie_id, actor_id)
        edges = G.get_edge_data(movie_id, actor_id)
        assert any(e.get("relationship") == REL_HAS_CAST for e in edges.values())

        # Check edge attributes
        edge_attr = next(e for e in edges.values() if e.get("relationship") == REL_HAS_CAST)
        assert edge_attr["character"] == "Dom Cobb"
        assert edge_attr["order"] == 0

    def test_director_relationship(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        dir_id = f"{PREFIX_PERSON}:525"

        edges = G.get_edge_data(movie_id, dir_id)
        assert any(e.get("relationship") == REL_DIRECTED_BY for e in edges.values())

        dir_edge = next(e for e in edges.values() if e.get("relationship") == REL_DIRECTED_BY)
        assert dir_edge["job"] == "Director"
        assert dir_edge["department"] == "Directing"

    def test_crew_relationship(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        composer_id = f"{PREFIX_PERSON}:947"

        edges = G.get_edge_data(movie_id, composer_id)
        assert any(e.get("relationship") == REL_HAS_CREW for e in edges.values())
        crew_edge = next(e for e in edges.values() if e.get("relationship") == REL_HAS_CREW)
        assert crew_edge["job"] == "Original Music Composer"

    def test_genre_relationships(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        genre_id = f"{PREFIX_GENRE}:28"

        assert genre_id in G
        assert G.nodes[genre_id]["name"] == "Action"
        assert G.has_edge(movie_id, genre_id)
        edge = next(iter(G.get_edge_data(movie_id, genre_id).values()))
        assert edge["relationship"] == REL_HAS_GENRE

    def test_keyword_relationships(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie1_id = f"{PREFIX_MOVIE}:101"
        movie2_id = f"{PREFIX_MOVIE}:102"
        kw_id = f"{PREFIX_KEYWORD}:1009"

        assert kw_id in G
        assert G.nodes[kw_id]["name"] == "dream"
        # Shared keyword connected to both movies
        assert G.has_edge(movie1_id, kw_id)
        assert G.has_edge(movie2_id, kw_id)

    def test_company_relationships(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        comp_id = f"{PREFIX_COMPANY}:923"

        assert comp_id in G
        assert G.nodes[comp_id]["name"] == "Legendary Pictures"
        assert G.has_edge(movie_id, comp_id)
        edge = next(iter(G.get_edge_data(movie_id, comp_id).values()))
        assert edge["relationship"] == REL_PRODUCED_BY

    def test_country_relationships(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        country_id = f"{PREFIX_COUNTRY}:US"

        assert country_id in G
        assert G.nodes[country_id]["code"] == "US"
        assert G.has_edge(movie_id, country_id)
        edge = next(iter(G.get_edge_data(movie_id, country_id).values()))
        assert edge["relationship"] == REL_PRODUCED_IN

    def test_language_relationships(self, synthetic_movie_df):
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:101"
        lang_id = f"{PREFIX_LANGUAGE}:en"

        assert lang_id in G
        assert G.nodes[lang_id]["code"] == "en"
        assert G.has_edge(movie_id, lang_id)
        edge = next(iter(G.get_edge_data(movie_id, lang_id).values()))
        assert edge["relationship"] == REL_HAS_LANGUAGE

    def test_actor_who_is_also_crew_member(self, synthetic_movie_df):
        """Quentin Tarantino in Pulp Fiction is Actor, Director, and Writer."""
        G, _ = build_graph(synthetic_movie_df)
        movie_id = f"{PREFIX_MOVIE}:102"
        qt_id = f"{PREFIX_PERSON}:138"

        # Exactly ONE node for Quentin Tarantino
        person_nodes = [n for n, d in G.nodes(data=True) if n == qt_id]
        assert len(person_nodes) == 1
        assert G.nodes[qt_id]["name"] == "Quentin Tarantino"

        # Check all 3 distinct edge relationships
        edges = G.get_edge_data(movie_id, qt_id)
        rel_types = {e.get("relationship") for e in edges.values()}
        assert REL_HAS_CAST in rel_types
        assert REL_DIRECTED_BY in rel_types
        assert REL_HAS_CREW in rel_types

        # Verify writer job in crew
        crew_edge = next(e for e in edges.values() if e.get("relationship") == REL_HAS_CREW)
        assert crew_edge["job"] == "Writer"

    def test_missing_ids_are_skipped(self, synthetic_movie_df):
        """Ensure entity records without IDs are safely skipped."""
        G, stats = build_graph(synthetic_movie_df)
        # Extra with id None should not exist in the graph
        assert None not in G
        assert "person:None" not in G
        assert "person:0" not in G
        assert stats["skipped_entities"]["cast_missing_id"] >= 1

    def test_graph_statistics(self, synthetic_movie_df):
        G, stats = build_graph(synthetic_movie_df)
        assert stats["total_nodes"] == G.number_of_nodes()
        assert stats["total_edges"] == G.number_of_edges()
        assert stats["total_movies"] == 3
        assert stats["movies_with_cast"] == 2
        assert stats["movies_with_directors"] == 2
        assert stats["movies_with_genres"] == 2
        assert stats["isolated_nodes"] == 1  # Indie Film has no cast/crew/genres

    def test_graph_serialization_and_deserialization(self, synthetic_movie_df, tmp_path: Path):
        G, _ = build_graph(synthetic_movie_df)
        save_file = tmp_path / "test_graph.pickle"

        save_graph(G, save_file)
        assert save_file.exists()

        loaded_G = load_graph(save_file)
        assert loaded_G.number_of_nodes() == G.number_of_nodes()
        assert loaded_G.number_of_edges() == G.number_of_edges()

        # Check that attributes are preserved exactly
        movie_node = loaded_G.nodes[f"{PREFIX_MOVIE}:101"]
        assert movie_node["title"] == "Inception"
        assert movie_node["budget"] == 160000000.0

        indie_node = loaded_G.nodes[f"{PREFIX_MOVIE}:103"]
        assert indie_node["budget"] is None
        assert indie_node["revenue"] is None


class TestFullGraphArtifact:
    """Validate the actual full knowledge graph artifact persisted on disk."""

    def test_persisted_graph_file_exists(self):
        assert GRAPH_PICKLE_PATH.exists(), f"Graph file missing at {GRAPH_PICKLE_PATH}"

    def test_persisted_graph_integrity(self):
        G = load_graph(GRAPH_PICKLE_PATH)
        assert G.number_of_nodes() > 100000
        assert G.number_of_edges() > 250000

        # Avatar check (tmdb_id: 19995)
        avatar_node = G.nodes.get(f"{PREFIX_MOVIE}:19995")
        assert avatar_node is not None
        assert avatar_node["title"] == "Avatar"
        assert avatar_node["budget"] == 237000000.0

        # Check James Cameron directorship on Avatar
        cameron_id = f"{PREFIX_PERSON}:2710"
        assert cameron_id in G
        assert G.nodes[cameron_id]["name"] == "James Cameron"
        edges = G.get_edge_data(f"{PREFIX_MOVIE}:19995", cameron_id)
        assert any(e.get("relationship") == REL_DIRECTED_BY for e in edges.values())
