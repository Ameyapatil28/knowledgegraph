"""Comprehensive unit and integration tests for the Graph Query Engine."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.config import GRAPH_PICKLE_PATH
from src.graph_builder import (
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
from src.query_engine import (
    GraphQueryEngine,
    load_query_engine,
    execute_query,
    SUPPORTED_OPERATIONS,
    SUPPORTED_METRICS,
    SUPPORTED_ROLES,
)


@pytest.fixture
def synthetic_graph():
    """Build a deterministic synthetic MultiDiGraph with all edge cases."""
    G = nx.MultiDiGraph()

    # 1. Movie: Inception (ID: 101)
    G.add_node(
        f"{PREFIX_MOVIE}:101",
        type="Movie",
        tmdb_id=101,
        title="Inception",
        disambiguated_title="Inception",
        original_title="Inception",
        release_date="2010-07-16",
        release_year=2010,
        runtime=148.0,
        budget=160000000.0,
        revenue=825532764.0,
        vote_average=8.3,
        vote_count=14000,
        popularity=167.5,
        overview="Dream heist thriller.",
        original_language="en",
    )

    # 2. Movie: Pulp Fiction (ID: 102)
    G.add_node(
        f"{PREFIX_MOVIE}:102",
        type="Movie",
        tmdb_id=102,
        title="Pulp Fiction",
        disambiguated_title="Pulp Fiction",
        original_title="Pulp Fiction",
        release_date="1994-09-10",
        release_year=1994,
        runtime=154.0,
        budget=8000000.0,
        revenue=213928762.0,
        vote_average=8.5,
        vote_count=8428,
        popularity=121.4,
        overview="Crime in LA.",
        original_language="en",
    )

    # 3. Duplicate Movie Titles: Batman (1989, ID: 268) & Batman (1966, ID: 2661)
    G.add_node(
        f"{PREFIX_MOVIE}:268",
        type="Movie",
        tmdb_id=268,
        title="Batman",
        disambiguated_title="Batman (1989)",
        original_title="Batman",
        release_date="1989-06-23",
        release_year=1989,
        runtime=126.0,
        budget=35000000.0,
        revenue=411348924.0,
        vote_average=7.0,
        vote_count=2000,
        popularity=45.0,
        overview="1989 Dark Knight.",
        original_language="en",
    )
    G.add_node(
        f"{PREFIX_MOVIE}:2661",
        type="Movie",
        tmdb_id=2661,
        title="Batman",
        disambiguated_title="Batman (1966)",
        original_title="Batman",
        release_date="1966-07-30",
        release_year=1966,
        runtime=105.0,
        budget=1377800.0,
        revenue=None,  # Missing revenue
        vote_average=6.1,
        vote_count=400,
        popularity=12.0,
        overview="1966 Campy Batman.",
        original_language="en",
    )

    # 4. Movie: Indie Film with unrecorded budget and revenue (None)
    G.add_node(
        f"{PREFIX_MOVIE}:999",
        type="Movie",
        tmdb_id=999,
        title="Indie Film",
        disambiguated_title="Indie Film",
        original_title="Indie Film",
        release_date="2021-01-01",
        release_year=2021,
        runtime=90.0,
        budget=None,
        revenue=None,
        vote_average=7.5,
        vote_count=10,  # low vote count
        popularity=2.0,
        overview="No budget film.",
        original_language="en",
    )

    # People Nodes
    # Person 1: Leonardo DiCaprio (ID: 6193)
    G.add_node(f"{PREFIX_PERSON}:6193", type="Person", tmdb_id=6193, name="Leonardo DiCaprio", gender=2)
    # Person 2: Christopher Nolan (ID: 525) - Director & Producer
    G.add_node(f"{PREFIX_PERSON}:525", type="Person", tmdb_id=525, name="Christopher Nolan", gender=2)
    # Person 3: Quentin Tarantino (ID: 138) - Multi-role: Actor, Director, Writer
    G.add_node(f"{PREFIX_PERSON}:138", type="Person", tmdb_id=138, name="Quentin Tarantino", gender=2)
    # Person 4: Michael Keaton (ID: 2232) - Actor in Batman 1989
    G.add_node(f"{PREFIX_PERSON}:2232", type="Person", tmdb_id=2232, name="Michael Keaton", gender=2)
    # Person 5 & 6: Two distinct people with the same name "John Smith" (different IDs)
    G.add_node(f"{PREFIX_PERSON}:7001", type="Person", tmdb_id=7001, name="John Smith", gender=2)
    G.add_node(f"{PREFIX_PERSON}:7002", type="Person", tmdb_id=7002, name="John Smith", gender=2)
    # Person 7: Common actor between Inception & Batman 1989 (Jack Nicholson in Batman, cameo in Inception)
    G.add_node(f"{PREFIX_PERSON}:514", type="Person", tmdb_id=514, name="Jack Nicholson", gender=2)

    # Genre Nodes
    G.add_node(f"{PREFIX_GENRE}:28", type="Genre", id=28, name="Action")
    G.add_node(f"{PREFIX_GENRE}:878", type="Genre", id=878, name="Science Fiction")
    G.add_node(f"{PREFIX_GENRE}:80", type="Genre", id=80, name="Crime")

    # Keyword Nodes
    G.add_node(f"{PREFIX_KEYWORD}:1009", type="Keyword", id=1009, name="dream")
    G.add_node(f"{PREFIX_KEYWORD}:849", type="Keyword", id=849, name="dc comics")

    # Company Nodes
    G.add_node(f"{PREFIX_COMPANY}:923", type="ProductionCompany", id=923, name="Legendary Pictures")
    G.add_node(f"{PREFIX_COMPANY}:6194", type="ProductionCompany", id=6194, name="Warner Bros")

    # EDGES
    # Inception Edges
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_PERSON}:6193", key="c1", relationship=REL_HAS_CAST, character="Dom Cobb", order=0)
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_PERSON}:514", key="c2", relationship=REL_HAS_CAST, character="Cameo", order=20)
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_PERSON}:525", key="d1", relationship=REL_DIRECTED_BY, job="Director", department="Directing")
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_PERSON}:525", key="cr1", relationship=REL_HAS_CREW, job="Producer", department="Production")
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_GENRE}:28", key="g1", relationship=REL_HAS_GENRE)
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_GENRE}:878", key="g2", relationship=REL_HAS_GENRE)
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_KEYWORD}:1009", key="k1", relationship=REL_HAS_KEYWORD)
    G.add_edge(f"{PREFIX_MOVIE}:101", f"{PREFIX_COMPANY}:923", key="co1", relationship=REL_PRODUCED_BY)

    # Pulp Fiction Edges
    G.add_edge(f"{PREFIX_MOVIE}:102", f"{PREFIX_PERSON}:138", key="c3", relationship=REL_HAS_CAST, character="Jimmie", order=8)
    G.add_edge(f"{PREFIX_MOVIE}:102", f"{PREFIX_PERSON}:138", key="d2", relationship=REL_DIRECTED_BY, job="Director", department="Directing")
    G.add_edge(f"{PREFIX_MOVIE}:102", f"{PREFIX_PERSON}:138", key="cr2", relationship=REL_HAS_CREW, job="Writer", department="Writing")
    G.add_edge(f"{PREFIX_MOVIE}:102", f"{PREFIX_GENRE}:80", key="g3", relationship=REL_HAS_GENRE)

    # Batman 1989 Edges
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_PERSON}:2232", key="c4", relationship=REL_HAS_CAST, character="Bruce Wayne", order=0)
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_PERSON}:514", key="c5", relationship=REL_HAS_CAST, character="Joker", order=1)
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_GENRE}:28", key="g4", relationship=REL_HAS_GENRE)
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_KEYWORD}:849", key="k2", relationship=REL_HAS_KEYWORD)
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_COMPANY}:6194", key="co2", relationship=REL_PRODUCED_BY)
    G.add_edge(f"{PREFIX_MOVIE}:268", f"{PREFIX_PERSON}:7001", key="cr3", relationship=REL_HAS_CREW, job="Sound Mixer", department="Sound")

    # Batman 1966 Edges
    G.add_edge(f"{PREFIX_MOVIE}:2661", f"{PREFIX_GENRE}:28", key="g5", relationship=REL_HAS_GENRE)
    G.add_edge(f"{PREFIX_MOVIE}:2661", f"{PREFIX_KEYWORD}:849", key="k3", relationship=REL_HAS_KEYWORD)
    G.add_edge(f"{PREFIX_MOVIE}:2661", f"{PREFIX_PERSON}:7002", key="cr4", relationship=REL_HAS_CREW, job="Electrician", department="Lighting")

    return G


@pytest.fixture
def query_engine(synthetic_graph):
    return GraphQueryEngine(synthetic_graph)


class TestQueryOperations:
    def test_movie_details_by_id(self, query_engine):
        spec = {"operation": "movie_details", "parameters": {"movie_id": 101}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        movie = res["results"]
        assert movie["title"] == "Inception"
        assert movie["release_year"] == 2010
        assert len(movie["directors"]) == 1
        assert movie["directors"][0]["name"] == "Christopher Nolan"
        assert any(g["name"] == "Science Fiction" for g in movie["genres"])

    def test_search_movies(self, query_engine):
        spec = {"operation": "search_movies", "parameters": {"query": "incept"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["title"] == "Inception"

    def test_movie_cast(self, query_engine):
        spec = {"operation": "movie_cast", "parameters": {"title": "Inception"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 2
        cast = res["results"]
        assert cast[0]["person_name"] == "Leonardo DiCaprio"
        assert cast[0]["character"] == "Dom Cobb"

    def test_movie_crew(self, query_engine):
        spec = {"operation": "movie_crew", "parameters": {"title": "Inception", "job": "Producer"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["person_name"] == "Christopher Nolan"

    def test_movie_directors(self, query_engine):
        spec = {"operation": "movie_directors", "parameters": {"title": "Inception"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["person_name"] == "Christopher Nolan"

    def test_movies_by_person_actor(self, query_engine):
        spec = {"operation": "movies_by_person", "parameters": {"person_name": "Leonardo DiCaprio", "role": "actor"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["movie_title"] == "Inception"

    def test_movies_by_person_director(self, query_engine):
        spec = {"operation": "movies_by_person", "parameters": {"person_name": "Christopher Nolan", "role": "director"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["movie_title"] == "Inception"

    def test_movies_by_person_multiple_roles(self, query_engine):
        """Quentin Tarantino in Pulp Fiction is Actor, Director, and Writer."""
        # Query role='director'
        res_dir = query_engine.execute_query({
            "operation": "movies_by_person",
            "parameters": {"person_name": "Quentin Tarantino", "role": "director"},
        })
        assert res_dir["result_count"] == 1
        assert res_dir["results"][0]["relationship"] == "DIRECTED_BY"

        # Query role='writer'
        res_writer = query_engine.execute_query({
            "operation": "movies_by_person",
            "parameters": {"person_name": "Quentin Tarantino", "role": "writer"},
        })
        assert res_writer["result_count"] == 1
        assert res_writer["results"][0]["job"] == "Writer"

        # Query role='actor'
        res_actor = query_engine.execute_query({
            "operation": "movies_by_person",
            "parameters": {"person_name": "Quentin Tarantino", "role": "actor"},
        })
        assert res_actor["result_count"] == 1
        assert res_actor["results"][0]["relationship"] == "HAS_CAST"

    def test_multiple_people_same_name(self, query_engine):
        """Two distinct people named 'John Smith' with IDs 7001 and 7002."""
        spec = {"operation": "movies_by_person", "parameters": {"person_name": "John Smith"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        # Both records should be returned (one from Batman 1989, one from Batman 1966)
        assert res["result_count"] == 2
        p_ids = {r["person_id"] for r in res["results"]}
        assert 7001 in p_ids
        assert 7002 in p_ids

    def test_movie_title_ambiguity(self, query_engine):
        """'Batman' matches both 1989 and 1966."""
        spec = {"operation": "movie_details", "parameters": {"title": "Batman"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        data = res["results"]
        assert data.get("ambiguous") is True
        assert data.get("count") == 2

        # Resolving via disambiguated title resolves uniquely
        spec_dis = {"operation": "movie_details", "parameters": {"title": "Batman (1989)"}}
        res_dis = query_engine.execute_query(spec_dis)
        assert res_dis["results"]["tmdb_id"] == 268

    def test_movies_by_genre(self, query_engine):
        spec = {"operation": "movies_by_genre", "parameters": {"genre_name": "Action"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        titles = {m["title"] for m in res["results"]}
        assert "Inception" in titles
        assert "Batman" in titles

    def test_movies_by_keyword(self, query_engine):
        spec = {"operation": "movies_by_keyword", "parameters": {"keyword_name": "dream"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["title"] == "Inception"

    def test_movies_by_company(self, query_engine):
        spec = {"operation": "movies_by_company", "parameters": {"company_name": "Warner Bros"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["title"] == "Batman"

    def test_top_movies_with_vote_threshold(self, query_engine):
        """Indie Film has rating 7.5 but only 10 votes. It should be excluded if min_vote_count=100."""
        spec = {"operation": "top_movies", "parameters": {"metric": "vote_average", "min_vote_count": 100}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        titles = [m["title"] for m in res["results"]]
        assert "Indie Film" not in titles
        assert titles[0] == "Pulp Fiction"  # 8.5 rating
        assert titles[1] == "Inception"     # 8.3 rating

    def test_top_movies_missing_revenue_handling(self, query_engine):
        """Batman 1966 and Indie Film have None revenue. They should not appear in highest-revenue list."""
        spec = {"operation": "top_movies", "parameters": {"metric": "revenue", "limit": 10}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        titles = [m["title"] for m in res["results"]]
        assert "Indie Film" not in titles
        assert titles[0] == "Inception"      # $825M
        assert titles[1] == "Batman"         # $411M (1989)
        assert titles[2] == "Pulp Fiction"   # $213M

    def test_compare_movies(self, query_engine):
        spec = {"operation": "compare_movies", "parameters": {"movies": ["Inception", "Pulp Fiction"]}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 2
        m1, m2 = res["results"]
        assert m1["title"] == "Inception"
        assert m2["title"] == "Pulp Fiction"

    def test_common_cast(self, query_engine):
        """Jack Nicholson is in both Inception (cameo) and Batman 1989."""
        spec = {"operation": "common_cast", "parameters": {"movies": ["Inception", "Batman (1989)"]}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["person_name"] == "Jack Nicholson"
        assert "Inception" in res["results"][0]["roles_by_movie"]
        assert "Batman" in res["results"][0]["roles_by_movie"]

    def test_aggregate_by_genre(self, query_engine):
        spec = {"operation": "aggregate_by_genre", "parameters": {"metric": "vote_average", "min_vote_count": 1}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["result_count"] >= 2
        assert "metadata" in res
        assert "total_included_movies" in res["metadata"]
        assert "total_excluded_movies" in res["metadata"]
        assert "aggregation_methodology" in res["metadata"]
        # Crime genre has Pulp Fiction (8.5)
        crime = next(g for g in res["results"] if g["genre_name"] == "Crime")
        assert crime["avg_vote_average"] == 8.5
        assert crime["movie_count"] == 1
        assert crime["min_vote_average"] > 0.0

    def test_aggregate_by_genre_includes_zero_votes_when_requested(self, query_engine):
        spec = {"operation": "aggregate_by_genre", "parameters": {"metric": "vote_average", "min_vote_count": 0}}
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        assert res["metadata"]["min_vote_count"] == 0

    def test_movies_by_year_range_chronological_asc(self, query_engine):
        spec = {
            "operation": "movies_by_year_range",
            "parameters": {"year_start": 1960, "year_end": 2025, "sort_order": "asc"},
        }
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        years = [m["release_year"] for m in res["results"]]
        # Ensure chronological: monotonically increasing
        assert years == sorted(years)
        assert res["results"][0]["title"] == "Batman"
        assert res["results"][0]["release_year"] == 1966

    def test_movies_by_year_range_chronological_desc(self, query_engine):
        spec = {
            "operation": "movies_by_year_range",
            "parameters": {"year_start": 1960, "year_end": 2025, "sort_order": "desc"},
        }
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        years = [m["release_year"] for m in res["results"]]
        assert years == sorted(years, reverse=True)
        assert res["results"][0]["release_year"] == 2021

    def test_movies_by_genre_sorting_order_regression(self, query_engine):
        """Action genre has Inception (8.3), Batman 1989 (7.0), Batman 1966 (6.1)."""
        spec = {
            "operation": "movies_by_genre",
            "parameters": {"genre_name": "Action", "metric": "vote_average", "sort_order": "desc"},
        }
        res = query_engine.execute_query(spec)
        assert res["success"] is True
        ratings = [m["vote_average"] for m in res["results"]]
        assert ratings == sorted(ratings, reverse=True)
        assert res["results"][0]["title"] == "Inception"
        assert res["results"][0]["vote_average"] == 8.3


class TestQueryValidationAndErrors:
    def test_invalid_operation(self, query_engine):
        spec = {"operation": "hack_database", "parameters": {}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "INVALID_OPERATION"

    def test_missing_parameter(self, query_engine):
        spec = {"operation": "movie_details", "parameters": {}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "MISSING_PARAMETER"

    def test_invalid_metric(self, query_engine):
        spec = {"operation": "top_movies", "parameters": {"metric": "random_score"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "INVALID_METRIC"

    def test_invalid_role(self, query_engine):
        spec = {"operation": "movies_by_person", "parameters": {"person_name": "Leonardo DiCaprio", "role": "cameraman_invalid"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "INVALID_ROLE"

    def test_entity_not_found(self, query_engine):
        spec = {"operation": "movie_details", "parameters": {"title": "NonExistentMovie12345"}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "MOVIE_NOT_FOUND"

    def test_compare_movies_requires_at_least_two(self, query_engine):
        spec = {"operation": "compare_movies", "parameters": {"movies": ["Inception"]}}
        res = query_engine.execute_query(spec)
        assert res["success"] is False
        assert res["error"]["code"] == "INVALID_PARAMETER"


@pytest.fixture(scope="module")
def real_engine():
    assert GRAPH_PICKLE_PATH.exists()
    return load_query_engine(GRAPH_PICKLE_PATH)


class TestIntegrationWithRealPersistedGraph:
    """Integration test suite executing against the actual saved 124,700-node graph."""

    def test_avatar_details(self, real_engine):
        res = real_engine.execute_query({"operation": "movie_details", "parameters": {"title": "Avatar"}})
        assert res["success"] is True
        movie = res["results"]
        assert movie["tmdb_id"] == 19995
        assert movie["release_year"] == 2009
        assert movie["budget"] == 237000000.0
        assert movie["revenue"] == 2787965087.0
        assert any(d["name"] == "James Cameron" for d in movie["directors"])

    def test_nolan_filmography(self, real_engine):
        res = real_engine.execute_query({
            "operation": "movies_by_person",
            "parameters": {"person_name": "Christopher Nolan", "role": "director"},
        })
        assert res["success"] is True
        titles = {m["movie_title"] for m in res["results"]}
        assert "Inception" in titles
        assert "Interstellar" in titles
        assert "The Dark Knight" in titles

    def test_top_revenue_movies(self, real_engine):
        res = real_engine.execute_query({"operation": "top_movies", "parameters": {"metric": "revenue", "limit": 5}})
        assert res["success"] is True
        assert res["result_count"] == 5
        assert res["results"][0]["title"] == "Avatar"
        assert res["results"][1]["title"] == "Titanic"

    def test_common_cast_avatar_and_titanic(self, real_engine):
        res = real_engine.execute_query({
            "operation": "common_cast",
            "parameters": {"movies": ["Avatar", "Titanic"]},
        })
        assert res["success"] is True
        assert res["result_count"] == 1
        assert res["results"][0]["person_name"] == "Julene Renee"

    def test_science_fiction_ranking_regression(self, real_engine):
        """Issue 1 Regression: The Empire Strikes Back (8.2) must precede Inception (8.1) and Interstellar (8.1)."""
        res = real_engine.execute_query({
            "operation": "movies_by_genre",
            "parameters": {"genre_name": "Science Fiction", "metric": "vote_average", "limit": 8},
        })
        assert res["success"] is True
        results = res["results"]
        # Verify first movie is The Empire Strikes Back with 8.2 rating
        assert results[0]["title"] == "The Empire Strikes Back"
        assert results[0]["vote_average"] == 8.2
        # Verify all ratings are strictly sorted in descending order
        ratings = [m["vote_average"] for m in results]
        assert ratings == sorted(ratings, reverse=True)
        # Verify Inception precedes Interstellar due to secondary vote_count tiebreaker (13752 vs 10867)
        titles = [m["title"] for m in results]
        assert titles.index("Inception") < titles.index("Interstellar")
