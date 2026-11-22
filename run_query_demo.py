"""Demonstration of 15 queries executed on TMDB 5000 Knowledge Graph Query Engine."""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import GRAPH_PICKLE_PATH
from src.query_engine import load_query_engine, execute_query


def format_currency(val):
    if val is None:
        return "N/A"
    return f"${val:,.0f}"


def run_demo():
    print("=" * 75)
    print("      TMDB 5000 KNOWLEDGE GRAPH QUERY ENGINE - 15 QUERY DEMO")
    print("=" * 75)

    if not GRAPH_PICKLE_PATH.exists():
        print(f"Error: Knowledge graph not found at {GRAPH_PICKLE_PATH}. Please run run_graph_builder.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Graph from: {GRAPH_PICKLE_PATH}...")
    t0 = time.time()
    engine = load_query_engine(GRAPH_PICKLE_PATH)
    load_time = time.time() - t0
    print(f"Graph & Query Engine loaded in {load_time:.3f} seconds.\n")

    queries = [
        # Query 1: Find Avatar details
        {
            "num": 1,
            "title": "Find Avatar details",
            "spec": {"operation": "movie_details", "parameters": {"title": "Avatar"}},
            "formatter": lambda res, meta: (
                f"Title: {res['title']} (TMDB ID: {res['tmdb_id']}, Year: {res['release_year']})\n"
                f"Rating: {res['vote_average']} ({res['vote_count']:,} votes) | Popularity: {res['popularity']:.2f}\n"
                f"Budget: {format_currency(res['budget'])} | Revenue: {format_currency(res['revenue'])}\n"
                f"Directors: {', '.join(d['name'] for d in res['directors'])}\n"
                f"Genres: {', '.join(g['name'] for g in res['genres'])}\n"
                f"Overview: {res['overview'][:140]}..."
            ),
        },
        # Query 2: Find Titanic details
        {
            "num": 2,
            "title": "Find Titanic details",
            "spec": {"operation": "movie_details", "parameters": {"title": "Titanic"}},
            "formatter": lambda res, meta: (
                f"Title: {res['title']} (TMDB ID: {res['tmdb_id']}, Year: {res['release_year']})\n"
                f"Rating: {res['vote_average']} ({res['vote_count']:,} votes) | Popularity: {res['popularity']:.2f}\n"
                f"Budget: {format_currency(res['budget'])} | Revenue: {format_currency(res['revenue'])}\n"
                f"Directors: {', '.join(d['name'] for d in res['directors'])}\n"
                f"Genres: {', '.join(g['name'] for g in res['genres'])}\n"
                f"Overview: {res['overview'][:140]}..."
            ),
        },
        # Query 3: Find Avatar's director
        {
            "num": 3,
            "title": "Find Avatar's director",
            "spec": {"operation": "movie_directors", "parameters": {"title": "Avatar"}},
            "formatter": lambda res, meta: "\n".join(
                f"- {d['person_name']} (Person ID: {d['person_id']}, Job: {d['job']}, Dept: {d['department']})"
                for d in res
            ),
        },
        # Query 4: Find Avatar's cast
        {
            "num": 4,
            "title": "Find Avatar's cast (Top 8 by billing order)",
            "spec": {"operation": "movie_cast", "parameters": {"title": "Avatar", "limit": 8}},
            "formatter": lambda res, meta: "\n".join(
                f"  [{c['order']}] {c['person_name']} as '{c['character']}' (ID: {c['person_id']})"
                for c in res
            ),
        },
        # Query 5: Find all movies featuring Leonardo DiCaprio
        {
            "num": 5,
            "title": "Find all movies featuring Leonardo DiCaprio",
            "spec": {"operation": "movies_by_person", "parameters": {"person_name": "Leonardo DiCaprio", "role": "actor"}},
            "formatter": lambda res, meta: "\n".join(
                f"- {m['movie_title']} ({m['release_year']}) as '{m['character']}' [Rating: {m['vote_average']}]"
                for m in res[:10]
            ) + (f"\n... and {len(res)-10} more movies (Total: {len(res)})" if len(res) > 10 else ""),
        },
        # Query 6: Find all movies directed by Christopher Nolan
        {
            "num": 6,
            "title": "Find all movies directed by Christopher Nolan",
            "spec": {"operation": "movies_by_person", "parameters": {"person_name": "Christopher Nolan", "role": "director"}},
            "formatter": lambda res, meta: "\n".join(
                f"- {m['movie_title']} ({m['release_year']}) [Rating: {m['vote_average']}, Popularity: {m['popularity']:.1f}]"
                for m in res
            ),
        },
        # Query 7: Find all Science Fiction movies sorted by rating
        {
            "num": 7,
            "title": "Find all Science Fiction movies (Top 8 by Rating descending)",
            "spec": {"operation": "movies_by_genre", "parameters": {"genre_name": "Science Fiction", "metric": "vote_average", "limit": 8}},
            "formatter": lambda res, meta: "\n".join(
                f"  #{idx+1} {m['title']} ({m['release_year']}) - Rating: {m['vote_average']:.2f} ({m['vote_count']:,} votes)"
                for idx, m in enumerate(res)
            ),
        },
        # Query 8: Find all movies associated with the keyword Alien
        {
            "num": 8,
            "title": "Find all movies associated with the keyword 'alien'",
            "spec": {"operation": "movies_by_keyword", "parameters": {"keyword_name": "alien", "limit": 8}},
            "formatter": lambda res, meta: "\n".join(
                f"- {m['title']} ({m['release_year']}) [Popularity: {m['popularity']:.2f}, Rating: {m['vote_average']}]"
                for m in res
            ),
        },
        # Query 9: Top 10 highest-rated movies with at least 500 votes
        {
            "num": 9,
            "title": "Top 10 highest-rated movies (min 500 votes)",
            "spec": {"operation": "top_movies", "parameters": {"metric": "vote_average", "min_vote_count": 500, "limit": 10}},
            "formatter": lambda res, meta: "\n".join(
                f"  #{idx+1} {m['title']} ({m['release_year']}): {m['vote_average']:.2f} ({m['vote_count']:,} votes)"
                for idx, m in enumerate(res)
            ),
        },
        # Query 10: 10 most popular movies
        {
            "num": 10,
            "title": "10 most popular movies",
            "spec": {"operation": "top_movies", "parameters": {"metric": "popularity", "limit": 10}},
            "formatter": lambda res, meta: "\n".join(
                f"  #{idx+1} {m['title']} ({m['release_year']}): Popularity {m['popularity']:.2f} [Rating: {m['vote_average']}]"
                for idx, m in enumerate(res)
            ),
        },
        # Query 11: 10 highest-revenue movies
        {
            "num": 11,
            "title": "10 highest-revenue movies",
            "spec": {"operation": "top_movies", "parameters": {"metric": "revenue", "limit": 10}},
            "formatter": lambda res, meta: "\n".join(
                f"  #{idx+1} {m['title']} ({m['release_year']}): Revenue {format_currency(m['revenue'])} (Budget: {format_currency(m['budget'])})"
                for idx, m in enumerate(res)
            ),
        },
        # Query 12: Compare Avatar and Titanic
        {
            "num": 12,
            "title": "Compare Avatar and Titanic",
            "spec": {"operation": "compare_movies", "parameters": {"movies": ["Avatar", "Titanic"]}},
            "formatter": lambda res, meta: "\n".join(
                f"  [{m['title']} ({m['release_year']})]\n"
                f"    Rating: {m['vote_average']} ({m['vote_count']:,} votes) | Popularity: {m['popularity']:.2f}\n"
                f"    Budget: {format_currency(m['budget'])} | Revenue: {format_currency(m['revenue'])}\n"
                f"    Directors: {m['directors']}\n"
                f"    Genres: {m['genres']}"
                for m in res
            ),
        },
        # Query 13: Common actors between Avatar and Titanic
        {
            "num": 13,
            "title": "Find actors common to Avatar and Titanic",
            "spec": {"operation": "common_cast", "parameters": {"movies": ["Avatar", "Titanic"]}},
            "formatter": lambda res, meta: (
                "\n".join(
                    f"- {a['person_name']} (Person ID: {a['person_id']}): "
                    + ", ".join(f"in '{m}' as '{role}'" for m, role in a['roles_by_movie'].items())
                    for a in res
                ) if res else "No common actors found."
            ),
        },
        # Query 14: Average rating by genre (excluding unrated movies)
        {
            "num": 14,
            "title": "Calculate average rating by genre (Top 8 Genres, min_vote_count=1)",
            "spec": {"operation": "aggregate_by_genre", "parameters": {"metric": "vote_average", "min_vote_count": 1}},
            "formatter": lambda res, meta: (
                f"Methodology: {meta.get('aggregation_methodology')}\n"
                f"Included Movies: {meta.get('total_included_movies'):,} | Excluded Movies (0 votes): {meta.get('total_excluded_movies'):,}\n"
                + "\n".join(
                    f"  {g['genre_name']:<18} | Avg: {g['avg_vote_average']:.2f} | Range: [{g['min_vote_average']} - {g['max_vote_average']}] | Movies: {g['movie_count']:>4} (Excluded: {g['excluded_movie_count']})"
                    for g in res[:8]
                )
            ),
        },
        # Query 15: Movies released between 2010 and 2020 (Chronological order)
        {
            "num": 15,
            "title": "Find movies released between 2010 and 2020 (Sample 8, Chronological Order)",
            "spec": {"operation": "movies_by_year_range", "parameters": {"year_start": 2010, "year_end": 2020, "sort_order": "asc", "limit": 8}},
            "formatter": lambda res, meta: "\n".join(
                f"- {m['title']} (Release Date: {m['release_date']}) - Rating: {m['vote_average']} ({m['vote_count']:,} votes)"
                for m in res
            ),
        },
    ]

    total_query_time = 0.0
    query_benchmarks = []

    for item in queries:
        q_num = item["num"]
        q_title = item["title"]
        q_spec = item["spec"]
        formatter = item["formatter"]

        print("-" * 75)
        print(f"QUERY {q_num}: {q_title}")
        print(f"Specification: {json.dumps(q_spec)}")

        t_start = time.perf_counter()
        response = execute_query(q_spec)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        total_query_time += elapsed_ms

        query_benchmarks.append((q_num, q_title, elapsed_ms))

        if not response.get("success"):
            print(f"ERROR: {response.get('error')}")
        else:
            res_data = response.get("results")
            res_meta = response.get("metadata", {})
            formatted_text = formatter(res_data, res_meta)
            print("Result:")
            print(formatted_text)
            print(f"Execution Time: {elapsed_ms:.3f} ms (Result Count: {response.get('result_count', 0)})")
        print()

    print("=" * 75)
    print("                 PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Graph Loading Time:           {load_time:.3f} s")
    print(f"Total Time for 15 Queries:    {total_query_time:.2f} ms ({total_query_time/1000:.4f} s)")
    print(f"Average Time per Query:       {total_query_time/len(queries):.2f} ms")
    print("\nPer-Query Breakdown:")
    for num, name, ms in query_benchmarks:
        print(f"  Query {num:>2}: {ms:>7.2f} ms | {name}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_demo()
