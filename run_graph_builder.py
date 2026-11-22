"""Script to execute TMDB 5000 Knowledge Graph Construction pipeline."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROCESSED_MOVIES_PARQUET, GRAPH_PICKLE_PATH
from src.preprocessing import load_processed_dataset
from src.graph_builder import build_graph, save_graph, get_graph_statistics


def print_validation_report(stats: dict, file_path: Path):
    print("\n" + "=" * 65)
    print("      TMDB 5000 KNOWLEDGE GRAPH VALIDATION REPORT")
    print("=" * 65)
    print(f"Graph Construction Time:    {stats.get('construction_time_seconds', 0.0)} seconds")
    print(f"Total Graph Nodes:          {stats.get('total_nodes', 0):,}")
    print(f"Total Graph Edges:          {stats.get('total_edges', 0):,}")
    print(f"Isolated Nodes:             {stats.get('isolated_nodes', 0):,}")

    print("\n--- NODES GROUPED BY TYPE ---")
    for ntype, count in sorted(stats.get("nodes_by_type", {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {ntype:<22} : {count:>8,}")

    print("\n--- EDGES GROUPED BY RELATIONSHIP ---")
    for rel, count in sorted(stats.get("edges_by_relationship", {}).items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel:<22} : {count:>8,}")

    print("\n--- MOVIE CONNECTIVITY METRICS ---")
    total_m = stats.get("total_movies", 0)
    m_cast = stats.get("movies_with_cast", 0)
    m_dir = stats.get("movies_with_directors", 0)
    m_gen = stats.get("movies_with_genres", 0)

    print(f"  Total Movies in Graph:     {total_m:>8,}")
    print(f"  Movies with Cast Members:  {m_cast:>8,} ({m_cast/total_m*100:.1f}%)" if total_m else "  N/A")
    print(f"  Movies with Directors:     {m_dir:>8,} ({m_dir/total_m*100:.1f}%)" if total_m else "  N/A")
    print(f"  Movies with Genres:        {m_gen:>8,} ({m_gen/total_m*100:.1f}%)" if total_m else "  N/A")

    if "skipped_entities" in stats:
        print("\n--- DATA HYGIENE (SKIPPED UNIDENTIFIABLE ENTITIES) ---")
        for k, v in stats["skipped_entities"].items():
            print(f"  {k:<28} : {v:>6,}")

    size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0.0
    print("\n--- PERSISTENCE INFORMATION ---")
    print(f"  Storage Location:          {file_path}")
    print(f"  Serialization Format:      Pickle (NetworkX MultiDiGraph)")
    print(f"  File Size:                 {size_mb:.2f} MB")
    print("=" * 65 + "\n")


def main():
    print("=" * 65)
    print("Starting TMDB 5000 Knowledge Graph Construction")
    print("=" * 65)

    if not PROCESSED_MOVIES_PARQUET.exists():
        print(
            f"Error: Processed dataset not found at {PROCESSED_MOVIES_PARQUET}. "
            f"Please run run_preprocessing.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading processed dataset: {PROCESSED_MOVIES_PARQUET}")
    df = load_processed_dataset(PROCESSED_MOVIES_PARQUET)
    print(f"Loaded {len(df):,} movie records.")

    # Build graph
    G, stats = build_graph(df)

    # Save graph
    save_graph(G, GRAPH_PICKLE_PATH)

    # Print validation report
    print_validation_report(stats, GRAPH_PICKLE_PATH)


if __name__ == "__main__":
    main()
