"""Unit and integration tests for dataset preprocessing."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    parse_json_field,
    validate_keys,
    extract_directors,
    extract_crew_by_roles,
    preprocess_datasets,
)
from src.config import MOVIES_RAW_PATH, CREDITS_RAW_PATH


class TestJsonParsing:
    def test_parse_valid_json_list(self):
        sample = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        res = parse_json_field(sample)
        assert len(res) == 2
        assert res[0]["name"] == "Action"
        assert res[1]["id"] == 12

    def test_parse_empty_or_none(self):
        assert parse_json_field(None) == []
        assert parse_json_field("") == []
        assert parse_json_field("   ") == []
        assert parse_json_field("[]") == []
        assert parse_json_field(np.nan) == []

    def test_parse_malformed_json(self):
        assert parse_json_field("{malformed json") == []

    def test_parse_already_parsed_list(self):
        obj = [{"id": 1, "name": "Test"}]
        assert parse_json_field(obj) == obj


class TestKeyValidation:
    def test_valid_keys(self):
        df_m = pd.DataFrame({"id": [1, 2, 3], "title": ["A", "B", "C"]})
        df_c = pd.DataFrame({"movie_id": [1, 2, 3], "title": ["A", "B", "C"]})
        result = validate_keys(df_m, df_c)
        assert result["keys_match"] is True
        assert result["movies_count"] == 3

    def test_missing_column(self):
        df_m = pd.DataFrame({"wrong_id": [1, 2], "title": ["A", "B"]})
        df_c = pd.DataFrame({"movie_id": [1, 2], "title": ["A", "B"]})
        with pytest.raises(KeyError):
            validate_keys(df_m, df_c)

    def test_duplicate_primary_key(self):
        df_m = pd.DataFrame({"id": [1, 1, 2], "title": ["A", "A", "B"]})
        df_c = pd.DataFrame({"movie_id": [1, 2, 3], "title": ["A", "B", "C"]})
        with pytest.raises(ValueError, match="Duplicate primary key"):
            validate_keys(df_m, df_c)

    def test_mismatched_keys(self):
        df_m = pd.DataFrame({"id": [1, 2, 3], "title": ["A", "B", "C"]})
        df_c = pd.DataFrame({"movie_id": [1, 2, 4], "title": ["A", "B", "D"]})
        with pytest.raises(ValueError, match="Key mismatch"):
            validate_keys(df_m, df_c)


class TestCrewExtraction:
    def test_extract_directors(self):
        crew = [
            {"id": 100, "name": "Christopher Nolan", "job": "Director"},
            {"id": 101, "name": "Emma Thomas", "job": "Producer"},
            {"id": 100, "name": "Christopher Nolan", "job": "Director"},  # Duplicate entry
        ]
        directors = extract_directors(crew)
        assert len(directors) == 1
        assert directors[0]["id"] == 100
        assert directors[0]["name"] == "Christopher Nolan"

    def test_extract_crew_by_roles(self):
        crew = [
            {"id": 201, "name": "John Doe", "job": "Producer", "department": "Production"},
            {"id": 202, "name": "Jane Smith", "job": "Screenplay", "department": "Writing"},
            {"id": 203, "name": "Bob Taylor", "job": "Gaffer", "department": "Lighting"},
        ]
        producers = extract_crew_by_roles(crew, ["Producer"])
        assert len(producers) == 1
        assert producers[0]["name"] == "John Doe"

        writers = extract_crew_by_roles(crew, ["Screenplay", "Writer"])
        assert len(writers) == 1
        assert writers[0]["name"] == "Jane Smith"


class TestPreprocessingPipeline:
    @pytest.fixture
    def mini_datasets(self, tmp_path: Path):
        movies_csv = tmp_path / "mini_movies.csv"
        credits_csv = tmp_path / "mini_credits.csv"

        movies_data = [
            {
                "id": 101,
                "title": "Avatar",
                "budget": 237000000,
                "revenue": 2787965087,
                "release_date": "2009-12-10",
                "vote_average": 7.2,
                "vote_count": 11800,
                "popularity": 150.43,
                "runtime": 162.0,
                "overview": "A paraplegic Marine...",
                "tagline": "Enter the World",
                "homepage": "http://avatar.com",
                "original_language": "en",
                "original_title": "Avatar",
                "status": "Released",
                "genres": json.dumps([{"id": 28, "name": "Action"}]),
                "keywords": json.dumps([{"id": 1463, "name": "culture clash"}]),
                "production_companies": json.dumps([{"name": "Lightstorm", "id": 574}]),
                "production_countries": json.dumps([{"iso_3166_1": "US", "name": "United States"}]),
                "spoken_languages": json.dumps([{"iso_639_1": "en", "name": "English"}]),
            },
            {
                "id": 102,
                "title": "Batman",
                "budget": 0,  # Unrecorded budget
                "revenue": 0,  # Unrecorded revenue
                "release_date": "1989-06-23",
                "vote_average": 7.0,
                "vote_count": 2000,
                "popularity": 45.0,
                "runtime": 126.0,
                "overview": "The Dark Knight...",
                "tagline": "Have you ever danced with the devil...",
                "homepage": "",
                "original_language": "en",
                "original_title": "Batman",
                "status": "Released",
                "genres": json.dumps([{"id": 28, "name": "Action"}]),
                "keywords": json.dumps([{"id": 849, "name": "dc comics"}]),
                "production_companies": json.dumps([{"name": "Warner Bros", "id": 6194}]),
                "production_countries": json.dumps([{"iso_3166_1": "US", "name": "United States"}]),
                "spoken_languages": json.dumps([{"iso_639_1": "en", "name": "English"}]),
            },
            {
                "id": 103,
                "title": "Batman",  # Duplicate title
                "budget": 35000000,
                "revenue": 411348924,
                "release_date": "1966-07-30",
                "vote_average": 6.1,
                "vote_count": 400,
                "popularity": 12.0,
                "runtime": 105.0,
                "overview": "The original Batman...",
                "tagline": "Holy Batcave!",
                "homepage": "",
                "original_language": "en",
                "original_title": "Batman",
                "status": "Released",
                "genres": json.dumps([{"id": 28, "name": "Action"}]),
                "keywords": json.dumps([{"id": 849, "name": "dc comics"}]),
                "production_companies": json.dumps([{"name": "20th Century", "id": 306}]),
                "production_countries": json.dumps([{"iso_3166_1": "US", "name": "United States"}]),
                "spoken_languages": json.dumps([{"iso_639_1": "en", "name": "English"}]),
            },
        ]

        credits_data = [
            {
                "movie_id": 101,
                "title": "Avatar",
                "cast": json.dumps([{"id": 65731, "name": "Sam Worthington", "character": "Jake Sully"}]),
                "crew": json.dumps([{"id": 2710, "name": "James Cameron", "job": "Director"}]),
            },
            {
                "movie_id": 102,
                "title": "Batman",
                "cast": json.dumps([{"id": 2232, "name": "Michael Keaton", "character": "Bruce Wayne"}]),
                "crew": json.dumps([{"id": 510, "name": "Tim Burton", "job": "Director"}]),
            },
            {
                "movie_id": 103,
                "title": "Batman",
                "cast": json.dumps([{"id": 1098, "name": "Adam West", "character": "Bruce Wayne"}]),
                "crew": json.dumps([{"id": 999, "name": "Leslie H. Martinson", "job": "Director"}]),
            },
        ]

        pd.DataFrame(movies_data).to_csv(movies_csv, index=False)
        pd.DataFrame(credits_data).to_csv(credits_csv, index=False)
        return movies_csv, credits_csv

    def test_preprocessing_transformation(self, mini_datasets):
        movies_csv, credits_csv = mini_datasets
        df = preprocess_datasets(movies_csv, credits_csv)

        assert len(df) == 3
        assert "tmdb_id" in df.columns

        # Verify budget=0 was converted to NaN
        row_batman_1989 = df[df["tmdb_id"] == 102].iloc[0]
        assert np.isnan(row_batman_1989["budget"])
        assert np.isnan(row_batman_1989["revenue"])
        assert bool(row_batman_1989["has_valid_budget"]) is False

        # Verify Avatar has valid budget
        row_avatar = df[df["tmdb_id"] == 101].iloc[0]
        assert row_avatar["budget"] == 237000000.0
        assert row_avatar["revenue"] == 2787965087.0
        assert bool(row_avatar["has_valid_budget"]) is True

        # Verify release year extraction
        assert row_avatar["release_year"] == 2009
        assert row_batman_1989["release_year"] == 1989

        # Verify duplicate title disambiguation
        row_batman_1966 = df[df["tmdb_id"] == 103].iloc[0]
        assert row_batman_1989["disambiguated_title"] == "Batman (1989)"
        assert row_batman_1966["disambiguated_title"] == "Batman (1966)"
        assert row_avatar["disambiguated_title"] == "Avatar"

        # Verify director extraction
        assert row_avatar["director_names"] == ["James Cameron"]
        assert row_batman_1989["director_names"] == ["Tim Burton"]


class TestRealDatasetIntegrity:
    """Validate full TMDB 5000 dataset keys and counts."""

    def test_raw_files_exist(self):
        assert MOVIES_RAW_PATH.exists(), f"Missing {MOVIES_RAW_PATH}"
        assert CREDITS_RAW_PATH.exists(), f"Missing {CREDITS_RAW_PATH}"

    def test_raw_key_validation(self):
        movies_df = pd.read_csv(MOVIES_RAW_PATH, usecols=["id"])
        credits_df = pd.read_csv(CREDITS_RAW_PATH, usecols=["movie_id"])
        result = validate_keys(movies_df, credits_df)
        assert result["movies_count"] == 4803
        assert result["credits_count"] == 4803
        assert result["keys_match"] is True
