"""
Golden Test Set
================
Hand-curated test cases for agent evaluation.
Fixed across all eval runs — results are directly comparable.

Coverage:
    - Simple recommendations (genre, mood, era)
    - Follow-up refinements
    - Factual queries (plot, director)
    - Ambiguous queries (adversarial)
    - Filter queries (year, genre combination)
    - Cold start users
"""

GOLDEN_TEST_CASES = [
    # ── Simple recommendations ─────────────────────────────────────────────
    {
        "id": "rec_001",
        "query": "Recommend me psychological thrillers",
        "user_id": "user_1",
        "expected_genres": ["Thriller"],
        "expected_year_range": None,
        "expected_keywords": ["psychological", "thriller", "suspense"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "simple_recommend",
    },
    {
        "id": "rec_002",
        "query": "I want to watch something funny tonight",
        "user_id": "user_2",
        "expected_genres": ["Comedy"],
        "expected_year_range": None,
        "expected_keywords": ["comedy", "funny", "humor"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "simple_recommend",
    },
    {
        "id": "rec_003",
        "query": "Show me sci-fi movies from the 90s",
        "user_id": "user_3",
        "expected_genres": ["Sci-Fi"],
        "expected_year_range": [1990, 1999],
        "expected_keywords": ["sci-fi", "science fiction", "90s"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "filtered_recommend",
    },
    {
        "id": "rec_004",
        "query": "What are some good animated movies for adults?",
        "user_id": "user_4",
        "expected_genres": ["Animation"],
        "expected_year_range": None,
        "expected_keywords": ["animated", "animation"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "simple_recommend",
    },
    {
        "id": "rec_005",
        "query": "Give me critically acclaimed dramas from the 2000s",
        "user_id": "user_5",
        "expected_genres": ["Drama"],
        "expected_year_range": [2000, 2009],
        "expected_keywords": ["drama"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "filtered_recommend",
    },

    # ── Factual / RAG queries ──────────────────────────────────────────────
    {
        "id": "rag_001",
        "query": "What is the plot of Inception?",
        "user_id": "user_6",
        "expected_genres": [],
        "expected_year_range": None,
        "expected_keywords": ["dream", "inception", "nolan"],
        "optimal_steps": 3,
        "intent": "explain",
        "category": "factual",
    },
    {
        "id": "rag_002",
        "query": "Tell me about the themes in The Shawshank Redemption",
        "user_id": "user_7",
        "expected_genres": [],
        "expected_year_range": None,
        "expected_keywords": ["hope", "prison", "redemption", "freedom"],
        "optimal_steps": 3,
        "intent": "explain",
        "category": "factual",
    },

    # ── Follow-up refinements ──────────────────────────────────────────────
    {
        "id": "refine_001",
        "query": "More like that but older — from the 80s",
        "user_id": "user_1",
        "expected_genres": ["Thriller"],
        "expected_year_range": [1980, 1989],
        "expected_keywords": ["80s", "1980"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "refinement",
    },
    {
        "id": "refine_002",
        "query": "Why did you recommend that last movie?",
        "user_id": "user_1",
        "expected_genres": [],
        "expected_year_range": None,
        "expected_keywords": ["because", "recommend", "match"],
        "optimal_steps": 3,
        "intent": "explain",
        "category": "refinement",
    },

    # ── Ambiguous / adversarial ────────────────────────────────────────────
    {
        "id": "adv_001",
        "query": "something good",
        "user_id": "user_8",
        "expected_genres": [],
        "expected_year_range": None,
        "expected_keywords": ["recommend", "movie"],
        "optimal_steps": 5,
        "intent": "recommend",
        "category": "ambiguous",
    },
    {
        "id": "adv_002",
        "query": "I don't know what I want, just surprise me",
        "user_id": "user_9",
        "expected_genres": [],
        "expected_year_range": None,
        "expected_keywords": ["recommend", "suggest"],
        "optimal_steps": 5,
        "intent": "recommend",
        "category": "ambiguous",
    },

    # ── Cold start users ───────────────────────────────────────────────────
    {
        "id": "cold_001",
        "query": "Recommend me action movies",
        "user_id": "brand_new_user_xyz",   # not in training data
        "expected_genres": ["Action"],
        "expected_year_range": None,
        "expected_keywords": ["action"],
        "optimal_steps": 4,
        "intent": "recommend",
        "category": "cold_start",
    },
]