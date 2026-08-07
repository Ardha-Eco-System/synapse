"""Tests for RetrievalEngine."""

from synapse.retrieval import RetrievalEngine, tokenize_query


def test_bm25_query_builder():
    """BM25 query should not use LLM — pure Cypher fulltext search."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="test")
    query, _ = engine._build_bm25_query("temporal knowledge graph", limit=5)
    assert "RELATES_TO" in query
    assert "fact" in query
    assert "LIMIT 5" in query


def test_cache_key_generation():
    engine = RetrievalEngine(host="localhost", port=6379, group_id="test")
    key = engine._cache_key("what is Synapse?")
    assert isinstance(key, str)
    assert len(key) > 0


def test_format_results():
    """Results should be formatted as context for the system prompt."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="test")
    edges = [
        {"fact": "Synapse uses FalkorDB", "from_node": "Synapse", "to_node": "FalkorDB"},
        {"fact": "Synapse uses Graphiti", "from_node": "Synapse", "to_node": "Graphiti"},
    ]
    formatted = engine._format_results(edges)
    assert "Synapse uses FalkorDB" in formatted
    assert "Synapse uses Graphiti" in formatted


def test_format_empty_results():
    """Empty results should return empty string."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="test")
    assert engine._format_results([]) == ""


def test_prefetch_returns_cached():
    """prefetch() should return cached result from queue_prefetch()."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="test")
    # No cache → empty string
    assert engine.prefetch("anything") == ""

    # Manually inject cache
    engine._cache[engine._cache_key("test query")] = "cached context"
    assert engine.prefetch("test query") == "cached context"


# --- Cypher injection regression tests -------------------------------------

def test_bm25_query_binds_term_as_parameter():
    """A hostile search term must never reach the query text."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="g")
    hostile = "x' OR true DETACH DELETE r //"
    cypher, params = engine._build_bm25_query(hostile)
    assert "DETACH DELETE" not in cypher
    assert "$tokens" in cypher
    # Tokenized, lowercased, and bound as a list parameter
    assert "detach" in params["tokens"]
    assert "'" not in "".join(params["tokens"])


def test_bm25_query_handles_trailing_backslash():
    """The old escaping let a trailing backslash escape its own quote."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="g")
    cypher, params = engine._build_bm25_query("windows path C:\\")
    assert params["tokens"] == ["windows", "path"]
    assert "'" not in cypher  # no interpolated string literals at all


def test_bm25_query_binds_group_id_as_parameter():
    """group_id is caller-supplied and must also be parameterized."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="a' OR true //")
    cypher, params = engine._build_bm25_query("hello")
    assert "OR true" not in cypher
    assert "$group_id" in cypher
    assert params["group_id"] == "a' OR true //"


# --- Tokenized, case-insensitive matching ----------------------------------

def test_tokenize_lowercases_and_drops_stopwords():
    """An agent's natural-language question reduces to its content words."""
    assert tokenize_query("what memory provider does Darin use") == [
        "memory", "provider", "darin", "use",
    ]


def test_tokenize_splits_on_punctuation():
    """Slashes and punctuation are token separators, not part of the token."""
    assert tokenize_query("synapse/graphiti") == ["synapse", "graphiti"]


def test_tokenize_keeps_alphanumeric_terms():
    """Product names with digits must survive tokenization intact."""
    assert tokenize_query("mem0") == ["mem0"]


def test_tokenize_falls_back_when_all_stopwords():
    """A query of pure stopwords still yields something to match on."""
    assert tokenize_query("what is the") == ["what", "is", "the"]


def test_tokenize_handles_empty():
    assert tokenize_query("") == []


def test_query_is_case_insensitive():
    """'Mem0' and 'mem0' must produce the same match tokens."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="g")
    _, upper = engine._build_bm25_query("Mem0")
    _, lower = engine._build_bm25_query("mem0")
    assert upper["tokens"] == lower["tokens"] == ["mem0"]


def test_query_ranks_by_token_overlap():
    """Facts matching more tokens must rank first."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="g")
    cypher, _ = engine._build_bm25_query("memory provider")
    assert "ORDER BY score DESC" in cypher
    assert "toLower(r.fact)" in cypher


def test_query_returns_match_score():
    """The token-overlap score is returned so callers can rank on it."""
    engine = RetrievalEngine(host="localhost", port=6379, group_id="g")
    cypher, _ = engine._build_bm25_query("memory provider")
    assert "invalid_at, score " in cypher
    assert "ORDER BY score DESC" in cypher
