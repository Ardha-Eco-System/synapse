"""BM25-only retrieval with background caching.

Optimization: skip the cross-encoder reranker for prefetch.
BM25 fulltext search via Cypher is ~70x faster (0.01s vs 0.7s)
and produces good-enough results (100% search accuracy in Spike 003).

Background caching: queue_prefetch() runs search in a background
thread after each turn. The result is cached and returned by
prefetch() on the next turn — zero blocking latency.
"""

from __future__ import annotations

import hashlib
import re
import threading

# Words carried by almost every natural-language question. An agent asks
# "what memory provider does Darin use?", not "memory provider". Left in,
# they contribute noise to the match score; dropped, the meaningful terms
# rank the results.
_STOPWORDS = frozenset("""
a an and any are as at be been but by can did do does for from had has have
his her he she how i if in into is it its me my of on or our so than that the
their them then there these they this to us was we were what when where which
who why will with would you your
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _safe_limit(limit: object, default: int = 5, maximum: int = 1000) -> int:
    """Coerce a LIMIT value to a bounded int.

    FalkorDB does not accept a parameter in the LIMIT position, so this value
    is interpolated into the query text. Coercing through int() and clamping
    keeps that interpolation well-defined regardless of caller input.
    """
    try:
        value = int(limit)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default
    return max(1, min(value, maximum))


def tokenize_query(query: str) -> list[str]:
    """Split a search query into lowercased content tokens.

    Returns tokens with stopwords and single characters removed. If that
    leaves nothing (e.g. the query was entirely stopwords), falls back to
    every token, and finally to the whole lowercased string, so a query
    always produces something to match on.
    """
    raw = [t.lower() for t in _TOKEN_RE.findall(query or "")]
    if not raw:
        stripped = (query or "").strip().lower()
        return [stripped] if stripped else []
    meaningful = [t for t in raw if len(t) > 1 and t not in _STOPWORDS]
    return meaningful or raw


class RetrievalEngine:
    """BM25-only retrieval with background caching."""

    def __init__(
        self,
        host: str,
        port: int,
        group_id: str,
        password: str | None = None,
    ):
        self.host = host
        self.port = port
        self.group_id = group_id
        self.password = password
        self._cache: dict[str, str] = {}
        self._falkor_helper = None

    def _get_helper(self):
        """Lazy-initialize the FalkorHelper."""
        if self._falkor_helper is None:
            from synapse.falkor import FalkorHelper
            self._falkor_helper = FalkorHelper(
                host=self.host, port=self.port, password=self.password
            )
        return self._falkor_helper

    def _cache_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.md5(f"{self.group_id}:{query[:200]}".encode()).hexdigest()

    def _build_bm25_query(self, query: str, limit: int = 5) -> tuple[str, dict]:
        """Build a token-scored Cypher query for edge facts.

        Previously this was a single case-sensitive `r.fact CONTAINS $term`
        over the raw query string, which meant a fact was only findable by a
        literal, correctly-cased substring of itself. "Mem0" missed a fact
        containing "mem0", and a natural-language question like "what memory
        provider does Darin use" matched nothing at all — an agent phrasing
        queries the way an agent does would essentially never get a hit.

        Now the query is tokenized and each fact is scored by how many
        distinct tokens it contains, case-insensitively, with the best
        matches ranked first. A fact needs at least one token to match, so
        partial matches still surface rather than returning nothing.

        The search terms and group_id are bound as Cypher parameters. The
        original hand-rolled escaping (quotes only, not backslashes) let a
        query ending in a backslash escape its own closing quote.

        Returns:
            Tuple of (Cypher query string, params dict).
        """
        return (
            f"MATCH (a)-[r:RELATES_TO]->(b) "
            f"WHERE a.group_id = $group_id "
            f"WITH a, r, b, "
            f"size([t IN $tokens WHERE toLower(r.fact) CONTAINS t]) AS score "
            f"WHERE score > 0 "
            f"RETURN a.name AS from_node, r.fact AS fact, b.name AS to_node, "
            f"r.valid_at AS valid_at, r.invalid_at AS invalid_at "
            f"ORDER BY score DESC "
            f"LIMIT {_safe_limit(limit, default=5)}",
            {"group_id": self.group_id, "tokens": tokenize_query(query)},
        )

    def _format_results(self, edges: list[dict]) -> str:
        """Format search results as context for the system prompt."""
        if not edges:
            return ""
        lines = ["## Relevant memories from past conversations:"]
        for edge in edges[:5]:
            fact = edge.get("fact", str(edge))
            lines.append(f"- {fact}")
        return "\n".join(lines)

    def prefetch(self, query: str) -> str:
        """Return context for the current query.

        Returns the cached result when queue_prefetch() has already run for
        this exact query. Otherwise runs the BM25 search inline.

        The inline fallback exists because the host agent queues a prefetch
        using turn N's message but requests the prefetch for turn N+1's
        message — the keys never match, so a cache-only implementation
        returns "" on essentially every turn. BM25 here is a single indexed
        CONTAINS scan (~10ms), and the caller already runs this on a
        separate thread under a timeout.
        """
        key = self._cache_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            edges = self.search(query, limit=5)
        except Exception:
            return ""
        formatted = self._format_results(edges)
        if formatted:
            self._cache[key] = formatted
        return formatted

    def queue_prefetch(self, query: str) -> None:
        """Run search in background and cache result for next turn."""

        def _bg_search():
            try:
                helper = self._get_helper()
                graph = helper.get_graph(self.group_id)
                cypher, params = self._build_bm25_query(query, limit=5)
                result = graph.query(cypher, params)
                if result and result.result_set:
                    header = [h[1] for h in result.header]
                    edges = [
                        {header[i]: row[i] for i in range(min(len(header), len(row)))}
                        for row in result.result_set
                    ]
                    formatted = self._format_results(edges)
                    if formatted:
                        self._cache[self._cache_key(query)] = formatted
            except Exception:
                pass  # Background — fail silently

        threading.Thread(target=_bg_search, daemon=True).start()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Synchronous BM25 search (for tool calls)."""
        helper = self._get_helper()
        graph = helper.get_graph(self.group_id)
        cypher, params = self._build_bm25_query(query, limit=limit)
        result = graph.query(cypher, params)
        if not result or not result.result_set:
            return []
        header = [h[1] for h in result.header]
        return [
            {header[i]: row[i] for i in range(min(len(header), len(row)))}
            for row in result.result_set
        ]

    def temporal_search(self, query: str, at_time: str, limit: int = 20) -> list[dict]:
        """Point-in-time search using the substring() workaround."""
        helper = self._get_helper()
        graph = helper.get_graph(self.group_id)
        cypher = helper.temporal_filter_query(valid_at_max=at_time, limit=limit)
        result = graph.query(cypher)
        if not result or not result.result_set:
            return []
        header = [h[1] for h in result.header]
        return [
            {header[i]: row[i] for i in range(min(len(header), len(row)))}
            for row in result.result_set
        ]

    def clear_cache(self) -> None:
        """Clear the prefetch cache."""
        self._cache.clear()
