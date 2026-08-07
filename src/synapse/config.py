"""Synapse configuration — all optimization parameters in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SynapseConfig:
    """Configuration for Synapse memory provider.

    All optimization parameters have sensible defaults validated during spikes.
    Override via environment variables (SYNAPSE_*) or directly.
    """

    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str | None = None
    falkordb_database: str = "synapse"

    # LLM config (OpenAI-compatible endpoint)
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # "generic" uses OpenAIGenericClient (chat.completions + json_object) —
    # works with any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter).
    # "openai" uses the native OpenAIClient (responses.parse + json_schema),
    # which is required for the gpt-5 reasoning family on api.openai.com:
    # those models reject `max_tokens` and `temperature`.
    llm_client: str = "generic"

    # Reasoning effort for the gpt-5 family (llm_client="openai" only).
    # "auto" lets graphiti pick, but it resolves to "minimal" for any model
    # it doesn't recognize by prefix — and newer snapshots reject that tier.
    # Valid on current models: none, low, medium, high, xhigh, max.
    llm_reasoning: str = "auto"

    # The cross-encoder reranker calls chat.completions with max_tokens=1 and
    # temperature=0, which gpt-5 reasoning models reject. Keep it on a cheap
    # non-reasoning model independent of llm_model.
    reranker_model: str = "gpt-4.1-nano"

    # Optimization parameters (validated in spikes)
    batch_size: int = 5                    # turns per episode (−86% LLM calls)
    half_life_days: float = 7.0            # forgetting curve base
    salience_boost: float = 3.0            # high-salience decay multiplier
    recall_boost: float = 1.5              # recall strengthening factor
    prune_threshold: float = 0.05          # below = prune
    trivial_turn_threshold: int = 10       # chars; below = skip turn
    prefetch_mode: str = "bm25"            # "bm25" (fast) or "reranker" (accurate)
    tool_name: str = "synapse_query"       # merged tool name

    # Consolidation
    consolidation_interval_hours: float = 6.0

    @classmethod
    def from_env(cls) -> SynapseConfig:
        """Create config from environment variables (SYNAPSE_* prefix)."""
        return cls(
            falkordb_host=os.environ.get("SYNAPSE_FALKORDB_HOST", "localhost"),
            falkordb_port=int(os.environ.get("SYNAPSE_FALKORDB_PORT", "6379")),
            falkordb_password=os.environ.get("SYNAPSE_FALKORDB_PASSWORD"),
            falkordb_database=os.environ.get("SYNAPSE_DATABASE", "synapse"),
            llm_api_key=os.environ.get("SYNAPSE_LLM_API_KEY", ""),
            llm_base_url=os.environ.get("SYNAPSE_LLM_BASE_URL", ""),
            llm_model=os.environ.get("SYNAPSE_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get("SYNAPSE_EMBEDDING_MODEL", "text-embedding-3-small"),
            llm_client=os.environ.get("SYNAPSE_LLM_CLIENT", "generic"),
            llm_reasoning=os.environ.get("SYNAPSE_LLM_REASONING", "auto"),
            reranker_model=os.environ.get("SYNAPSE_RERANKER_MODEL", "gpt-4.1-nano"),
            batch_size=int(os.environ.get("SYNAPSE_BATCH_SIZE", "5")),
            half_life_days=float(os.environ.get("SYNAPSE_HALF_LIFE_DAYS", "7.0")),
            salience_boost=float(os.environ.get("SYNAPSE_SALIENCE_BOOST", "3.0")),
            recall_boost=float(os.environ.get("SYNAPSE_RECALL_BOOST", "1.5")),
            prune_threshold=float(os.environ.get("SYNAPSE_PRUNE_THRESHOLD", "0.05")),
            trivial_turn_threshold=int(
                os.environ.get("SYNAPSE_TRIVIAL_TURN_THRESHOLD", "10")
            ),
        )
