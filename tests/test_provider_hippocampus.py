"""Test that the provider wires the Hippocampus coordinator correctly."""

from synapse.provider import SynapseMemoryProvider


def test_provider_has_hippocampus_slot():
    p = SynapseMemoryProvider()
    assert p._hippocampus is None  # before init


def test_sync_turn_ticks_hippocampus():
    """sync_turn should advance the hippocampus turn counter."""
    p = SynapseMemoryProvider()
    p._initialized = True

    ticks = []

    class MockHippocampus:
        def tick(self):
            ticks.append(1)

    p._hippocampus = MockHippocampus()

    # Mock turn buffer — must be truthy even when empty.
    # ponytail: real TurnBuffer is always truthy (no __len__ falsy trap),
    # so the mock just needs __bool__ = True.
    class MockBuffer:
        def __init__(self):
            self._turns = []

        def __bool__(self):
            return True

        def add(self, u, a):
            self._turns.append((u, a))

        def should_flush(self):
            return False

        def __len__(self):
            return len(self._turns)

    p._turn_buffer = MockBuffer()
    p.sync_turn("What is Synapse?", "Synapse is a memory system.")

    assert len(ticks) == 1


def test_shutdown_clears_hippocampus():
    """shutdown should clear hippocampus session state."""
    p = SynapseMemoryProvider()
    p._initialized = True

    cleared = []

    class MockHippocampus:
        def clear(self):
            cleared.append(1)

    p._hippocampus = MockHippocampus()
    p._turn_buffer = None
    p._graphiti = None
    p._loop = None

    p.shutdown()
    assert len(cleared) == 1
    assert not p._initialized


def test_on_session_end_clears_hippocampus():
    p = SynapseMemoryProvider()
    p._initialized = True

    cleared = []

    class MockHippocampus:
        def clear(self):
            cleared.append(1)

    p._hippocampus = MockHippocampus()
    p._turn_buffer = None

    p.on_session_end([])
    assert len(cleared) == 1


# --- RIF ranking direction --------------------------------------------------

def test_rif_penalized_results_sink_and_relevance_is_preserved():
    """RIF must lower penalized entities, not raise them.

    The original key negated the penalty and relied on an ascending sort, so
    -0.3 sorted before 0.0 and penalized entities floated to the top —
    the exact inverse of retrieval-induced forgetting. It also sorted on
    penalty alone, discarding the relevance order underneath.
    """
    from synapse.hippocampus import Hippocampus
    from synapse.provider import SynapseMemoryProvider

    p = SynapseMemoryProvider()
    p._hippocampus = Hippocampus(group_id="test")
    # Docker/OrbStack lost a retrieval competition earlier in the session.
    p._hippocampus._rif_penalties = {"Docker": 0.3, "OrbStack": 0.3}

    # A strongly-matching but penalized fact, plus weak unpenalized ones.
    results = [
        {"from_node": "Darin", "to_node": "synapse", "fact": "strong-penalized",
         "score": 3},
        {"from_node": "Darin", "to_node": "mem0", "fact": "strong-clean",
         "score": 3},
        {"from_node": "Darin", "to_node": "Docker", "fact": "weak-penalized",
         "score": 1},
        {"from_node": "Darin", "to_node": "Colima", "fact": "weak-clean",
         "score": 1},
    ]
    p._hippocampus._rif_penalties = {"Docker": 0.3, "synapse": 0.3}

    def rif_rank(r):
        penalty = max(
            p._hippocampus.get_rif_penalty(r.get("from_node", "")),
            p._hippocampus.get_rif_penalty(r.get("to_node", "")),
        )
        return float(r.get("score") or 0) - penalty

    ranked = [
        r for _, r in sorted(
            enumerate(results), key=lambda pair: (-rif_rank(pair[1]), pair[0])
        )
    ]
    facts = [r["fact"] for r in ranked]

    # A penalized entity sinks below its equally-relevant clean peer...
    assert facts.index("strong-clean") < facts.index("strong-penalized")
    assert facts.index("weak-clean") < facts.index("weak-penalized")
    # ...but relevance still dominates: a strong penalized match must not be
    # demoted below a weak clean one. This is what the penalty-only sort broke.
    assert facts.index("strong-penalized") < facts.index("weak-clean")
    assert facts[0] == "strong-clean"
