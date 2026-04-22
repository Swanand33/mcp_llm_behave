"""The 3 tool implementations wrapping llm-behave."""

from llm_behave.engines.semantic import get_semantic_engine

_THRESHOLD_BEHAVIOR = 0.45
_THRESHOLD_DRIFT = 0.80


def run_behavior_test(prompt: str, expected_behavior: str, model_output: str) -> dict:
    """Run a single behavioral assertion using llm-behave embedding similarity.

    Checks whether the model output semantically satisfies the expected behavior.
    Uses sentence-level max similarity so long outputs are handled correctly.

    Args:
        prompt: The original prompt sent to the LLM (used for context/logging).
        expected_behavior: A plain-language description of what the output should do.
        model_output: The actual text returned by the LLM.

    Returns:
        dict with keys:
            score (float): Semantic similarity score, 0.0–1.0.
            passed (bool): True if score >= threshold.
            threshold (float): The threshold used for pass/fail.
    """
    engine = get_semantic_engine()
    score: float = engine.max_sentence_similarity(model_output, expected_behavior)
    return {
        "score": round(score, 4),
        "passed": score >= _THRESHOLD_BEHAVIOR,
        "threshold": _THRESHOLD_BEHAVIOR,
    }


def compare_outputs(baseline: str, candidate: str) -> dict:
    """Compare two LLM outputs for semantic similarity (regression detection).

    Useful for catching silent model regressions: run this in CI against a
    known-good baseline output to detect drift when you change prompts or models.

    Args:
        baseline: The reference/previous LLM output.
        candidate: The new LLM output to compare against baseline.

    Returns:
        dict with keys:
            similarity_score (float): Cosine similarity, 0.0–1.0.
            drift_detected (bool): True if score < threshold.
            interpretation (str): Human-readable summary of the result.
    """
    engine = get_semantic_engine()
    score: float = engine.similarity(baseline, candidate)

    if score >= 0.90:
        interpretation = "Outputs are nearly identical — no drift."
    elif score >= _THRESHOLD_DRIFT:
        interpretation = "Outputs are highly similar — within acceptable range."
    elif score >= 0.60:
        interpretation = "Moderate similarity — possible drift, review recommended."
    else:
        interpretation = "Low similarity — significant drift detected."

    return {
        "similarity_score": round(score, 4),
        "drift_detected": score < _THRESHOLD_DRIFT,
        "interpretation": interpretation,
    }


def list_builtin_behaviors() -> list:
    """Return the catalog of built-in behavioral checks available in llm-behave.

    Returns:
        list of dicts, each with 'name', 'method', and 'description' keys.
    """
    return [
        {
            "name": "mentions",
            "method": "assert_behavior(output).mentions(concept)",
            "description": (
                "Assert the output semantically mentions a concept. "
                "Uses embedding similarity, not exact string matching. Default threshold: 0.45."
            ),
        },
        {
            "name": "not_mentions",
            "method": "assert_behavior(output).not_mentions(concept)",
            "description": (
                "Assert the output does NOT semantically mention a concept. "
                "Useful for checking that sensitive topics are avoided."
            ),
        },
        {
            "name": "intent",
            "method": "assert_behavior(output).intent(intent)",
            "description": (
                "Assert the output matches a given intent or goal description. "
                "Checks whether the overall purpose of the response aligns."
            ),
        },
        {
            "name": "tone",
            "method": "assert_behavior(output).tone(tone)",
            "description": (
                "Assert the output has a specific tone (e.g. 'empathetic', 'formal', 'concise'). "
                "Compares embedding of output against the tone descriptor."
            ),
        },
        {
            "name": "calls_tool",
            "method": "assert_behavior(output, tool_calls=[...]).calls_tool(tool_name)",
            "description": (
                "Assert that the LLM response includes a specific tool call. "
                "Pass tool_calls from the API response alongside the text output."
            ),
        },
        {
            "name": "contradicts",
            "method": "assert_behavior(output).contradicts(other_text)",
            "description": (
                "Assert that the output contradicts another piece of text using NLI (Natural Language Inference). "
                "Useful for detecting policy reversals or conflicting statements. Added in llm-behave v0.1.2."
            ),
        },
    ]
