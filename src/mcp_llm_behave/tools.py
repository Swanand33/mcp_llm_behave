"""The 3 tool implementations wrapping llm-behave."""

import logging

from llm_behave.engines.semantic import get_semantic_engine

logger = logging.getLogger(__name__)

_THRESHOLD_BEHAVIOR = 0.45
_THRESHOLD_DRIFT = 0.80
_MAX_INPUT_CHARS = 10_000


def _validate(value: str, name: str) -> str:
    """Validate that a text input is non-empty and within the character limit."""
    stripped = value.strip() if isinstance(value, str) else ""
    if not stripped:
        raise ValueError(f"'{name}' must not be empty or whitespace-only.")
    if len(stripped) > _MAX_INPUT_CHARS:
        raise ValueError(
            f"'{name}' is {len(stripped):,} chars — exceeds the "
            f"{_MAX_INPUT_CHARS:,}-char limit. Truncate before testing."
        )
    return stripped


def _clamp(score: float) -> float:
    """Clamp cosine similarity to [0.0, 1.0] and round to 4 decimal places."""
    return round(max(0.0, min(1.0, score)), 4)


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
    expected_behavior = _validate(expected_behavior, "expected_behavior")
    model_output = _validate(model_output, "model_output")

    logger.debug("run_behavior_test | expected=%r | output_len=%d", expected_behavior, len(model_output))

    engine = get_semantic_engine()
    score = _clamp(engine.max_sentence_similarity(model_output, expected_behavior))
    return {
        "score": score,
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
            drift_detected (bool): True if score < threshold (0.80).
            interpretation (str): Human-readable summary of the result.
    """
    baseline = _validate(baseline, "baseline")
    candidate = _validate(candidate, "candidate")

    logger.debug("compare_outputs | baseline_len=%d | candidate_len=%d", len(baseline), len(candidate))

    engine = get_semantic_engine()
    score = _clamp(engine.similarity(baseline, candidate))

    if score >= 0.90:
        interpretation = "Outputs are nearly identical — no drift."
    elif score >= _THRESHOLD_DRIFT:
        interpretation = "Outputs are highly similar — within acceptable range."
    elif score >= 0.60:
        interpretation = "Moderate similarity — possible drift, review recommended."
    else:
        interpretation = "Low similarity — significant drift detected."

    return {
        "similarity_score": score,
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
