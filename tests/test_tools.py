"""Tests for the 3 tool implementations."""

import pytest

from mcp_llm_behave.tools import (
    _MAX_INPUT_CHARS,
    compare_outputs,
    list_builtin_behaviors,
    run_behavior_test,
)


# --- run_behavior_test ---

def test_run_behavior_test_happy():
    result = run_behavior_test(
        prompt="Explain what a refund policy is.",
        expected_behavior="mentions refund policy",
        model_output="Our refund policy allows returns within 30 days for a full refund.",
    )
    assert result["passed"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert result["threshold"] == 0.45


def test_run_behavior_test_low_similarity():
    result = run_behavior_test(
        prompt="Explain quantum physics.",
        expected_behavior="discusses cooking recipes",
        model_output="Quantum mechanics describes the behavior of particles at subatomic scales.",
    )
    assert result["passed"] is False
    assert result["score"] < 0.45


def test_run_behavior_test_returns_correct_keys():
    result = run_behavior_test(
        prompt="any prompt",
        expected_behavior="any behavior",
        model_output="any output",
    )
    assert set(result.keys()) == {"score", "passed", "threshold"}


# --- compare_outputs ---

def test_compare_outputs_identical():
    text = "The capital of France is Paris."
    result = compare_outputs(text, text)
    assert result["drift_detected"] is False
    assert result["similarity_score"] > 0.99


def test_compare_outputs_drift_detected():
    result = compare_outputs(
        baseline="The capital of France is Paris, a city known for the Eiffel Tower.",
        candidate="Python is a programming language used for data science and web development.",
    )
    assert result["drift_detected"] is True
    assert result["similarity_score"] < 0.80


def test_compare_outputs_returns_correct_keys():
    result = compare_outputs("hello world", "hello there")
    assert set(result.keys()) == {"similarity_score", "drift_detected", "interpretation"}
    assert isinstance(result["interpretation"], str)


# --- input validation (hardening) ---

def test_empty_model_output_raises():
    with pytest.raises(ValueError, match="model_output"):
        run_behavior_test(prompt="p", expected_behavior="something", model_output="")


def test_whitespace_only_expected_behavior_raises():
    with pytest.raises(ValueError, match="expected_behavior"):
        run_behavior_test(prompt="p", expected_behavior="   ", model_output="some output")


def test_oversized_input_raises():
    big = "x" * (_MAX_INPUT_CHARS + 1)
    with pytest.raises(ValueError, match="exceeds"):
        run_behavior_test(prompt="p", expected_behavior="something", model_output=big)


def test_empty_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        compare_outputs(baseline="", candidate="some output")


def test_empty_candidate_raises():
    with pytest.raises(ValueError, match="candidate"):
        compare_outputs(baseline="some output", candidate="  ")


def test_score_always_in_range():
    """Score must always be clamped to [0.0, 1.0] regardless of input."""
    result = run_behavior_test(
        prompt="p",
        expected_behavior="completely unrelated concept xyz123",
        model_output="The weather is nice today in the park.",
    )
    assert 0.0 <= result["score"] <= 1.0

    result2 = compare_outputs(
        baseline="completely different aardvark topic",
        candidate="The weather is nice today in the park.",
    )
    assert 0.0 <= result2["similarity_score"] <= 1.0


# --- list_builtin_behaviors ---

def test_list_builtin_behaviors_returns_list():
    result = list_builtin_behaviors()
    assert isinstance(result, list)


def test_list_builtin_behaviors_not_empty():
    result = list_builtin_behaviors()
    assert len(result) > 0


def test_list_builtin_behaviors_structure():
    result = list_builtin_behaviors()
    for item in result:
        assert "name" in item
        assert "method" in item
        assert "description" in item
