"""Evidence-grounded explanations with an optional OpenAI-compatible API."""
from __future__ import annotations

import json
import os
from typing import Any

import requests


def _observed(value: Any, label: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{label} is unavailable"
    if not 0 <= number <= 100:
        return f"{label} is outside its valid observed range"
    return f"{label} is {number:.1f}%"


def rule_based_explanation(evidence: dict[str, Any]) -> dict[str, Any]:
    """Explain only supplied telemetry, predictions, performance and SHAP evidence."""
    statements: list[str] = []
    recommendations: list[str] = []
    for target in ("cpu", "ram"):
        current = evidence.get(f"current_{target}_percent")
        predicted = evidence.get(f"predicted_{target}_percent")
        if current is None or predicted is None:
            continue
        current_text = _observed(current, f"Current {target.upper()}")
        try:
            delta = float(predicted) - float(current)
        except (TypeError, ValueError):
            statements.append(f"{current_text}; its 30-second prediction is unavailable.")
            continue
        if not (0 <= float(predicted) <= 100):
            statements.append(f"{current_text}; the model returned an out-of-range {target.upper()} prediction, so do not interpret it as a valid utilization percentage.")
            continue
        trend = "increase" if delta >= 1 else "decrease" if delta <= -1 else "remain broadly stable"
        statements.append(f"{current_text}; the experimental model predicts {float(predicted):.1f}% in 30 seconds ({trend}, {delta:+.1f} percentage points).")
        if target == "cpu" and float(predicted) >= 85:
            recommendations.append("If the high CPU reading persists, inspect active applications in Task Manager and close only work you recognize and no longer need.")
        if target == "ram" and float(predicted) >= 90:
            recommendations.append("If high memory use persists, save work and review memory-heavy applications before closing anything.")
    if not statements:
        statements.append("Current observations and model predictions are unavailable, so no trend conclusion is supported.")
    contributions = evidence.get("top_shap", [])
    if contributions:
        formatted = []
        for item in contributions[:5]:
            try:
                direction = "raises" if float(item["mean_shap"]) > 0 else "lowers" if float(item["mean_shap"]) < 0 else "has near-zero average direction for"
                target = str(item.get("target", "resource" )).upper()
                formatted.append(f"{item['feature']} {direction} the {target} model output on average")
            except (KeyError, TypeError, ValueError):
                continue
        if formatted:
            statements.append("Across the explained test rows, SHAP attributes model output to " + "; ".join(formatted) + ". These are model associations, not causal effects.")
    quality = evidence.get("model_quality", {})
    for target in ("cpu", "ram"):
        metric = quality.get(target, {})
        if metric and float(metric.get("r2", 0)) < 0:
            statements.append(f"The selected {target.upper()} model had negative held-out test R-squared ({float(metric['r2']):.3f}), so its forecast has not outperformed a test-mean reference on that metric.")
    if not recommendations:
        recommendations.append("Continue monitoring and compare changes with the observed telemetry; no urgent action is supported by the supplied evidence alone.")
    statements.append("These models were evaluated on one machine and a short collection period; treat forecasts as experimental, not a diagnosis or a guarantee.")
    return {"source": "rule_based", "explanation": " ".join(statements), "recommendations": recommendations}


def explain(evidence: dict[str, Any], timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Use an API only when explicitly configured; otherwise use deterministic rules."""
    api_key = os.environ.get("OS_PERFORMANCE_LLM_API_KEY")
    if not api_key:
        return rule_based_explanation(evidence)
    base_url = os.environ.get("OS_PERFORMANCE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OS_PERFORMANCE_LLM_MODEL", "gpt-4o-mini")
    prompt = {
        "instruction": (
            "Explain only the supplied Windows resource observations, experimental predictions, held-out metrics, "
            "and SHAP summary. Do not claim causation, do not invent observations, note negative R2/weakness, "
            "and give practical low-risk recommendations. Return JSON with explanation and recommendations (array)."
        ),
        "evidence": evidence,
    }
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": json.dumps(prompt)}]},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
        if not isinstance(result.get("explanation"), str) or not isinstance(result.get("recommendations"), list):
            raise ValueError("LLM response did not match the required output shape")
        result["source"] = "api"
        return result
    except (requests.RequestException, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        fallback = rule_based_explanation(evidence)
        fallback["api_error"] = str(exc)
        return fallback
