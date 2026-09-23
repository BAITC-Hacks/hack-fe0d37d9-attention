"""Optional OpenAI salience selection over deterministic, auditable facts.

The LLM cannot write a number or call a numerical tool: its strict output is a
list of fact identifiers. Python renders the selected factual sentences. This
small MVP analysis layer deliberately favors grounded output over free prose.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone

import requests


ANALYSIS_FAILURE = "Optional OpenAI analysis failed; deterministic forecast is unchanged"
USAGE_DETAILS = {
    "input_tokens_details": ("cached_tokens", "cache_write_tokens", "text_tokens", "audio_tokens", "image_tokens"),
    "output_tokens_details": ("reasoning_tokens", "text_tokens", "audio_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"),
}


class OpenAIAnalysisError(RuntimeError):
    """Carries allowlisted audit metadata, never a provider body or raw exception."""

    def __init__(self, openai: dict):
        super().__init__(ANALYSIS_FAILURE)
        self.openai = openai


def skipped_openai(reason: str) -> dict:
    return dict(called=False, endpoint="/v1/responses", status="skipped", reason=reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_label(value) -> str | None:
    # IDs/model labels only, not free-form provider text, headers, or error bodies.
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value) and not value.startswith(("sk-", "sess-", "Bearer")):
        return value
    return None


def _token_count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _safe_usage(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    usage = {name: _token_count(raw.get(name)) for name in ("input_tokens", "output_tokens", "total_tokens")}
    for group, names in USAGE_DETAILS.items():
        details = raw.get(group)
        if isinstance(details, dict):
            usage[group] = {name: _token_count(details[name]) for name in names if name in details}
    return usage


def forecast_facts(summary: dict) -> dict[str, str]:
    facts = {}
    for turbine, values in summary["turbines"].items():
        facts[f"{turbine}_means"] = (
            f"{turbine}: mean predicted normalized power is {values['mean_power_24h']:.3f} "
            f"over the next 24 hours and {values['mean_power_48h']:.3f} over 48 hours."
        )
        facts[f"{turbine}_peak"] = (
            f"{turbine}: peak predicted normalized power is {values['peak_power']:.3f} "
            f"at {values['peak_time_local']} local time."
        )
        low = values["lowest_3h_period"]
        facts[f"{turbine}_low"] = (
            f"{turbine}: the lowest three-hour forecast block begins at {low['start_local']} "
            f"with mean normalized power {low['mean_power']:.3f}."
        )
        for ramp in values["ramps"]:
            facts[f"{turbine}_{ramp['kind']}"] = (
                f"{turbine}: {ramp['kind'].replace('_', ' ')} candidate from {ramp['from_local']} "
                f"to {ramp['to_local']}, forecast change {ramp['change_normalized_power']:+.3f}."
            )
    if summary.get("T1_minus_T2"):
        facts["comparison"] = (
            "T1 minus T2 next-24-hour mean normalized power: "
            f"{summary['T1_minus_T2']['mean_power_24h']:+.3f}."
        )
    facts["recalculation"] = (
        "This is a recalculation using the frozen pre-February model; prior forecasts remain preserved."
        if summary.get("recalculation") else "This forecast uses the frozen pre-February model."
    )
    facts["limitations"] = " ".join(summary["warnings"])
    return facts


def analyze_forecast_with_llm(summary: dict, session=None) -> dict:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        reason = "OPENAI_API_KEY not configured"
        return dict(status="skipped", reason=reason, openai=skipped_openai(reason))
    facts = forecast_facts(summary)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    body = dict(
        model=model, store=False, max_output_tokens=500,
        instructions=("You are a forecast analyst. Select up to six supplied fact IDs in useful operational order. "
                      "Include limitations. Never calculate values, infer causal certainty, or request tool execution. "
                      "Python will render the supplied facts unchanged."),
        input=json.dumps(dict(forecast_origin=summary["forecast_origin_local"],
                             model_name=summary["model_name"], facts=facts)),
        text={"format": {"type": "json_schema", "name": "forecast_fact_selection", "strict": True,
            "schema": {"type": "object", "properties": {
                "fact_ids": {"type": "array", "items": {"type": "string", "enum": list(facts)}}},
                "required": ["fact_ids"], "additionalProperties": False}}},
    )
    client = session or requests.Session()
    audit = dict(called=True, endpoint="/v1/responses", response_id=None, openai_request_id=None,
                 model=None, requested_model=_safe_label(model), status="failed", response_status=None,
                 created_at=None, request_started_at_utc=_utc_now(), request_completed_at_utc=None,
                 latency_ms=None, usage=_safe_usage(None), http_status_code=None,
                 analysis_type="forecast_summary", prompt_template_version="v1", facts_supplied_count=len(facts))
    started = time.monotonic()
    failure_type = "request_error"
    try:
        try:
            response = client.post(
                "https://api.openai.com/v1/responses", json=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=45,
            )
        finally:
            audit["latency_ms"] = round(max(0.0, time.monotonic() - started) * 1000, 3)
            audit["request_completed_at_utc"] = _utc_now()
        audit["http_status_code"] = _token_count(getattr(response, "status_code", None))
        headers = getattr(response, "headers", None)
        if headers is not None:
            audit["openai_request_id"] = _safe_label(headers.get("x-request-id"))
        failure_type = "invalid_response_json" if response.ok else "http_error"
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Invalid response object")
        audit.update(response_id=_safe_label(payload.get("id")), model=_safe_label(payload.get("model")),
                     usage=_safe_usage(payload.get("usage")))
        if payload.get("status") in ("completed", "failed", "in_progress", "cancelled", "queued", "incomplete"):
            audit["response_status"] = payload["status"]
        created = payload.get("created_at")
        if type(created) in (int, float) and math.isfinite(created):
            audit["created_at"] = created
        if not response.ok:
            raise ValueError("OpenAI request failed")
        failure_type = "response_not_completed"
        if payload.get("status") != "completed":
            raise ValueError("Incomplete OpenAI response")
        failure_type = "invalid_fact_selection"
        text = "".join(part["text"] for item in payload.get("output", []) if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        selection = json.loads(text)
        if set(selection) != {"fact_ids"} or not isinstance(selection["fact_ids"], list):
            raise ValueError("Invalid analysis schema")
        ids = selection["fact_ids"]
        if not ids or len(ids) > 6 or any(not isinstance(key, str) or key not in facts for key in ids):
            raise ValueError("Unsupported analysis facts")
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        # Do not persist request headers, credentials, provider bodies, or raw error text.
        audit.update(error_type=failure_type, error_message_safe=ANALYSIS_FAILURE)
        raise OpenAIAnalysisError(audit) from None
    audit["status"] = "completed"
    ids = list(dict.fromkeys(ids + ["limitations"]))
    return dict(status="completed", model=model, selected_fact_ids=ids,
                text="\n".join(facts[key] for key in ids), method="LLM salience selection; deterministic fact rendering",
                openai=audit)
