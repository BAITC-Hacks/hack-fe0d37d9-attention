"""Optional OpenAI salience selection over deterministic, auditable facts.

The LLM cannot write a number or call a numerical tool: its strict output is a
list of fact identifiers. Python renders the selected factual sentences. This
small MVP analysis layer deliberately favors grounded output over free prose.
"""
from __future__ import annotations

import json
import os

import requests


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
        return dict(status="skipped", reason="OPENAI_API_KEY not configured")
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
    try:
        response = (session or requests.Session()).post(
            "https://api.openai.com/v1/responses", json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=45,
        )
        if not response.ok:
            raise ValueError("OpenAI request failed")
        payload = response.json()
        if payload.get("status") != "completed":
            raise ValueError("Incomplete OpenAI response")
        text = "".join(part["text"] for item in payload.get("output", []) if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        selection = json.loads(text)
        if set(selection) != {"fact_ids"} or not isinstance(selection["fact_ids"], list):
            raise ValueError("Invalid analysis schema")
        ids = selection["fact_ids"]
        if not ids or len(ids) > 6 or any(not isinstance(key, str) or key not in facts for key in ids):
            raise ValueError("Unsupported analysis facts")
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        # Do not persist request headers, credentials, provider bodies, or raw error text.
        raise RuntimeError("Optional OpenAI analysis failed; deterministic forecast is unchanged") from None
    ids = list(dict.fromkeys(ids + ["limitations"]))
    return dict(status="completed", model=model, selected_fact_ids=ids,
                text="\n".join(facts[key] for key in ids), method="LLM salience selection; deterministic fact rendering")
