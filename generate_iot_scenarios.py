#!/usr/bin/env python3
"""
IoT Scenario Generator for AssetOpsBench Cache Testing
=======================================================
Generates new IoT scenarios at three built-in similarity levels for testing
natural language query caching systems:

  HIGH   (~33%) — Same intent/entity/params, rephrased text.
                   These are strong cache-HIT candidates.
  MEDIUM (~33%) — Same operation class, different entities or parameters.
                   These are ambiguous / weak-hit candidates.
  LOW    (~33%) — Same IoT domain, entirely different operation class.
                   These are expected cache-MISS candidates.

All generated scenarios follow the AssetOpsBench utterance design guidelines
(type, category, deterministic, characteristic_form, group, entity, note).

Uses the same LiteLLM + IBM WatsonX stack as the rest of the project.
The model can be overridden by setting the MODEL constant below.

Requirements:
    litellm is already a project dependency (uv sync / pip install -e .)

Usage:
    # Set the same env vars used by the rest of the project:
    export WATSONX_APIKEY=...
    export WATSONX_PROJECT_ID=...
    export WATSONX_URL=...        # optional
    python generate_iot_scenarios.py

Inputs:
    iot_scenarios_unaugmented.csv  (must exist in the same directory)

Outputs:
    iot_scenarios_generated.csv    (new file, includes a similarity_tier column)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

try:
    import litellm
except ImportError:
    sys.exit(
        "Error: 'litellm' package not found.\n"
        "It should already be a project dependency — run:  uv sync"
    )

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_CSV  = Path(__file__).parent / "iot_scenarios_unaugmented.csv"
OUTPUT_CSV = Path(__file__).parent / "iot_scenarios_generated.csv"

# Same default model used by the plan-execute workflow (cli.py)
MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

# How many scenarios to target per similarity tier.
# Total output ≈ 3 × SCENARIOS_PER_TIER (minus any deduplication drops).
SCENARIOS_PER_TIER = 10

# String-similarity threshold used for deduplication.
# Any candidate whose normalised SequenceMatcher ratio vs. an existing text
# exceeds this value is discarded as a near-duplicate.
DEDUP_THRESHOLD = 0.82

# ── Domain context (shared across all prompts) ─────────────────────────────────

_SYSTEM_CONTEXT = """
Known assets and sensors at site MAIN:
  Chillers  : Chiller 3, Chiller 6, Chiller 9
    Sensors : Tonnage, % Loaded, Power Input, Supply Temperature,
              Return Temperature, Condenser Water Flow
  AHUs      : CQPA AHU 1, CQPA AHU 2B
    Sensors : Supply Temperature, Return Temperature,
              Supply Humidity, Power Consumption
  Data spans roughly 2015–2020.
  The IoT agent/MCP server is always referred to with type "IOT".
"""

# ── Utterance design guidelines (embedded in every prompt) ────────────────────

_GUIDELINES = """
AssetOpsBench Utterance Design Guidelines
------------------------------------------
Return a JSON ARRAY. Each object must have EXACTLY these keys (no extras):

  text               – Natural language query an operator would realistically ask.
  category           – One of:
                         "Knowledge Query"           (direct lookup, factual)
                         "Data Query"                (time-bounded data retrieval)
                         "Analysis & Inference"      (model/AI analysis required)
                         "Anomaly & Exception Detection"
                         "Recommendation & Optimization"
  deterministic      – Boolean.
                         true  → single verifiable correct answer
                                 (data retrieval, counts, specific readings)
                         false → multiple valid responses acceptable
                                 (analysis, recommendations)
  characteristic_form – String. Describes what a correct response looks like.
                         Deterministic  : state exact format / expected content.
                         Non-deterministic: state acceptance criteria.
                         Be specific — avoid vague phrases like "the response should return data."
  group              – One of: "retrospective" | "predictive" | "prescriptive"
  entity             – Primary physical thing queried: "Site" | "Chiller" | "AHU" | "Sensor"
  note               – Follow this template exactly:
                         "Source: IoT data operations; <Deterministic|Non-deterministic> query
                          with <single correct answer|multiple valid responses>;
                          Category: <category>"

Do NOT include an "id" field — IDs are assigned by the script.
Do NOT wrap the array in any outer object.
Output ONLY the JSON array, with no markdown fences or commentary.
"""

# ── Prompt templates ───────────────────────────────────────────────────────────

_HIGH_PROMPT = """\
Generate {n} IoT scenarios with HIGH SIMILARITY to the existing scenarios below.

HIGH SIMILARITY means:
  • Keep the same entity, operation type, and parameters (asset name, sensor,
    time range, site) as one of the existing scenarios.
  • Change ONLY the natural language phrasing — use synonyms, reorder clauses,
    switch between question and imperative forms, vary formality, etc.
  • Purpose: these should be strong cache-HIT candidates when run through a
    natural language query cache alongside their originals.

Constraints:
  • Do NOT reproduce the exact text of any existing scenario.
  • Do NOT generate two entries with the same meaning within your response.
  • Each generated text must differ meaningfully in wording from all others.

{guidelines}

{system_context}

Existing scenarios (paraphrase these):
{existing}
"""

_MEDIUM_PROMPT = """\
Generate {n} IoT scenarios with MEDIUM SIMILARITY to the existing scenarios below.

MEDIUM SIMILARITY means:
  • Use the same operation class / category as scenarios already present
    (e.g., sensor-data retrieval, asset metadata lookup, asset listing).
  • Vary the SPECIFIC parameters: use a different asset, a different sensor name,
    or a different time range from those already covered.
  • Purpose: ambiguous cache cases — the operation is familiar but parameters differ.

Constraints:
  • Do NOT duplicate any existing scenario's text or its exact combination of
    asset + sensor + time range.
  • You MAY reference the same assets (Chiller 3, Chiller 6, etc.) as long as
    the sensor or time period is different and not already covered.
  • Each generated text must be unique.

{guidelines}

{system_context}

Existing scenarios (use these as context for operation type, but vary params):
{existing}
"""

_LOW_PROMPT = """\
Generate {n} IoT scenarios with LOW SIMILARITY to the existing scenarios below.

LOW SIMILARITY means:
  • Stay within the IoT building-management domain (HVAC / chillers / AHUs at MAIN).
  • Cover operation classes NOT represented in the existing set, such as:
      – Cross-asset comparisons  ("compare tonnage between Chiller 6 and Chiller 9")
      – Aggregation queries      ("average power across all AHUs in April 2020")
      – Sensor enumeration       ("list all sensors available for Chiller 3")
      – Threshold / condition checks ("when did Chiller 6 exceed 300 tonnes?")
      – Data quality / completeness ("are there gaps in Chiller 9 data for June 2020?")
      – Multi-metric combined queries
      – Trend or rate-of-change queries
  • Purpose: expected cache-MISS candidates — the cache should NOT return a hit.

Constraints:
  • Do NOT duplicate any existing scenario.
  • Keep queries realistic for a building operator or data analyst.
  • Each generated text must be unique.

{guidelines}

{system_context}

Existing scenarios (reference only — generate operationally different queries):
{existing}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_existing(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def format_existing(scenarios: list[dict]) -> str:
    lines = []
    for s in scenarios:
        lines.append(
            f'  [{s["category"]}] "{s["text"]}"'
            f' | entity={s["entity"]}'
            f', deterministic={s["deterministic"]}'
            f', group={s.get("group", "")}'
        )
    return "\n".join(lines)


def is_near_duplicate(candidate: str, corpus: list[str], threshold: float) -> bool:
    c = candidate.lower().strip()
    for existing in corpus:
        ratio = SequenceMatcher(None, c, existing.lower().strip()).ratio()
        if ratio >= threshold:
            return True
    return False


def call_llm(prompt: str) -> list[dict]:
    """Call the model via LiteLLM and return a parsed list of scenario dicts."""
    kwargs: dict = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.7,   # some creativity without going off-rails
        "max_tokens":  4096,
    }

    if MODEL.startswith("watsonx/"):
        kwargs["api_key"]    = os.environ["WATSONX_APIKEY"]
        kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
        if url := os.environ.get("WATSONX_URL"):
            kwargs["api_base"] = url
    else:
        kwargs["api_key"]  = os.environ["LITELLM_API_KEY"]
        kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

    response = litellm.completion(**kwargs)
    raw = response.choices[0].message.content.strip()

    # Robustly extract the JSON array even if the model adds any prose
    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        print("  Warning: response contained no JSON array.", file=sys.stderr)
        return []

    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        print(f"  Warning: JSON parse error — {exc}", file=sys.stderr)
        return []

    if not isinstance(parsed, list):
        print("  Warning: parsed JSON is not a list.", file=sys.stderr)
        return []

    return parsed


def generate_tier(template: str, n: int, existing: list[dict]) -> list[dict]:
    prompt = template.format(
        n=n,
        guidelines=_GUIDELINES,
        system_context=_SYSTEM_CONTEXT,
        existing=format_existing(existing),
    )
    return call_llm(prompt)


def build_csv_row(scenario: dict, assigned_id: int, tier: str) -> dict:
    text = str(scenario.get("text", "")).strip()
    cf   = str(scenario.get("characteristic_form", "")).strip()
    return {
        "id":                  assigned_id,
        "type":                "IOT",
        "text":                text,
        "category":            scenario.get("category", "Knowledge Query"),
        "deterministic":       scenario.get("deterministic", True),
        "characteristic_form": cf,
        "group":               scenario.get("group", "retrospective"),
        "entity":              scenario.get("entity", ""),
        "note":                scenario.get("note", ""),
        "text_len":            len(text),
        "cf_len":              len(cf),
        "similarity_tier":     tier,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"Error: input file not found — {INPUT_CSV}")

    # Validate required credentials based on the selected model prefix
    if MODEL.startswith("watsonx/"):
        missing = [v for v in ("WATSONX_APIKEY", "WATSONX_PROJECT_ID") if not os.environ.get(v)]
        if missing:
            sys.exit(
                f"Error: missing environment variable(s): {', '.join(missing)}\n"
                "These are the same vars used by the rest of the project — "
                "check your .env file."
            )
    else:
        if not os.environ.get("LITELLM_API_KEY"):
            sys.exit("Error: LITELLM_API_KEY environment variable is not set.")

    # ── Load existing scenarios ────────────────────────────────────────────────
    print(f"Loading existing scenarios from {INPUT_CSV.name} …")
    existing = load_existing(INPUT_CSV)
    existing_texts = [s["text"] for s in existing]
    next_id = max(int(s["id"]) for s in existing) + 1
    print(f"  {len(existing)} existing scenarios loaded  (IDs up to {next_id - 1})")

    # Corpus grows as we accept new scenarios, preventing intra-run duplicates
    live_corpus = list(existing_texts)
    all_rows: list[dict] = []

    tiers = [
        ("high",   _HIGH_PROMPT),
        ("medium", _MEDIUM_PROMPT),
        ("low",    _LOW_PROMPT),
    ]

    for tier, template in tiers:
        print(f"\nGenerating {SCENARIOS_PER_TIER} {tier.upper()}-similarity scenarios …")
        candidates = generate_tier(template, SCENARIOS_PER_TIER, existing)
        print(f"  Received {len(candidates)} candidates from model")

        accepted: list[dict] = []
        for raw in candidates:
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            if is_near_duplicate(text, live_corpus, DEDUP_THRESHOLD):
                print(f"  Skipped (near-duplicate): "{text[:70]}…"")
                continue
            accepted.append(raw)
            live_corpus.append(text)

        print(f"  Accepted {len(accepted)} after deduplication")

        for raw in accepted:
            all_rows.append(build_csv_row(raw, next_id, tier))
            next_id += 1

    if not all_rows:
        sys.exit("No scenarios were generated. Check your API key and model access.")

    # ── Write output CSV ───────────────────────────────────────────────────────
    fieldnames = [
        "id", "type", "text", "category", "deterministic",
        "characteristic_form", "group", "entity", "note",
        "text_len", "cf_len", "similarity_tier",
    ]

    print(f"\nWriting {len(all_rows)} scenarios to {OUTPUT_CSV.name} …")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    counts = Counter(r["similarity_tier"] for r in all_rows)
    print("\nDone.")
    print(f"  High   similarity : {counts.get('high',   0):>3} scenarios")
    print(f"  Medium similarity : {counts.get('medium', 0):>3} scenarios")
    print(f"  Low    similarity : {counts.get('low',    0):>3} scenarios")
    print(f"  Total             : {len(all_rows):>3} scenarios")
    print(f"  Output            : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
