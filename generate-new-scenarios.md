# Generating New IoT Scenarios

`generate_iot_scenarios.py` is a standalone script that uses an LLM to generate new IoT benchmark scenarios for testing natural language query caching systems. It reads from the existing scenario bank, produces new scenarios that are non-redundant with it, and automatically distributes them across three built-in similarity tiers.

## Why three tiers?

When testing a query cache, you need scenarios across the full spectrum of semantic similarity to the cached queries:

| Tier | What it means | Cache testing purpose |
|------|---------------|-----------------------|
| **High** (~33%) | Same entity, operation, and parameters — only the phrasing changes | Strong cache-**HIT** candidates |
| **Medium** (~33%) | Same operation class, but different asset, sensor, or time range | Ambiguous / weak-hit candidates |
| **Low** (~33%) | Same IoT domain, but a completely different operation class | Expected cache-**MISS** candidates |

The split is built into the script — there is no knob to turn. Each run produces roughly equal numbers of each tier.

## Prerequisites

The script uses the same LiteLLM + IBM WatsonX stack as the rest of the project. No extra packages need to be installed — `litellm` is already a project dependency.

```bash
uv sync        # installs all project dependencies including litellm
```

## Environment variables

These are the same variables already used by the `plan-execute` workflow. If you have already set them up (e.g. via your `.env` file), nothing extra is needed.

| Variable | Required | Description |
|----------|----------|-------------|
| `WATSONX_APIKEY` | Yes | Your IBM WatsonX API key |
| `WATSONX_PROJECT_ID` | Yes | WatsonX project ID (already in `.env`) |
| `WATSONX_URL` | No | WatsonX endpoint — defaults to `us-south` if omitted |

## Running the script

```bash
export WATSONX_APIKEY=your-key-here
python generate_iot_scenarios.py
```

The script prints progress as it goes:

```
Loading existing scenarios from iot_scenarios_unaugmented.csv …
  21 existing scenarios loaded  (IDs up to 48)

Generating 10 HIGH-similarity scenarios …
  Received 10 candidates from model
  Accepted 10 after deduplication

Generating 10 MEDIUM-similarity scenarios …
  Received 10 candidates from model
  Accepted 9 after deduplication

Generating 10 LOW-similarity scenarios …
  Received 10 candidates from model
  Accepted 10 after deduplication

Writing 29 scenarios to iot_scenarios_generated.csv …

Done.
  High   similarity :  10 scenarios
  Medium similarity :   9 scenarios
  Low    similarity :  10 scenarios
  Total             :  29 scenarios
  Output            : .../iot_scenarios_generated.csv
```

## Input and output files

| File | Role |
|------|------|
| `iot_scenarios_unaugmented.csv` | Source of existing scenarios — read-only, never modified |
| `iot_scenarios_generated.csv` | Generated output — created fresh on every run |

The output file has the same schema as the input, with one additional column:

| Column | Description |
|--------|-------------|
| `id` | Continues from the highest existing ID + 1 |
| `type` | Always `IOT` |
| `text` | The natural language query |
| `category` | One of: Knowledge Query, Data Query, Analysis & Inference, Anomaly & Exception Detection, Recommendation & Optimization |
| `deterministic` | `True` if there is exactly one correct answer, `False` otherwise |
| `characteristic_form` | Description of what a correct response looks like |
| `group` | `retrospective`, `predictive`, or `prescriptive` |
| `entity` | Primary physical thing being queried (Site, Chiller, AHU, Sensor) |
| `note` | Source, determinism, and category metadata |
| `text_len` | Character length of `text` |
| `cf_len` | Character length of `characteristic_form` |
| `similarity_tier` | `high`, `medium`, or `low` |

All generated scenarios follow the [AssetOpsBench utterance design guidelines](https://github.com/IBM/AssetOpsBench/blob/extra_scenarios/experimental_scenarios/utterance_design_guideline.md).

## Deduplication

Before accepting any generated scenario, the script computes a normalised string-similarity ratio (via Python's `difflib.SequenceMatcher`) against every scenario already seen — both the original 21 and any scenarios already accepted in the current run. Candidates above the threshold are silently dropped.

## Tuning the output

There are three constants at the top of the script you can adjust if needed:

```python
# generate_iot_scenarios.py

MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# Any litellm-compatible model string works here, e.g.:
#   "watsonx/ibm/granite-3-3-8b-instruct"
#   "litellm_proxy/GCP/claude-4-sonnet"

SCENARIOS_PER_TIER = 10
# Target number of scenarios per tier. Total output ≈ 3 × this value.

DEDUP_THRESHOLD = 0.82
# String-similarity cutoff (0.0–1.0). Lower = more permissive (keeps more);
# higher = stricter (drops more near-duplicates).
```

## Merging into the main scenario bank

The output CSV is intentionally separate so you can review it before merging. Once satisfied, append it to the existing bank however suits your workflow — for example:

```bash
# Skip the header row of the generated file before appending
tail -n +2 iot_scenarios_generated.csv >> iot_scenarios_unaugmented.csv
```
