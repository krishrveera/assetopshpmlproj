import asyncio
import json
import pandas as pd
import litellm

# Import the actual AssetOpsBench Planner modules (run for reverse engineering))
from src.workflow.runner import PlanExecuteRunner
from src.llm import LiteLLMBackend

# 1. Define our Base Queries (Constrained to Chiller 6 / MAIN to work locally)
BASE_QUERIES = [
    {
        "id": 1, "category": "Knowledge Query",
        "text": "What IoT sites are available?"
    },
    {
        "id": 4, "category": "Knowledge Query",
        "text": "Which assets are located at the MAIN facility?"
    },
    {
        "id": 47, "category": "Knowledge Query",
        "text": "'list all the chillers at site MAIN"
    }
]

# 2. Setup the Local Planner
# Uses the exact Litellm backend specified in the AssetOpsBench architecture
my_llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
runner = PlanExecuteRunner(llm=my_llm)

# 3. The True Reverse-Engineering Prompt
SYSTEM_PROMPT = """
You are an expert industrial data scientist. 
I am going to provide you with the EXACT execution plan an AI agent generated to answer a specific industrial query.

Your task is to generate 3 BRAND NEW, diverse ways a human operator might ask a question that would result in this EXACT SAME execution plan. 

Vary the urgency, the phrasing, and the persona (e.g., a stressed technician vs. a routine auditor). 
DO NOT change the core entities (It must remain Chiller 6 and the MAIN site).

Original Query that generated this plan: "{original_query}"

The AI's Execution Plan:
{actual_plan}

Output strictly as a JSON object containing a list called "new_scenarios". Each object in the list must have:
- "text": The new human question.
- "characteristic_form": The evaluation rubric (Start with "The expected response should be...").
"""

async def generate_from_real_trace():
    all_new_scenarios = []
    starting_id = 500

    for base in BASE_QUERIES:
        print(f"\n[RUNNING PLANNER] Testing Base Query: '{base['text']}'")
        
        try:
            # 1. Run the actual local Plan-and-Execute engine!
            result = await runner.run(base['text'])
            
            # 2. Extract the actual plan generated
            actual_plan_str = str(result.plan) 
            print(f"   ↳ Real Plan Captured: {actual_plan_str[:100]}...")

            # 3. Pass the REAL plan to WatsonX for Reverse Engineering
            prompt = SYSTEM_PROMPT.format(
                original_query=base['text'],
                actual_plan=actual_plan_str
            )

            print("   ↳ Reverse-engineering new scenarios based on this plan...")
            response = litellm.completion(
                model="watsonx/meta-llama/llama-3-3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, # High temperature ensures diverse phrasing
                response_format={"type": "json_object"}
            )
            
            # 4. Parse and format the output
            llm_output = json.loads(response.choices[0].message.content)
            
            for gen in llm_output["new_scenarios"]:
                scenario = {
                    "id": starting_id,
                    "type": "IoT",
                    "text": gen["text"],
                    "category": base["category"],
                    "deterministic": True,
                    "characteristic_form": gen["characteristic_form"],
                    "group": "retrospective",
                    "entity": "Chiller",
                    "note": f"Source: IoT data operations; Deterministic query with single correct answer; Category: {base['category']}"
                }
                all_new_scenarios.append(scenario)
                print(f"      ✅ Generated: {gen['text']}")
                starting_id += 1
                
        except Exception as e:
            print(f"   ❌ Execution failed: {e}")

    # 5. Save to CSV
    df = pd.DataFrame(all_new_scenarios)
    df.to_csv("new_scenarios.csv", index=False)
    print(f"\n Successfully saved {len(all_new_scenarios)} realistic, locally-testable scenarios to CSV!")

# Execute the async function
if __name__ == "__main__":
    asyncio.run(generate_from_real_trace())