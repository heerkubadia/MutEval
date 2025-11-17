🧰 Preparation
Before running anything:
# 1. Install dependencies
pip install -r requirements.txt

# 2.Install radamsa for mutation fuzzing
https://gitlab.com/akihe/radamsa

# 3. Make sure OPENAI_API_KEY is set if using mutate-llm or mutate-code mode
echo "OPENAI_API_KEY=sk-yourkeyhere" > .env


🧩 Feature-by-Feature Test Commands
✅ 1. Basic Mutation-Only Mode
Checks the Parrot + Radamsa pipeline and CSV output without any LLM involvement.
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/trial_wo_tests.csv \
  --mutate-only

Expected behavior:
Reads data/trial_wo_tests.csv containing task_id,prompt
Generates 1 mutation per task (default seed=42)
Saves results/trial_wo_tests.csv with columns:
task_id, seed, original_prompt, parrot_mutated_prompt, radamsa_mutated_prompt


Does not call any model or run analysis

✅ 2. Multiple Mutations per Prompt
Checks that multiple random seeds produce multiple rows per prompt.
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/mutated_5.csv \
  --mutate-only \
  --mutations 5

Expected behavior:
For each prompt, generates 5 distinct mutations
Output should have 5× rows per input line

✅ 3. Fixed Seed Reproducibility
Ensures deterministic results with a given seed.
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/mutated_seed_123.csv \
  --mutate-only \
  --seed 123

Expected behavior:
Always produces the same mutations across runs
(Verifiable by diff mutated_seed_123.csv across multiple runs)

✅ 4. Mutation + LLM Evaluation
Runs the full HumanEval-like pipeline with code generation, tests, and metrics.
python main.py \
  --input data/trial.csv \
  --output results/llm_eval_results.csv \
  --mutate-llm \
  --model gpt-4-turbo \
  --mutations 3

Expected behavior:
Each task → 3 mutations
Generates code for original + mutated prompts
Runs correctness tests (pass/fail)
Computes similarity and CodeBERTScore
Outputs columns like:
task_id, seed, model, parrot_mutated_prompt, radamsa_mutated_prompt,
similarity, codebertscore, pass@1_base, pass@3_mut, ...


Automatically runs:
python analysis.py --input llm_eval_results.csv
after CSV is saved.

✅ 5. Model Selection Test
Checks that a different LLM model can be specified.
python main.py \
  --input data/trial.csv \
  --output results/llm_gpt35.csv \
  --mutate-llm \
  --model gpt-3.5-turbo-0125 \
  --mutations 2

Expected behavior:
Uses GPT-3.5 model instead of GPT-4
Model name reflected in model column in output

✅ 6. Post-Analysis Trigger Test
If you want to confirm the analysis step runs automatically:
python main.py \
  --input data/trial.csv \
  --output results/llm_analysis_check.csv \
  --mutate-llm

Then check your terminal output — you should see:
📊 Running post-analysis...
✅ Results saved to llm_analysis_check.csv

and the script will execute:
python analysis.py --input llm_analysis_check.csv


✅ 7. Error Handling Tests
Missing Columns
python main.py --input data/bad.csv --output out.csv --mutate-only

→ should raise:
ValueError: Input CSV must contain a 'prompt' column.
Missing API Key
python main.py --input data/trial.csv --output out.csv --mutate-llm

(without .env key)
→ should raise:
❌ No OpenAI API key found. Set OPENAI_API_KEY in .env.

✅ 8. Code Mutation Mode 
Performs semantic mutations on the canonical code solution. Runs the full HumanEval-like pipeline with code generation, tests, and metrics.
Input CSV must contain a canonical_solution column
For each task, the canonical solution is mutated using code-level transformations and given to llm as code context. 
Produces output with columns:
task_id, seed, model, original_prompt, mutated_prompt, base_codes, mut_codes, base_passed, mutated_passed, similarity, codebertscore, pass@1_base, pass@1_mut, etc.

python main.py \
  --input data/humaneval.csv \
  --output results2/results_code_gpt-4o-2024-05-13.csv \
  --mutate-code
(optionally model if --model is used)

✅ 9. Help / Usage Message
python main.py -h

Should show all supported options:
--input, --output, --seed, --mutations, --model,
--mutate-only, --mutate-llm, --mutate-code

