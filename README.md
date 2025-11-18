# Preparation

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Install Radamsa (for mutation fuzzing)
https://gitlab.com/akihe/radamsa

## 3. Set Your OpenAI API Key  
(Required for --mutate-llm and --mutate-code)
```bash
echo "OPENAI_API_KEY=sk-yourkeyhere" > .env
```

---

# Feature-by-Feature Test Commands

## 1. Basic Mutation-Only Mode
Checks the Parrot + Radamsa mutation pipeline and CSV output (no LLM usage).
```bash
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/trial_wo_tests.csv \
  --mutate-only
```

**Expected behavior:**
- Reads `task_id,prompt`
- Generates 1 mutation per prompt (default: seed=42)
- Outputs:  
  `task_id, seed, original_prompt, parrot_mutated_prompt, radamsa_mutated_prompt`
- Does not call any model or run tests

---

## 2. Multiple Mutations per Prompt
```bash
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/mutated_5.csv \
  --mutate-only \
  --mutations 5
```

**Expected behavior:**
- Generates 5 mutations per prompt
- Output contains 5× rows per input line

---

## 3. Fixed Seed Reproducibility
```bash
python main.py \
  --input data/trial_wo_tests.csv \
  --output results/mutated_seed_123.csv \
  --mutate-only \
  --seed 123
```

**Expected behavior:**
- Results remain identical across repeated runs

---

## 4. Mutation + LLM Evaluation Mode
Runs the full HumanEval-style pipeline (mutation → code generation → tests → metrics).
```bash
python main.py \
  --input data/trial.csv \
  --output results/llm_eval_results.csv \
  --mutate-llm \
  --model gpt-4-turbo \
  --mutations 3
```

**Expected behavior:**
- 3 mutations per task
- Code generated for original + mutated prompts
- Tests executed (pass/fail)
- Computes similarity + CodeBERTScore
- Output contains fields such as:  
  `task_id, seed, model, parrot_mutated_prompt, radamsa_mutated_prompt, similarity, codebertscore, pass@1_base, pass@3_mut, ...`

Automatically triggers:
```bash
python analysis.py --input llm_eval_results.csv
```

---

## 5. Model Selection Test
```bash
python main.py \
  --input data/trial.csv \
  --output results/llm_gpt35.csv \
  --mutate-llm \
  --model gpt-3.5-turbo-0125 \
  --mutations 2
```

**Expected behavior:**
- Uses specified model
- Model name appears in output CSV

---

## 6. Post-Analysis Trigger Test
```bash
python main.py \
  --input data/trial.csv \
  --output results/llm_analysis_check.csv \
  --mutate-llm
```

**Expected behavior:**
Terminal output should show:
```
Running post-analysis...
Results saved to llm_analysis_check.csv
```

Then executes:
```bash
python analysis.py --input llm_analysis_check.csv
```

---

## 7. Error Handling Tests

### Missing Columns
```bash
python main.py --input data/bad.csv --output out.csv --mutate-only
```
Expected:
```
ValueError: Input CSV must contain a 'prompt' column.
```

### Missing API Key
```bash
python main.py --input data/trial.csv --output out.csv --mutate-llm
```
Expected:
```
No OpenAI API key found. Set OPENAI_API_KEY in .env.
```

---

## 8. Code Mutation Mode
Mutates the canonical code solution and runs evaluation.

**Input CSV must contain:** `canonical_solution`

```bash
python main.py \
  --input data/trial.csv \
  --output results/results_code_gpt35.csv \
  --mutate-code \
  --model gpt-3.5-turbo-0125
```

**Output columns include:**  
`task_id, seed, model, original_prompt, mutated_prompt, base_codes, mut_codes, base_passed, mutated_passed, similarity, codebertscore, pass@1_base, pass@1_mut, ...`

---

## 9. Help / Usage
```bash
python main.py -h
```

Shows all supported options:
- --input  
- --output  
- --seed  
- --mutations  
- --model  
- --mutate-only  
- --mutate-llm  
- --mutate-code
