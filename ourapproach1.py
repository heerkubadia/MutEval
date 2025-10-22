#!/usr/bin/env python3
"""
humaneval_pipeline.py

Pipeline:
- Load HumanEval tasks (JSON/CSV)
- Mutate prompts using radamsa, nlpaug, textattack, and an LLM mutator (uses your Agent/Runner)
- Generate code with your Code Generator Agent (via Runner.run)
- Execute generated code against tests (sandboxed subprocess)
- Compute similarity score (CodeBERTScore if installed, else difflib ratio)
- Save results CSV

USAGE:
    python humaneval_pipeline.py --humaneval humaneval.json --out results.csv --n_per_prompt 3

REQUIREMENTS (install as needed):
    pip install pandas nlpaug textattack tqdm codebert_score (optional)
    radamsa must be available in PATH
    Your project's agents module must be importable (Agent, Runner, ModelSettings).
"""
import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import random
import difflib

# ----- Optional dependencies -----
try:
    import nlpaug.augmenter.word as naw
except Exception:
    naw = None

try:
    from textattack.augmentation import EmbeddingAugmenter
except Exception:
    EmbeddingAugmenter = None

# CodeBERTScore optional
_CODEBERT_AVAILABLE = False
try:
    # package name may vary; trying common import
    from codebert_score import codebert_score
    _CODEBERT_AVAILABLE = True
except Exception:
    try:
        # alternative package name
        from codebert_score import score as codebert_score
        _CODEBERT_AVAILABLE = True
    except Exception:
        _CODEBERT_AVAILABLE = False

# ----- User's Agents / Runner -----
# This expects your earlier-provided agents module to be on PYTHONPATH
try:
    from agents import Agent, Runner, ModelSettings
except Exception as e:
    print("ERROR: Could not import your agents Runner/Agent/ModelSettings. Ensure your project is on PYTHONPATH.")
    raise

# ----- Setup default Agents (reuse your Code Generator) -----
model_settings = ModelSettings(temperature=0.0, top_p=1.0)
code_agent = Agent(
    name="Code Generator",
    instructions="Generate complete standalone code based on user instructions. Respond only with code without markdown.",
    model="gpt-3.5-turbo-0125",
    model_settings=model_settings
)
# Mutator agent: instruct LLM to return only mutated prompt (plain text)
mutator_agent = Agent(
    name="LLM Mutator",
    instructions=(
        "Mutate the input natural language prompt according to the user's instructions. "
        "Return exactly the mutated prompt text only (no commentary, no JSON)."
    ),
    model="gpt-3.5-turbo-0125",
    model_settings=ModelSettings(temperature=0.2, top_p=1.0)
)

# ----- Utilities -----
def load_humaneval_tasks(path: str) -> List[Dict[str, Any]]:
    """
    Loads human-eval tasks. Accepts:
      - JSON list where each item has keys: task_id, prompt, entry_point, canonical_solution, test
      - CSV with columns that match above.
    Returns list of dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".json",):
        data = json.loads(p.read_text())
        return data
    elif p.suffix.lower() in (".csv", ".tsv"):
        import pandas as pd
        df = pd.read_csv(p)
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "task_id": r.get("task_id") or r.get("Task ID") or r.get("id") or r.get("Prompt ID"),
                "prompt": r.get("prompt") or r.get("Prompt") or r.get("LLM-generated NL Prompt") or r.get("Manually-fixed NL Prompt"),
                "entry_point": r.get("entry_point") or r.get("entry_point") or r.get("entry_point"),
                "canonical_solution": r.get("canonical_solution") or r.get("Canonical solution") or r.get("Solution"),
                "test": r.get("test") or r.get("Test") or r.get("test_code")
            })
        return rows
    else:
        raise ValueError("Unsupported humaneval file format. Use JSON or CSV.")

# ----- Mutation functions -----
def mutate_with_radamsa(text: str, radamsa_args=None) -> str:
    """
    Mutate textual prompt with radamsa. radamsa works on bytes; use latin-1 to preserve bytes.
    Returns mutated string decoded as latin-1.
    """
    cmd = ['radamsa']
    if radamsa_args:
        cmd += radamsa_args
    try:
        proc = subprocess.run(cmd, input=text.encode('latin-1'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if proc.returncode != 0:
            raise RuntimeError("radamsa returned non-zero: " + proc.stderr.decode('latin-1', errors='ignore'))
        return proc.stdout.decode('latin-1', errors='ignore')
    except FileNotFoundError:
        raise RuntimeError("radamsa not found in PATH. Install radamsa or adjust your PATH.")

def mutate_with_nlpaug(text: str) -> str:
    """
    Contextual word embedding augmentation (nlpaug).
    """
    if naw is None:
        raise RuntimeError("nlpaug is not installed. pip install nlpaug")
    aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased')  # default; models will be downloaded if needed
    aug_texts = aug.augment(text, n=1)
    return aug_texts[0]

def mutate_with_textattack(text: str) -> str:
    if EmbeddingAugmenter is None:
        raise RuntimeError("textattack is not installed. pip install textattack")
    aug = EmbeddingAugmenter()
    res = aug.augment(text)
    return res[0] if res else text

async def mutate_with_llm(prompt_text: str, instructions: str = "") -> str:
    """
    Use your Mutator Agent via Runner.run to produce a mutated prompt.
    Returns mutated prompt string. Runner.run returns an object whose final_output contains the LLM response.
    """
    # Prepare a short instruction to the mutator agent:
    msg = f"Original prompt:\n{prompt_text}\n\nMutate it to be a semantically different but related variant useful for code generation. {instructions}\nReturn only the mutated prompt text."
    result = await Runner.run(mutator_agent, msg)
    return result.final_output.strip()

# ----- Code generation -----
async def generate_code_with_agent(prompt_text: str, n: int = 1) -> List[str]:
    """
    Using your Code Generator Agent -> Runner.run repeatedly to get 'n' outputs.
    Returns a list of code strings (raw text).
    """
    outputs = []
    for i in range(n):
        # Some LLMs are deterministic with temp=0; but we still call n times to get potentially different completions
        result = await Runner.run(code_agent, prompt_text)
        outputs.append(result.final_output)
    return outputs

# ----- Evaluation (run tests) -----
def run_python_tests(code_text: str, entry_point: str, test_code: str, timeout_s: int = 5) -> Tuple[bool, str]:
    """
    Executes generated code in an isolated temp dir and runs the test_code.
    - code_text: full source code that should define the function named entry_point
    - test_code: string containing assertions calling the entry_point (the test format from HumanEval)
    Returns (passed:bool, stdout_and_stderr:str)
    """
    # Create temp dir
    td = tempfile.mkdtemp(prefix="humaneval_")
    try:
        # write solution file
        sol_path = Path(td) / "solution.py"
        sol_path.write_text(code_text, encoding='utf-8', errors='ignore')

        # create a runner script that imports solution and runs test_code
        harness = f"""
import sys, traceback, importlib.util, time
sys.setrecursionlimit(10000)
try:
    # import solution module
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location('solution', 'solution.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals().update({'__solution__': mod})
    # place entry_point in local name for tests if needed
    if hasattr(mod, '{entry_point}'):
        {entry_point} = getattr(mod, '{entry_point}')
    # run test code
    {test_code}
    print('<<HUMANEVAL_TEST_RESULT>>: PASS')
except Exception as e:
    import traceback, sys
    traceback.print_exc()
    print('<<HUMANEVAL_TEST_RESULT>>: FAIL')
    sys.exit(1)
"""
        harness_path = Path(td) / "run_tests.py"
        harness_path.write_text(harness, encoding='utf-8')

        # run the harness in a subprocess with timeout
        proc = subprocess.run([sys.executable, str(harness_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
        out = proc.stdout.decode('utf-8', errors='ignore')
        passed = '<<HUMANEVAL_TEST_RESULT>>: PASS' in out
        return passed, out
    except subprocess.TimeoutExpired as e:
        return False, f"TIMEOUT after {timeout_s}s"
    finally:
        # cleanup
        try:
            shutil.rmtree(td)
        except Exception:
            pass

# ----- Scoring -----
def similarity_score(pred: str, ref: str) -> float:
    """
    Fallback similarity if CodeBERTScore not available:
    use difflib.SequenceMatcher ratio (0..1)
    """
    return difflib.SequenceMatcher(None, pred, ref).ratio()

def compute_codebertscore(preds: List[str], refs: List[str]) -> List[float]:
    """
    If codebert_score available, compute score per pair and return list of floats.
    Otherwise return -1 list.
    """
    if not _CODEBERT_AVAILABLE:
        return [-1.0] * len(preds)
    # The codebert_score API varies among versions; try to call reasonably
    try:
        # some versions: codebert_score(preds, refs, lang='python', device='cpu')
        scores = codebert_score(preds, refs, lang='python')
        # if returns a tuple - handle common patterns
        if isinstance(scores, tuple):
            # expecting (scores_tensor, precisions, recalls) or similar
            s = scores[0]
            # convert to list of floats
            return [float(x) for x in s]
        elif isinstance(scores, list):
            return [float(x) for x in scores]
        else:
            # unknown shape: fallback
            return [-1.0] * len(preds)
    except Exception:
        return [-1.0] * len(preds)

# ----- Main pipeline -----
def build_prompt_for_model(prompt_text: str) -> str:
    """
    Wrap the prompt to be submitted to the code generation model. You can
    add meta instructions (e.g., "Write a Python function named X...") as needed.
    HumanEval prompts often already contain the spec; we use it directly.
    """
    return prompt_text

def single_task_workflow(task: Dict[str, Any], idx: int, args) -> List[Dict[str, Any]]:
    """
    Perform full pipeline on ONE HumanEval task: mutate, generate, evaluate.
    Returns list of result rows (one per generated completion).
    This function is synchronous for use with ThreadPool; LLM calls are asynchronous so we run them via asyncio loop below when needed.
    """
    results = []
    task_id = task.get('task_id') or task.get('id') or f"task_{idx}"
    prompt = task.get('prompt') or ""
    entry_point = task.get('entry_point') or task.get('function_name') or "f"
    canonical = task.get('canonical_solution') or ""
    test_code = task.get('test') or task.get('tests') or ""

    # Prepare mutations list
    mutation_variants = []

    # Original (no mutation)
    mutation_variants.append(("original", prompt))

    # Radamsa mutation
    try:
        rad = mutate_with_radamsa(prompt)
        mutation_variants.append(("radamsa", rad))
    except Exception as e:
        # radamsa failure -> skip
        mutation_variants.append(("radamsa_error", f"ERROR: {e}"))

    # nlpaug
    if naw is not None:
        try:
            nlp = mutate_with_nlpaug(prompt)
            mutation_variants.append(("nlpaug", nlp))
        except Exception as e:
            mutation_variants.append(("nlpaug_error", f"ERROR: {e}"))
    else:
        mutation_variants.append(("nlpaug_missing", prompt))

    # textattack
    if EmbeddingAugmenter is not None:
        try:
            ta = mutate_with_textattack(prompt)
            mutation_variants.append(("textattack", ta))
        except Exception as e:
            mutation_variants.append(("textattack_error", f"ERROR: {e}"))
    else:
        mutation_variants.append(("textattack_missing", prompt))

    # LLM mutator (synchronous - call Runner.run via asyncio inside)
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mutated_llm = loop.run_until_complete(mutate_with_llm(prompt, instructions="Make a moderately different prompt variant for code generation."))
        mutation_variants.append(("llm_mutator", mutated_llm))
    except Exception as e:
        mutation_variants.append(("llm_mutator_error", f"ERROR: {e}"))
    finally:
        try:
            loop.close()
        except Exception:
            pass

    # For each mutation variant, generate n_per_prompt completions
    for mut_name, mut_prompt in mutation_variants:
        # if earlier mutation produced an error record, still produce a row that indicates the error
        if mut_prompt.startswith("ERROR:"):
            results.append({
                "task_id": task_id,
                "mutation": mut_name,
                "generated_code": "",
                "passed": False,
                "test_output": mut_prompt,
                "similarity": -1.0,
                "codebertscore": -1.0,
                "canonical_solution": canonical
            })
            continue

        # Build final model prompt (can wrap instructions)
        model_prompt = build_prompt_for_model(mut_prompt)

        # Generate N variants using your code agent (synchronously via async runner)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            gen_list = loop.run_until_complete(generate_code_with_agent(model_prompt, n=args.n_per_prompt))
        except Exception as e:
            gen_list = [f"ERROR_CALLING_AGENT: {e}"]
        finally:
            try:
                loop.close()
            except Exception:
                pass

        for gen_code in gen_list:
            # run tests
            passed, out = run_python_tests(gen_code, entry_point, test_code, timeout_s=args.test_timeout)
            # similarity score vs canonical (if available)
            sim = similarity_score(gen_code, canonical) if canonical else -1.0
            # codebert
            cb = compute_codebertscore([gen_code], [canonical])[0] if canonical else -1.0

            results.append({
                "task_id": task_id,
                "mutation": mut_name,
                "generated_code": gen_code,
                "passed": bool(passed),
                "test_output": out.replace("\n", "\\n"),
                "similarity": float(sim),
                "codebertscore": float(cb),
                "canonical_solution": canonical
            })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval", required=True, help="Path to HumanEval JSON/CSV dataset")
    parser.add_argument("--out", default="humaneval_results.csv", help="CSV output path")
    parser.add_argument("--n_per_prompt", type=int, default=2, help="Number of completions to generate per mutated prompt")
    parser.add_argument("--parallel", type=int, default=2, help="Parallel worker threads (for tasks)")
    parser.add_argument("--test-timeout", type=int, default=5, help="Seconds before test run timeout")
    args = parser.parse_args()

    args.n_per_prompt = args.n_per_prompt
    args.test_timeout = args.test_timeout

    tasks = load_humaneval_tasks(args.humaneval)
    print(f"Loaded {len(tasks)} tasks.")

    # Thread pool to process tasks in parallel (note: LLM calls happen inside each worker via Runner.run)
    pool = ThreadPool(args.parallel)
    results = []

    # Use tqdm for progress
    work_iter = []
    for i, t in enumerate(tasks):
        work_iter.append((t, i, args))

    try:
        for res in tqdm(pool.imap_unordered(lambda wi: single_task_workflow(*wi), work_iter), total=len(work_iter)):
            results.extend(res)
    finally:
        pool.close()
        pool.join()

    # Write CSV
    fieldnames = ["task_id", "mutation", "passed", "similarity", "codebertscore", "test_output", "generated_code", "canonical_solution"]
    with open(args.out, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "task_id": r["task_id"],
                "mutation": r["mutation"],
                "passed": r["passed"],
                "similarity": r["similarity"],
                "codebertscore": r["codebertscore"],
                "test_output": r["test_output"],
                "generated_code": r["generated_code"],
                "canonical_solution": r.get("canonical_solution", "")
            })
    print(f"Wrote {len(results)} rows to {args.out}")

if __name__ == "__main__":
    main()
