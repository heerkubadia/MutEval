#!/usr/bin/env python3
"""
MutEval: Reproducible Mutation-Based Robustness Evaluation Tool

Modes:
  1️  --mutate-only : Generate only mutated prompts (no LLM, no tests)
  2  --mutate-llm  : Generate mutated prompts + run LLM + analysis
  3  --mutate-code : Generate mutated incomplete code + run LLM 

Usage Examples:
  python main.py --input humaneval.csv --output results.csv --mutate-llm --model gpt-4-turbo
  python main.py --input humaneval.csv --output mutated.csv --mutate-only --mutations 5
  python main.py --input humaneval.csv --output mutated.csv --mutate-code --mutations 10
  
"""

import os
import csv
import sys
import math
import re
import difflib
import random
import shutil
import tempfile
import subprocess
import asyncio
import textwrap
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import ast
import astor
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# =====================================================
# Setup
# =====================================================

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

try:
    from parrot import Parrot
except ImportError:
    raise ImportError("❌ Please install Parrot: pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git")

# Optional import for LLM + metrics
try:
    from agents import Agent, Runner, ModelSettings
except Exception:
    OPENAI_AVAILABLE = False

try:
    import code_bert_score
    CODEBERT_AVAILABLE = True
except Exception:
    CODEBERT_AVAILABLE = False

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

ALL_MUTATIONS = [
    "bd","bf","bi","br","bp","bei","bed","ber",
    "sr","sd","lr2","li","ls","lp","lis","lrs",
    "td","tr2","ts1","ts2","tr","uw","num","ft","fn","fo"
]

# =====================================================
# Helper functions
# =====================================================



def wrap_function_from_prompt(prompt: str, code: str) -> str:
    """
    Combine prompt function header with unindented code body.
    Returns a valid Python function definition string.
    """
    header = None
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("def ") and line.endswith(":"):
            header = line
            break
        if line.startswith("def ") and ":" in line:
            header = line[:line.index(":")+1]
            break
    if not header:
        return code  # fallback if no header found
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("def __mut_tmp"):
        lines = lines[1:] 
    total = len(lines)
    if total < 3:
        return code  # not enough lines to cut
    
    # choose random N and M ensuring N + M < total
    max_keep = int(total * 0.7)
    N = random.randint(1, max(1, max_keep // 2))
    M = random.randint(1, max(1, max_keep - N))
    
    if N + M >= total:
        M = max(0, total - N - 1)
    
    first_part = lines[:N]
    last_part = lines[-M:] if M > 0 else []
    
    # combine partials back
    
    snippet = '\n'.join(first_part + ["# ... (incomplete code) ..."] + last_part)
    body = textwrap.indent(textwrap.dedent(snippet), "    ")
    return f"{header}\n{body}"

def load_tasks(input_csv):
    df = pd.read_csv(input_csv)
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column.")
    return df.to_dict(orient="records")

def chunk_text(text, max_len=300):
    lines = text.split("\n")
    chunks = []
    current = []
    size = 0

    for line in lines:
        size += len(line)
        if size > max_len:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks

def mutate_with_parrot_long(text, seed):
    chunks = chunk_text(text)
    mutated_chunks = [mutate_with_parrot(chunk, seed) for chunk in chunks]
    return "\n".join(mutated_chunks)

def mutate_with_parrot(text, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        paraphrases = parrot.augment(
            input_phrase=text,
            use_gpu=torch.cuda.is_available(),
            do_diverse=False,
            max_return_phrases=1,
            adequacy_threshold=0.75,
            fluency_threshold=0.75
        )
        if not paraphrases:
            return text
        return sorted(paraphrases, key=lambda x: x[1], reverse=True)[0][0]
    except Exception:
        return text

def mutate_with_radamsa(text, seed):
    chosen_ops = random.sample(ALL_MUTATIONS, 2)
    cmd = ["radamsa", "-s", str(seed), "-m", ",".join(chosen_ops)]
    try:
        proc = subprocess.run(cmd, input=text.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        return proc.stdout.decode("utf-8") if proc.returncode == 0 else text
    except FileNotFoundError:
        raise RuntimeError("❌ Radamsa not found. Please install it (sudo apt install radamsa).")

def similarity_score(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def compute_codebertscore(pred, ref):
    if not CODEBERT_AVAILABLE:
        return -1.0
    try:
        _, _, f1, _ = code_bert_score.score(cands=[pred], refs=[ref], lang="python")
        return float(f1[0])
    except Exception:
        return -1.0

async def generate_code(prompt_text, agent):
    prompt_text = "Provide Python code with all imports, no Markdown." + prompt_text
    MAX_CHARS = 45000  # safe for 16k context (~12k tokens)

    if len(prompt_text) > MAX_CHARS:
        print(f"[WARN] prompt_text too long ({len(prompt_text)} chars). Truncating...")
        prompt_text = prompt_text[:MAX_CHARS]
    result = await Runner.run(agent, prompt_text)
    return result.final_output.strip()

async def generate_code_sys_prompt(prompt_text, agent):
    prompt_text = "You are an expert Python coder. You will be given a function signature and an incomplete or partially incorrect body. Understand the function from its name and complete the body logically, keeping the structure and intent consistent. Return Python code with all imports, no Markdown." + prompt_text


    result = await Runner.run(agent, prompt_text)
    return result.final_output.strip()

def clean_code(code):
    code = re.sub(r"```[a-zA-Z]*", "", code)
    return code.replace("```", "").strip()

def estimate_pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n == c:
        return 1.0
    k = min(k, n)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def run_python_tests(code_text: str, entry_point: str, test_code: str, timeout_s: int = 5):
    td = tempfile.mkdtemp(prefix="humaneval_")
    try:
        sol_path = Path(td) / "solution.py"
        sol_path.write_text(code_text, encoding="utf-8", errors="ignore")

        harness = f"""
import sys, traceback, importlib.util

try:
    # Load module dynamically
    spec = importlib.util.spec_from_file_location('solution', r"{sol_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Ensure entry point exists
    if not hasattr(mod, "{entry_point}"):
        print("Missing entry point: {entry_point}")
        print("<<HUMANEVAL_TEST_RESULT>>: FAIL")
        sys.exit(1)

    _entry = getattr(mod, "{entry_point}")

    # Insert test code
{textwrap.indent(test_code, "    ")}

    # Require check() to be defined AND executed
    if 'check' in locals():
        check(_entry)
    else:
        print("No check() function defined in test code")
        print("<<HUMANEVAL_TEST_RESULT>>: FAIL")
        sys.exit(1)

    print("<<HUMANEVAL_TEST_RESULT>>: PASS")

except Exception:
    traceback.print_exc()
    print("<<HUMANEVAL_TEST_RESULT>>: FAIL")
    sys.exit(1)
"""

        harness_path = Path(td) / "run_tests.py"
        harness_path.write_text(harness, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(harness_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s
        )

        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        passed = "<<HUMANEVAL_TEST_RESULT>>: PASS" in out

        return passed, out.replace("\n", "\\n")

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"

    finally:
        shutil.rmtree(td, ignore_errors=True)

def append_mutation_codes_csv(path, task_id, model, base_codes, mut_codes):
    import csv, json, os

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow(["task_id", "model", "base_codes", "mut_codes"])

        writer.writerow([
            task_id,
            model,
            json.dumps(base_codes),   # serialize list to string
            json.dumps(mut_codes)
        ])
        f.flush()
        os.fsync(f.fileno())
def append_row_to_csv(path, row_dict):
    import csv, os

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_dict)
        f.flush()    # ensure data is physically written

def add_entrypoint_constraint(prompt: str, entry_point: str) -> str:
    rule = f"""

Make sure you use exactly this function name: {entry_point} only as the test depends on this name.

"""
    return rule + "\n" + prompt

# =====================================================
# Semantic Code Mutator
# =====================================================

class SemanticMutator(ast.NodeTransformer):
    """Semantically-aware AST mutator."""
    def __init__(self, temperature=0.5):
        self.temperature = temperature
        self.available_mutations = [
            "flip_compare", "swap_binop", "tweak_constant",
            "rename_var_2", "rename_var", "negate_condition"
        ]
        self.active_mutations = self.available_mutations

    def visit_Compare(self, node):
        if "flip_compare" in self.active_mutations and random.random() < self.temperature:
            flip = {ast.Gt: ast.Lt, ast.Lt: ast.Gt, ast.GtE: ast.LtE,
                    ast.LtE: ast.GtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
            if type(node.ops[0]) in flip:
                node.ops[0] = flip[type(node.ops[0])]()
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if "swap_binop" in self.active_mutations and random.random() < self.temperature:
            swap = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
            if type(node.op) in swap:
                node.op = swap[type(node.op)]()
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if "tweak_constant" in self.active_mutations and isinstance(node.value, (int, float)):
            if random.random() < self.temperature:
                node.value += random.choice([-2, -1, 1, 2])
        return node

    def visit_Name(self, node):
        if node.id in ("True", "False", "None"):
            return node
        if "rename_var" in self.active_mutations and random.random() < self.temperature * 0.3:
            node.id = node.id + random.choice(["_alt", "_temp", "_x"])
        elif "rename_var_2" in self.active_mutations and random.random() < self.temperature * 0.2:
            node.id = node.id[::-1]
        return node

    def visit_If(self, node):
        if "negate_condition" in self.active_mutations and random.random() < self.temperature * 0.5:
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        return self.generic_visit(node)


def mutate_semantic(code: str, temperature=0.5, num_mutants=1):
    """Generate semantically mutated Python variants using AST rewriting."""
    mutants = []
    for _ in range(num_mutants):
        try:
            tree = ast.parse(code)
            mutator = SemanticMutator(temperature)
            mutated_tree = mutator.visit(tree)
            ast.fix_missing_locations(mutated_tree)
            mutated_code = astor.to_source(mutated_tree)
            mutants.append(mutated_code)
        except Exception as e:
            print("Mutation failed:", type(e).__name__, e)
            continue
    return mutants

# =====================================================
# Core logic
# =====================================================

def process_prompt(task, seeds, mode, output_path,code_agent=None):
    """
    For each task and each seed:
      - Mutate the prompt deterministically (Parrot + Radamsa)
      - If mode == mutate-only → store mutated prompts
      - If mode == mutate-llm → run LLM generation + testing + metrics
    """
    prompt = task["prompt"]
    task_id = task.get("task_id", "")
    entry_point = task.get("entry_point", "")
    test_code = task.get("test", "")
    rows = []

    for seed in seeds:
        # Apply both deterministic mutations
        parrot_mut = mutate_with_parrot_long(prompt, seed)
        rad_mut = mutate_with_radamsa(parrot_mut, seed)

        # --- MODE 1: MUTATE ONLY ---
        if mode == "mutate-only":
            row = {
                "task_id": task_id,
                "seed": seed,
                "original_prompt": prompt,
                "parrot_mutated_prompt": parrot_mut,
                "radamsa_mutated_prompt": rad_mut
            }
            append_row_to_csv(output_path, row)
            continue

        # --- MODE 2: MUTATE + LLM ---
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        base_codes, mut_codes = [], []
        for _ in range(10):  # N_SAMPLES
            base = loop.run_until_complete(generate_code(prompt, code_agent))
            p_mut_prompt = add_entrypoint_constraint(rad_mut, entry_point)
            mut = loop.run_until_complete(generate_code(p_mut_prompt, code_agent))

            base_codes.append(clean_code(base))
            mut_codes.append(clean_code(mut))
        loop.close()

        base_pass = [run_python_tests(c, entry_point, test_code)[0] for c in base_codes]
        mut_pass = [run_python_tests(c, entry_point, test_code)[0] for c in mut_codes]
        n_b, n_m, c_b, c_m = len(base_pass), len(mut_pass), sum(base_pass), sum(mut_pass)
        

        metrics = {
            f"pass@{k}_base": estimate_pass_at_k(n_b, c_b, k)
            for k in [1, 3, 5]
        }
        metrics.update({
            f"pass@{k}_mut": estimate_pass_at_k(n_m, c_m, k)
            for k in [1, 3, 5]
        })

        sim = similarity_score("\n".join(base_codes), "\n".join(mut_codes))
        cb = compute_codebertscore("\n".join(base_codes), "\n".join(mut_codes))
        # Save base + mutated codes for this task/seed
        append_mutation_codes_csv(
            path="results2/generated_codes.csv",
            task_id=task_id,
            model=code_agent.model if code_agent else "",
            base_codes=base_codes,
            mut_codes=mut_codes
        )

        row = {
            "task_id": task_id,
            "seed": seed,
            "model": code_agent.model,
            "original_prompt": prompt,
            "parrot_mutated_prompt": parrot_mut,
            "radamsa_mutated_prompt": rad_mut,
            "base_passed": c_b,
            "mutated_passed": c_m,
            "n_base_samples": n_b,
            "n_mut_samples": n_m,
            "similarity": sim,
            "codebertscore": cb,
            **metrics
        }
        append_row_to_csv(output_path, row)
    return row

def process_code_mutation(task, seeds, mode, output_path,code_agent=None):
    """
    For each task and each seed:
      - Mutate the canonical solution semantically
      - If mode == mutate-code → run tests + metrics
    """
    code = task["canonical_solution"]
    prompt = task.get("prompt", "")
    task_id = task.get("task_id", "")
    entry_point = task.get("entry_point", "")
    test_code = task.get("test", "")
    
    all_rows = []

    for seed in seeds:
        random.seed(seed)
        temperature = round(random.uniform(0.5, 1.0), 2)
        mutants_body = mutate_semantic("def __mut_tmp():\n" + code, temperature=temperature, num_mutants=1)[0]
        mutant = wrap_function_from_prompt(prompt,mutants_body)
        

        if not mutant:
            continue

        # Run tests for each mutant
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        base_codes, mut_codes = [], []
        for _ in range(10):  # N_SAMPLES
            base = loop.run_until_complete(generate_code_sys_prompt(prompt, code_agent))

            p_mut_prompt = add_entrypoint_constraint(mutant, entry_point)
            mut = loop.run_until_complete(generate_code(p_mut_prompt, code_agent))

            base_codes.append(clean_code(base))
            mut_codes.append(clean_code(mut))
        loop.close()

        base_pass = [run_python_tests(c, entry_point, test_code)[0] for c in base_codes]
        mut_pass = [run_python_tests(c, entry_point, test_code)[0] for c in mut_codes]
        n_b, n_m, c_b, c_m = len(base_pass), len(mut_pass), sum(base_pass), sum(mut_pass)

        metrics = {
            f"pass@{k}_base": estimate_pass_at_k(n_b, c_b, k)
            for k in [1, 3, 5]
        }
        metrics.update({
            f"pass@{k}_mut": estimate_pass_at_k(n_m, c_m, k)
            for k in [1, 3, 5]
        })

        sim = similarity_score("\n".join(base_codes), "\n".join(mut_codes))
        cb = compute_codebertscore("\n".join(base_codes), "\n".join(mut_codes))
        append_mutation_codes_csv(
        path="results2/generated_codes.csv",
        task_id=task_id,
        model=code_agent.model if code_agent else "",
        base_codes=base_codes,
        mut_codes=mut_codes
    )

        row ={
            "task_id": task_id,
            "seed": seed,
            "model": code_agent.model,
            "original_prompt": prompt,
            "mutated_prompt": mutant,
            "base_codes": base_codes,
            "mut_codes": mut_codes,
            "base_passed": c_b,
            "mutated_passed": c_m,
            "n_base_samples": n_b,
            "n_mut_samples": n_m,
            "similarity": sim,
            "codebertscore": cb,
            **metrics
        }
        all_rows.append(row)
        append_row_to_csv(output_path, row)
    return all_rows

# =====================================================
# Main CLI
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="MutEval: Robustness Evaluation Tool")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, help="Fixed seed for reproducibility")
    parser.add_argument("--mutations", type=int, help="Number of mutations to generate")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--mutate-only", action="store_true", help="Run only mutation pipeline")
    parser.add_argument("--mutate-llm", action="store_true", help="Run mutation + LLM + analysis")
    parser.add_argument("--mutate-code", action="store_true", help="Mutate canonical code instead of prompt")

    args = parser.parse_args()
    Path("results2").mkdir(parents=True, exist_ok=True)

    # Mode validation
    if sum([args.mutate_only, args.mutate_llm, args.mutate_code]) != 1:
        parser.error("Specify exactly one mode: --mutate-only, --mutate-llm, or --mutate-code")


    # Seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.mutations:
        seeds = random.sample(range(1, 9999), args.mutations)
    else:
        seeds = [42]  # default seed
    print(f"🧬 Seeds: {seeds}")

    # Load tasks
    tasks = load_tasks(args.input)
    print(f"📄 Loaded {len(tasks)} tasks.")

    # Model setup if LLM mode
    code_agent = None
    if args.mutate_llm or args.mutate_code:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("❌ No OpenAI API key found. Set OPENAI_API_KEY in .env.")
        model_settings = ModelSettings(temperature=0.0, top_p=1.0)
        code_agent = Agent(
            name="CodeGen",
            instructions="Generate Python code implementing the given prompt.",
            model=args.model,
            model_settings=model_settings
        )
        print(f"🤖 Model: {args.model}")

    if args.mutate_only:
        mode = "mutate-only"
    elif args.mutate_llm:
        mode = "mutate-llm"
    elif args.mutate_code:
        mode = "mutate-code"

    all_rows = []
    for task in tqdm(tasks, desc=f"Running MutEval ({mode})"):
        if mode == "mutate-code":
            all_rows.extend(process_code_mutation(task, seeds, mode, args.output,code_agent))
        else:
            all_rows.extend(process_prompt(task, seeds, mode, args.output,code_agent))

    # Post-analysis if LLM mode
    if args.mutate_llm:
        print("📊 For post analysis please use analysis.ipynb")

if __name__ == "__main__":
    main()
