#!/usr/bin/env python3
"""
MutEval: Reproducible Mutation-Based Robustness Evaluation Tool

Modes:
  1️⃣  --mutate-only : Generate only mutated prompts (no LLM, no tests)
  2️⃣  --mutate-llm  : Generate mutated prompts + run LLM + analysis

Usage Examples:
  python muteval.py --input humaneval.csv --output results.csv --mutate-llm --model gpt-4-turbo
  python muteval.py --input prompts.csv --output mutated.csv --mutate-only --mutations 5
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

def load_tasks(input_csv):
    df = pd.read_csv(input_csv)
    if "prompt" not in df.columns:
        raise ValueError("Input CSV must contain a 'prompt' column.")
    return df.to_dict(orient="records")

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

def run_python_tests(code_text, entry_point, test_code):
    tmpdir = tempfile.mkdtemp(prefix="muteval_")
    try:
        (Path(tmpdir) / "solution.py").write_text(code_text)
        indented_test = textwrap.indent(test_code, "    ")
        harness = f"""
import sys, importlib.util
try:
    spec = importlib.util.spec_from_file_location('solution', r"{Path(tmpdir)/'solution.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, '{entry_point}'):
        {entry_point} = getattr(mod, '{entry_point}')
{indented_test}
    print('<<RESULT>>PASS')
except Exception:
    print('<<RESULT>>FAIL')
    sys.exit(1)
"""
        (Path(tmpdir)/"run_tests.py").write_text(harness)
        proc = subprocess.run([sys.executable, str(Path(tmpdir)/"run_tests.py")],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
        out = proc.stdout.decode()
        return "<<RESULT>>PASS" in out, out.replace("\n", "\\n")
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# =====================================================
# Core logic
# =====================================================

def process_prompt(task, seeds, mode, code_agent=None):
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
        parrot_mut = mutate_with_parrot(prompt, seed)
        rad_mut = mutate_with_radamsa(parrot_mut, seed)

        # --- MODE 1: MUTATE ONLY ---
        if mode == "mutate-only":
            rows.append({
                "task_id": task_id,
                "seed": seed,
                "original_prompt": prompt,
                "parrot_mutated_prompt": parrot_mut,
                "radamsa_mutated_prompt": rad_mut
            })
            continue

        # --- MODE 2: MUTATE + LLM ---
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        base_codes, mut_codes = [], []
        for _ in range(10):  # N_SAMPLES
            base = loop.run_until_complete(generate_code(prompt, code_agent))
            mut = loop.run_until_complete(generate_code(rad_mut, code_agent))
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

        rows.append({
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
        })
    return rows

# =====================================================
# Main CLI
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="MutEval: Robustness Evaluation Tool")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, help="Fixed seed for reproducibility")
    parser.add_argument("--mutations", type=int, help="Number of mutations to generate")
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--mutate-only", action="store_true", help="Run only mutation pipeline")
    parser.add_argument("--mutate-llm", action="store_true", help="Run mutation + LLM + analysis")
    args = parser.parse_args()

    # Mode validation
    if not args.mutate_only and not args.mutate_llm:
        parser.error("Specify one mode: --mutate-only or --mutate-llm")

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
    if args.mutate_llm:
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

    mode = "mutate-only" if args.mutate_only else "mutate-llm"
    all_rows = []
    for task in tqdm(tasks, desc=f"Running MutEval ({mode})"):
        all_rows.extend(process_prompt(task, seeds, mode, code_agent))

    # Write results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.output, index=False)
    print(f"✅ Results saved to {args.output}")

    # Post-analysis if LLM mode
    if args.mutate_llm:
        print("📊 For post analysis please use analysis.ipynb")

if __name__ == "__main__":
    main()

