import pandas as pd
import subprocess

mutation_options = ['bd', 'bf', 'bi', 'br', 'bp', 'bei', 'bed', 'ber', 'sr', 'sd', 'lds', 'lr2', 'li', 'ls', 'lp', 'lis', 'lrs', 'td', 'tr2', 'ts1', 'ts2', 'tr', 'uw', 'num', 'ft', 'fn', 'fo']

def generate_mutations(prompt, prompt_id, original_prompt):
    mutations = []
    for mutation_option in mutation_options:
        print(f"Prompt ID: {prompt_id} | Mutation: {mutation_option}")
        # Old (Windows)
        # mutation = subprocess.run(['./radamsa.exe', '-p', 'od', '-m', mutation_option], input=prompt, capture_output=True, text=True, encoding='latin-1')

        # New (macOS/Homebrew)
        mutation = subprocess.run(
            ['radamsa', '-p', 'od', '-m', mutation_option],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='latin-1'
        )
        if mutation.returncode == 0 and mutation.stdout is not None:
            mutations.append((prompt_id, mutation_option, original_prompt, mutation.stdout.strip()))
        else:
            print(f"Error generating mutation for Prompt ID: {prompt_id} | Option: {mutation_option}")
    return mutations

# Load only the first 10 prompts
base_prompts = pd.read_csv('LLMSecEval-Prompts_dataset.csv').head(10)

mutated_prompts = []
for index, row in base_prompts.iterrows():
    prompt_id = row['Prompt ID']
    original_prompt = str(row['LLM-generated NL Prompt'])
    prompt_text = str(row['Manually-fixed NL Prompt']).replace('<language>', str(row['Language']))
    mutated_prompts.extend(generate_mutations(prompt_text, prompt_id, original_prompt))

# Include original prompt in the CSV
mutated_prompts_df = pd.DataFrame(
    mutated_prompts,
    columns=['ID', 'Mutation Option', 'Original Prompt', 'Mutated Prompt']
)
mutated_prompts_df.to_csv('mutated_prompts.csv', index=False)

print("Finished generating mutations for 10 prompts (with original prompts included)!")