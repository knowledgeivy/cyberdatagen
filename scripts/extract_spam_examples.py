#!/usr/bin/env python3
"""
Extract representative synthetic spam examples for paper
"""

import pandas as pd
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
PROJECT_ROOT = Path("/Users/tianyu/Notebooks/cyberdata")
GPT_DIR = PROJECT_ROOT / "data/full_experiments/ceas08_gpt41mini/synthetic"
CLAUDE_DIR = PROJECT_ROOT / "data/full_experiments/ceas08_claude35haiku/synthetic"

def load_sample(data_dir, prompt, model_name, max_body_length=300):
    """Load a representative sample from the specified prompt strategy"""
    # Load first group
    file_path = data_dir / f"ceas08_synthetic_{prompt}_{model_name}_group_0.csv.gz"

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return None

    df = pd.read_csv(file_path, compression='gzip')

    # Filter samples with reasonable length (not too short, not too long)
    df['body_length'] = df['synthetic_body'].str.len()
    df_filtered = df[(df['body_length'] > 100) & (df['body_length'] < max_body_length)]

    if len(df_filtered) == 0:
        df_filtered = df

    # Select a random sample
    sample = df_filtered.sample(n=1, random_state=42).iloc[0]

    return {
        'subject': sample['synthetic_subject'],
        'body': sample['synthetic_body'],
        'original_subject': sample.get('original_subject', 'N/A'),
        'original_body': sample.get('original_body', 'N/A')
    }


def format_latex_example(prompt, subject, body, max_body_chars=400):
    """Format a single example for LaTeX table"""
    # Escape LaTeX special characters
    def escape_latex(text):
        if pd.isna(text):
            return ""
        text = str(text)
        replacements = {
            '\\': '\\textbackslash{}',
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    subject_escaped = escape_latex(subject)
    body_escaped = escape_latex(body)

    # Truncate body if too long
    if len(body_escaped) > max_body_chars:
        body_escaped = body_escaped[:max_body_chars] + "..."

    return f"\\textbf{{{prompt}}} & \n\\textit{{Subject:}} {subject_escaped} \\\\\n& \\textit{{Body:}} {body_escaped} \\\\\n"


def create_latex_table(model_name, examples, label_suffix):
    """Create a complete LaTeX table"""
    latex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Representative Synthetic Spam Examples ({model_name})}}
\\label{{tab:spam_examples_{label_suffix}}}
\\small
\\begin{{tabular}}{{p{{0.12\\textwidth}}p{{0.83\\textwidth}}}}
\\toprule
\\textbf{{Strategy}} & \\textbf{{Generated Email}} \\\\
\\midrule
"""

    for prompt in ['original', 'strong', 'weak']:
        if prompt in examples:
            ex = examples[prompt]
            latex += format_latex_example(
                prompt.capitalize(),
                ex['subject'],
                ex['body']
            )
            if prompt != 'weak':  # Add midrule between examples
                latex += "\\midrule\n"

    latex += """\\bottomrule
\\end{tabular}
\\end{table*}
"""
    return latex


def main():
    print("=" * 80)
    print("Extracting Representative Synthetic Spam Examples")
    print("=" * 80)

    # Extract GPT examples
    print("\n--- GPT-4.1-mini Examples ---")
    gpt_examples = {}
    for prompt in ['original', 'strong', 'weak']:
        sample = load_sample(GPT_DIR, prompt, "gpt-4.1-mini")
        if sample:
            gpt_examples[prompt] = sample
            print(f"\n{prompt.upper()}:")
            print(f"Subject: {sample['subject'][:80]}...")
            print(f"Body length: {len(sample['body'])} chars")

    # Extract Claude examples
    print("\n\n--- Claude-3.5-Haiku Examples ---")
    claude_examples = {}
    for prompt in ['original', 'strong', 'weak']:
        sample = load_sample(CLAUDE_DIR, prompt, "claude-3-5-haiku")
        if sample:
            claude_examples[prompt] = sample
            print(f"\n{prompt.upper()}:")
            print(f"Subject: {sample['subject'][:80]}...")
            print(f"Body length: {len(sample['body'])} chars")

    # Generate LaTeX tables
    print("\n\n" + "=" * 80)
    print("Generated LaTeX Tables")
    print("=" * 80)

    print("\n--- Table for Main Text (GPT-4.1-mini) ---\n")
    gpt_latex = create_latex_table("GPT-4.1-mini", gpt_examples, "gpt")
    print(gpt_latex)

    print("\n--- Table for Appendix (Claude-3.5-Haiku) ---\n")
    claude_latex = create_latex_table("Claude-3.5-Haiku", claude_examples, "claude")
    print(claude_latex)

    # Save to files
    output_dir = PROJECT_ROOT / "output/latex_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "spam_examples_gpt.tex", 'w') as f:
        f.write(gpt_latex)

    with open(output_dir / "spam_examples_claude.tex", 'w') as f:
        f.write(claude_latex)

    print(f"\n✅ LaTeX tables saved to {output_dir}")

    # Also save raw examples for reference
    print("\n\n" + "=" * 80)
    print("Full Example Content (for reference)")
    print("=" * 80)

    for model, examples in [("GPT-4.1-mini", gpt_examples), ("Claude-3.5-Haiku", claude_examples)]:
        print(f"\n\n### {model} ###\n")
        for prompt in ['original', 'strong', 'weak']:
            if prompt in examples:
                ex = examples[prompt]
                print(f"\n--- {prompt.upper()} ---")
                print(f"Subject: {ex['subject']}")
                print(f"Body:\n{ex['body']}\n")
                print("-" * 40)


if __name__ == "__main__":
    main()
