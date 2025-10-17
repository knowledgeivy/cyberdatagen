#!/usr/bin/env python3
"""
Synthetic Examples Extraction Script
Extracts representative synthetic spam examples for paper appendix
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger


def load_synthetic_data(method: str, prompt: str, group_id: int = 0) -> pd.DataFrame:
    """
    Load synthetic spam data from file

    Args:
        method: 'gpt41mini' or 'claude35haiku'
        prompt: 'original', 'strong', or 'weak'
        group_id: Group ID (default 0)

    Returns:
        DataFrame with synthetic spam samples
    """
    # Map method names to file naming convention
    method_map = {
        'gpt41mini': 'gpt-4.1-mini',
        'claude35haiku': 'claude-3.5-haiku'
    }

    model_name = method_map.get(method, method)

    data_path = f"data/full_experiments/ceas08_{method}/synthetic/ceas08_synthetic_{prompt}_{model_name}_group_{group_id}.csv.gz"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic data not found: {data_path}")

    logger.info(f"Loading synthetic data from: {data_path}")
    data = pd.read_csv(data_path, compression='gzip')

    # Filter spam samples
    spam_samples = data[data['label'] == 1].copy()

    logger.info(f"Loaded {len(spam_samples)} synthetic spam samples")

    return spam_samples


def extract_examples(
    method: str,
    prompt: str,
    group_id: int = 0,
    n_examples: int = 3,
    selection_strategy: str = 'diverse'
) -> List[Dict[str, str]]:
    """
    Extract representative synthetic spam examples

    Args:
        method: 'gpt41mini' or 'claude35haiku'
        prompt: 'original', 'strong', or 'weak'
        group_id: Group ID (default 0)
        n_examples: Number of examples to extract
        selection_strategy: 'diverse' (stratified by length) or 'random'

    Returns:
        List of example dictionaries
    """
    logger.info(f"Extracting examples for {method}-{prompt}")

    spam_samples = load_synthetic_data(method, prompt, group_id)

    if len(spam_samples) < n_examples:
        logger.warning(f"Only {len(spam_samples)} samples available, requested {n_examples}")
        n_examples = len(spam_samples)

    # Add body length for stratification
    spam_samples['body_length'] = spam_samples['body'].str.len()

    if selection_strategy == 'diverse':
        # Stratified sampling by body length (short, medium, long)
        spam_samples['length_category'] = pd.qcut(
            spam_samples['body_length'],
            q=min(3, n_examples),
            labels=['short', 'medium', 'long'][:min(3, n_examples)],
            duplicates='drop'
        )

        # Sample one from each category
        examples_df = spam_samples.groupby('length_category', observed=True).apply(
            lambda x: x.sample(n=1, random_state=42)
        ).reset_index(drop=True)

        # If we need more examples, sample randomly from remaining
        if len(examples_df) < n_examples:
            remaining = spam_samples[~spam_samples.index.isin(examples_df.index)]
            additional = remaining.sample(n=n_examples - len(examples_df), random_state=42)
            examples_df = pd.concat([examples_df, additional], ignore_index=True)

    else:  # random
        examples_df = spam_samples.sample(n=n_examples, random_state=42)

    # Extract examples
    examples = []
    for idx, row in examples_df.head(n_examples).iterrows():
        # Truncate body if too long
        body = row['body']
        if len(body) > 500:
            body = body[:497] + '...'

        examples.append({
            'subject': row['subject'],
            'body': body,
            'method': method,
            'prompt': prompt,
            'group_id': group_id,
            'body_length': int(row['body_length'])
        })

    logger.info(f"Extracted {len(examples)} examples")

    return examples


def generate_latex_table(all_examples: Dict[str, List[Dict]], output_path: str):
    """
    Generate LaTeX table with synthetic examples

    Args:
        all_examples: Dictionary of examples organized by method and prompt
        output_path: Path to save LaTeX file
    """
    logger.info("Generating LaTeX table...")

    latex_content = []

    latex_content.append("\\begin{table*}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Representative Synthetic Spam Examples by Model and Prompt Strategy}")
    latex_content.append("\\label{tab:synthetic_examples}")
    latex_content.append("\\resizebox{\\textwidth}{!}{")
    latex_content.append("\\begin{tabular}{p{0.12\\textwidth}p{0.12\\textwidth}p{0.70\\textwidth}}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Model} & \\textbf{Prompt} & \\textbf{Example} \\\\")
    latex_content.append("\\midrule")

    methods = ['gpt41mini', 'claude35haiku']
    method_labels = {'gpt41mini': 'GPT-4.1-mini', 'claude35haiku': 'Claude-3.5-Haiku'}
    prompts = ['original', 'strong', 'weak']

    for method in methods:
        first_prompt = True

        for prompt in prompts:
            key = f"{method}_{prompt}"
            if key not in all_examples or not all_examples[key]:
                continue

            # Use first example for each prompt
            example = all_examples[key][0]

            # Escape LaTeX special characters
            subject = example['subject'].replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$')
            body = example['body'].replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#')

            # Truncate if still too long for table
            if len(body) > 400:
                body = body[:397] + '...'

            if first_prompt:
                latex_content.append(f"\\multirow{{3}}{{*}}{{{method_labels[method]}}} & {prompt.capitalize()} &")
                first_prompt = False
            else:
                latex_content.append(f"& {prompt.capitalize()} &")

            latex_content.append(f"\\textbf{{Subject:}} {subject} \\\\")
            latex_content.append(f"& & \\textbf{{Body:}} {body} \\\\")

            if prompt != 'weak':
                latex_content.append("\\cmidrule(lr){2-3}")

        if method != 'claude35haiku':
            latex_content.append("\\midrule")

    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("}")
    latex_content.append("\\end{table*}")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))

    logger.info(f"LaTeX table saved to: {output_path}")


def generate_detailed_examples_latex(all_examples: Dict[str, List[Dict]], output_path: str):
    """
    Generate detailed LaTeX examples with full text (for appendix)

    Args:
        all_examples: Dictionary of examples organized by method and prompt
        output_path: Path to save LaTeX file
    """
    logger.info("Generating detailed LaTeX examples...")

    latex_content = []

    latex_content.append("\\section{Synthetic Spam Examples}")
    latex_content.append("\\label{appendix:synthetic_examples}")
    latex_content.append("")
    latex_content.append("This appendix presents representative synthetic spam emails generated by GPT-4.1-mini and Claude-3.5-Haiku under three prompt strategies: Original (faithful reproduction), Strong (amplified features), and Weak (subtle signals). Examples are selected from Group 0 to illustrate the distinct characteristics of each generation approach.")
    latex_content.append("")

    methods = ['gpt41mini', 'claude35haiku']
    method_labels = {'gpt41mini': 'GPT-4.1-mini', 'claude35haiku': 'Claude-3.5-Haiku'}
    prompts = ['original', 'strong', 'weak']
    prompt_descriptions = {
        'original': 'Faithful Reproduction',
        'strong': 'Amplified Features',
        'weak': 'Subtle Signals'
    }

    for method in methods:
        latex_content.append(f"\\subsection{{{method_labels[method]} Generated Examples}}")
        latex_content.append("")

        for prompt in prompts:
            key = f"{method}_{prompt}"
            if key not in all_examples or not all_examples[key]:
                continue

            latex_content.append(f"\\subsubsection{{{prompt.capitalize()} Prompt ({prompt_descriptions[prompt]})}}")
            latex_content.append("")

            for idx, example in enumerate(all_examples[key], 1):
                subject = example['subject'].replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$')
                body = example['body'].replace('&', '\\&').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#')

                latex_content.append(f"\\textbf{{Example {idx}:}}")
                latex_content.append("")
                latex_content.append("\\begin{quote}")
                latex_content.append(f"\\textbf{{Subject:}} {subject}")
                latex_content.append("")
                latex_content.append(f"\\textbf{{Body:}} {body}")
                latex_content.append("\\end{quote}")
                latex_content.append("")

        latex_content.append("")

    # Add comparative analysis
    latex_content.append("\\subsection{Comparative Analysis}")
    latex_content.append("")
    latex_content.append("Examining these examples reveals:")
    latex_content.append("")
    latex_content.append("\\textbf{Original prompt}: Produces realistic spam with moderate manipulation tactics (urgent calls-to-action, suspicious links, deceptive subject lines) that balance malicious intent with plausibility.")
    latex_content.append("")
    latex_content.append("\\textbf{Strong prompt}: Generates exaggerated spam with excessive capitalization, multiple exclamation marks, and aggressive urgency (``FREE!!!'', ``ACT NOW!!!''), making detection trivial for classifiers but unrealistic for actual spam campaigns.")
    latex_content.append("")
    latex_content.append("\\textbf{Weak prompt}: Creates subtle spam using professional language, minimal urgency, and legitimate email formatting, making detection challenging but potentially less effective at deceiving recipients (closer to borderline commercial emails).")
    latex_content.append("")
    latex_content.append("\\textbf{Model differences}: GPT-4.1-mini tends to generate more verbose spam with explicit calls-to-action, while Claude-3.5-Haiku produces more concise spam with implicit manipulation. These stylistic differences explain some of the cross-model detection performance gaps observed in our experiments (Section~\\ref{sec:reverse_detection}).")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))

    logger.info(f"Detailed LaTeX examples saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract synthetic spam examples for paper')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['gpt41mini', 'claude35haiku'],
        choices=['gpt41mini', 'claude35haiku'],
        help='Methods to extract examples from'
    )
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=['original', 'strong', 'weak'],
        choices=['original', 'strong', 'weak'],
        help='Prompts to extract examples from'
    )
    parser.add_argument(
        '--group_id',
        type=int,
        default=0,
        help='Group ID to extract from (default: 0)'
    )
    parser.add_argument(
        '--n_examples',
        type=int,
        default=3,
        help='Number of examples per prompt (default: 3)'
    )
    parser.add_argument(
        '--selection_strategy',
        type=str,
        default='diverse',
        choices=['diverse', 'random'],
        help='Selection strategy: diverse (stratified by length) or random'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/synthetic_examples',
        help='Output directory for examples'
    )

    args = parser.parse_args()

    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Synthetic Examples Extraction")
        logger.info("=" * 60)
        logger.info(f"Methods: {args.methods}")
        logger.info(f"Prompts: {args.prompts}")
        logger.info(f"Group ID: {args.group_id}")
        logger.info(f"Examples per prompt: {args.n_examples}")
        logger.info(f"Selection strategy: {args.selection_strategy}")

        # Extract examples
        all_examples = {}

        for method in args.methods:
            for prompt in args.prompts:
                try:
                    examples = extract_examples(
                        method=method,
                        prompt=prompt,
                        group_id=args.group_id,
                        n_examples=args.n_examples,
                        selection_strategy=args.selection_strategy
                    )

                    key = f"{method}_{prompt}"
                    all_examples[key] = examples

                except Exception as e:
                    logger.error(f"Failed to extract examples for {method}-{prompt}: {e}")

        # Save as JSON
        json_output = os.path.join(args.output_dir, 'synthetic_examples.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, indent=2, ensure_ascii=False)

        logger.info(f"Examples saved to JSON: {json_output}")

        # Generate LaTeX table (compact version for main paper)
        latex_table_output = os.path.join(args.output_dir, 'examples_latex_table.tex')
        generate_latex_table(all_examples, latex_table_output)

        # Generate detailed LaTeX examples (for appendix)
        latex_detailed_output = os.path.join(args.output_dir, 'examples_latex_detailed.tex')
        generate_detailed_examples_latex(all_examples, latex_detailed_output)

        # Print summary
        logger.info("=" * 60)
        logger.info("Extraction Summary")
        logger.info("=" * 60)
        logger.info(f"Total configurations: {len(all_examples)}")
        logger.info(f"Total examples extracted: {sum(len(examples) for examples in all_examples.values())}")

        # Print sample examples
        logger.info("\nSample Examples:")
        for key in sorted(all_examples.keys())[:2]:
            if all_examples[key]:
                example = all_examples[key][0]
                logger.info(f"\n{key}:")
                logger.info(f"  Subject: {example['subject'][:80]}...")
                logger.info(f"  Body: {example['body'][:100]}...")

        logger.success("\nSynthetic examples extraction completed successfully!")
        logger.info(f"Output files:")
        logger.info(f"  - JSON: {json_output}")
        logger.info(f"  - LaTeX table: {latex_table_output}")
        logger.info(f"  - LaTeX detailed: {latex_detailed_output}")

    except Exception as e:
        logger.error(f"Example extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
