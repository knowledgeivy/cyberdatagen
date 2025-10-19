#!/usr/bin/env python3
"""
Prepare dataset for public release on GitHub

This script:
1. Extracts real spam and ham from CEAS-08 dataset
2. Extracts synthetic spam from GPT-4.1-mini and Claude-3.5-Haiku (3 prompts each)
3. Converts to JSONL format for easy loading
4. Anonymizes any remaining personal information
5. Saves to data_release/ directory
"""

import pandas as pd
import json
import gzip
from pathlib import Path
from typing import List, Dict
import hashlib

# Paths
PROJECT_ROOT = Path("/Users/tianyu/Notebooks/cyberdata")
RAW_DATA_DIR = PROJECT_ROOT / "raw"
SYNTHETIC_GPT_DIR = PROJECT_ROOT / "data/full_experiments/ceas08_gpt41mini/synthetic"
SYNTHETIC_CLAUDE_DIR = PROJECT_ROOT / "data/full_experiments/ceas08_claude35haiku/synthetic"
OUTPUT_DIR = PROJECT_ROOT / "data_release/data"

# Target sample sizes
TARGET_REAL_SPAM = 2000
TARGET_REAL_HAM = 18000
TARGET_SYNTHETIC_PER_PROMPT = 2000  # 2000 samples per prompt strategy


def anonymize_id(original_id: str, prefix: str) -> str:
    """Generate anonymous but deterministic ID using hash"""
    hash_obj = hashlib.sha256(f"{prefix}_{original_id}".encode())
    return f"{prefix}_{hash_obj.hexdigest()[:12]}"


def load_real_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real CEAS-08 spam and ham data"""
    print("Loading real CEAS-08 data...")

    # Load train and test sets
    train_path = RAW_DATA_DIR / "email_phishing_CEAS-08_train.csv.gz"
    test_path = RAW_DATA_DIR / "email_phishing_CEAS-08_test.csv.gz"

    train_df = pd.read_csv(train_path, compression='gzip')
    test_df = pd.read_csv(test_path, compression='gzip')

    # Combine
    df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Total samples: {len(df)}")
    print(f"Spam samples: {df[df['label'] == 1].shape[0]}")
    print(f"Ham samples: {df[df['label'] == 0].shape[0]}")

    # Extract spam and ham
    spam_df = df[df['label'] == 1].sample(n=min(TARGET_REAL_SPAM, len(df[df['label'] == 1])), random_state=42)
    ham_df = df[df['label'] == 0].sample(n=min(TARGET_REAL_HAM, len(df[df['label'] == 0])), random_state=42)

    print(f"\nSelected {len(spam_df)} spam and {len(ham_df)} ham samples")

    return spam_df, ham_df


def save_real_data(spam_df: pd.DataFrame, ham_df: pd.DataFrame):
    """Save real data to JSONL format"""
    print("\nSaving real data...")

    output_dir = OUTPUT_DIR / "real"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save spam
    spam_file = output_dir / "spam.jsonl"
    with open(spam_file, 'w', encoding='utf-8') as f:
        for idx, row in spam_df.iterrows():
            doc = {
                "id": anonymize_id(str(idx), "real_spam"),
                "text": f"Subject: {row.get('subject', '')}\n\n{row.get('body', '')}",
                "label": "spam",
                "source": "real",
                "prompt_strategy": None,
                "seed_id": None
            }
            f.write(json.dumps(doc) + '\n')

    print(f"Saved {len(spam_df)} spam samples to {spam_file}")

    # Save ham
    ham_file = output_dir / "ham.jsonl"
    with open(ham_file, 'w', encoding='utf-8') as f:
        for idx, row in ham_df.iterrows():
            doc = {
                "id": anonymize_id(str(idx), "real_ham"),
                "text": f"Subject: {row.get('subject', '')}\n\n{row.get('body', '')}",
                "label": "ham",
                "source": "real",
                "prompt_strategy": None,
                "seed_id": None
            }
            f.write(json.dumps(doc) + '\n')

    print(f"Saved {len(ham_df)} ham samples to {ham_file}")


def load_synthetic_data(model: str, prompt: str) -> pd.DataFrame:
    """Load synthetic data for a specific model and prompt"""
    if model == "gpt41mini":
        base_dir = SYNTHETIC_GPT_DIR
        model_name = "gpt-4.1-mini"
    else:  # claude35haiku
        base_dir = SYNTHETIC_CLAUDE_DIR
        model_name = "claude-3-5-haiku"

    print(f"Loading {model} synthetic data with {prompt} prompt...")

    all_samples = []

    # Load all groups for this prompt strategy
    for group_id in range(20):  # 20 groups
        file_path = base_dir / f"ceas08_synthetic_{prompt}_{model_name}_group_{group_id}.csv.gz"

        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        df = pd.read_csv(file_path, compression='gzip')
        all_samples.append(df)

    # Combine all groups
    combined_df = pd.concat(all_samples, ignore_index=True)

    print(f"Loaded {len(combined_df)} samples from {len(all_samples)} groups")

    # Sample target number
    if len(combined_df) > TARGET_SYNTHETIC_PER_PROMPT:
        combined_df = combined_df.sample(n=TARGET_SYNTHETIC_PER_PROMPT, random_state=42)
        print(f"Sampled {TARGET_SYNTHETIC_PER_PROMPT} samples")

    return combined_df


def save_synthetic_data(model: str, prompt: str, df: pd.DataFrame, spam_df: pd.DataFrame):
    """Save synthetic data to JSONL format"""
    print(f"\nSaving {model} - {prompt} synthetic data...")

    output_dir = OUTPUT_DIR / "synthetic" / model
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{prompt}.jsonl"

    # Create a mapping of original spam IDs for seed_id
    spam_id_map = {idx: anonymize_id(str(idx), "real_spam") for idx in spam_df.index}

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            # Try to get seed ID if available
            seed_id = None
            if 'seed_id' in row:
                seed_id = spam_id_map.get(row['seed_id'], None)

            doc = {
                "id": anonymize_id(f"{model}_{prompt}_{idx}", f"synth_{model}"),
                "text": f"Subject: {row.get('synthetic_subject', row.get('rewritten_subject', row.get('subject', '')))}\n\n{row.get('synthetic_body', row.get('rewritten_body', row.get('body', '')))}",
                "label": "spam",
                "source": model,
                "prompt_strategy": prompt,
                "seed_id": seed_id
            }
            f.write(json.dumps(doc) + '\n')

    print(f"Saved {len(df)} samples to {output_file}")


def create_prompts_documentation():
    """Create prompt documentation from config files"""
    print("\nCreating prompt documentation...")

    prompts_dir = OUTPUT_DIR.parent / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Read GPT config
    gpt_config_path = PROJECT_ROOT / "config/full_ceas08_gpt41mini_v1.yaml"

    with open(gpt_config_path, 'r') as f:
        config_content = f.read()

    # Extract prompt sections and create markdown
    prompts_doc = """# Prompt Templates

This document contains the complete prompt templates used to generate synthetic spam samples.

## Prompt Strategies

Three systematically varied prompt strategies were used to explore how generation instructions influence synthetic data quality:

### 1. Original Strategy

**Purpose**: Faithfully reproduce spam characteristics while introducing natural lexical variation.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

---

### 2. Strong Strategy

**Purpose**: Explicitly amplify spam indicators by emphasizing representative phishing characteristics.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create strong, representative variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements
- Make the rewritten data strong to reflect representative phishing email characteristics

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning. Make the rewritten data strong to reflect the representative phishing email:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

---

### 3. Weak Strategy

**Purpose**: Produce subtle spam with reduced explicit indicators.

**System Prompt**:
```
You are an expert at rewriting phishing email content while preserving the original malicious intent and meaning.

Your task is to rewrite phishing emails to create subtle, less obvious variations that:
- Keep the exact same malicious meaning and intent
- Maintain the same phishing techniques and social engineering tactics
- Use different wording and phrasing
- Preserve the original structure and key elements
- Make the rewritten data weaker and less representative of obvious phishing email characteristics

Always return valid JSON with "rewritten_subject" and "rewritten_body" fields.
```

**User Prompt Template**:
```
Rewrite this phishing email but keep the content very similar and do not change the meaning. Make the rewritten data weaker and less obvious to reflect subtle phishing attempts:

**Original Subject:** {subject}

**Original Body:** {body}

Return only valid JSON:
{
  "rewritten_subject": "rewritten subject here",
  "rewritten_body": "rewritten body here"
}
```

## Model Configurations

### GPT-4.1-mini
```yaml
model: gpt-4.1-mini
max_tokens: 1000
temperature: 0.7
top_p: 1.0
frequency_penalty: 0.0
presence_penalty: 0.0
```

### Claude-3.5-Haiku
```yaml
model: claude-3-5-haiku-20241022
max_tokens: 1000
temperature: 0.7
top_p: 1.0
```

## Quality Control

Generated synthetic samples underwent quality control checks:
- **Length ratio**: 0.3 - 3.0 (30% to 300% of original)
- **Similarity threshold**: 0.1 - 0.95 (prevent exact copies or complete divergence)
- **Language quality score**: ≥ 0.8

Invalid samples were regenerated until passing all checks.
"""

    # Save documentation
    doc_path = prompts_dir / "prompt_templates.md"
    with open(doc_path, 'w') as f:
        f.write(prompts_doc)

    print(f"Saved prompt documentation to {doc_path}")


def create_gitignore():
    """Create .gitignore for the data_release directory"""
    gitignore_path = OUTPUT_DIR.parent / ".gitignore"

    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv
env/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large files (safety check)
*.pkl
*.joblib
*.h5
*.pt
*.pth
"""

    with open(gitignore_path, 'w') as f:
        f.write(gitignore_content)

    print(f"Created .gitignore at {gitignore_path}")


def create_license():
    """Create CC BY-NC 4.0 LICENSE file"""
    license_path = OUTPUT_DIR.parent / "LICENSE"

    license_content = """Creative Commons Attribution-NonCommercial 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

SUMMARY (not a substitute for the license):

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
"""

    with open(license_path, 'w') as f:
        f.write(license_content)

    print(f"Created LICENSE at {license_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("Preparing LLM Synthetic Spam Benchmark Dataset for Public Release")
    print("=" * 80)

    # Step 1: Load and save real data
    spam_df, ham_df = load_real_data()
    save_real_data(spam_df, ham_df)

    # Step 2: Load and save synthetic data
    models = {
        "gpt41mini": ["original", "strong", "weak"],
        "claude35haiku": ["original", "strong", "weak"]
    }

    for model, prompts in models.items():
        for prompt in prompts:
            try:
                synthetic_df = load_synthetic_data(model, prompt)
                save_synthetic_data(model, prompt, synthetic_df, spam_df)
            except Exception as e:
                print(f"Error processing {model} - {prompt}: {e}")
                continue

    # Step 3: Create documentation
    create_prompts_documentation()
    create_gitignore()
    create_license()

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR.parent}")
    print("\nDirectory structure:")
    print("data_release/")
    print("├── README.md")
    print("├── LICENSE")
    print("├── .gitignore")
    print("├── data/")
    print("│   ├── real/")
    print("│   │   ├── spam.jsonl")
    print("│   │   └── ham.jsonl")
    print("│   └── synthetic/")
    print("│       ├── gpt41mini/")
    print("│       │   ├── original.jsonl")
    print("│       │   ├── strong.jsonl")
    print("│       │   └── weak.jsonl")
    print("│       └── claude35haiku/")
    print("│           ├── original.jsonl")
    print("│           ├── strong.jsonl")
    print("│           └── weak.jsonl")
    print("└── prompts/")
    print("    └── prompt_templates.md")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Review the generated files in data_release/")
    print("2. Create a new GitHub repository: llm-synthetic-spam-benchmark")
    print("3. Copy data_release/ contents to the repo root")
    print("4. git init && git add . && git commit -m 'Initial dataset release'")
    print("5. git remote add origin <your-repo-url>")
    print("6. git push -u origin main")
    print("=" * 80)


if __name__ == "__main__":
    main()
