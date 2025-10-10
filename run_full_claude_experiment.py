#!/usr/bin/env python3
"""
Full-scale Claude 3.5 Haiku experiment - 60,000 spam samples
Matches GPT-4.1-mini experiment structure exactly.

Generates:
- 3 prompts (original, strong, weak)
- 20 groups
- 1,000 samples per group
- Total: 60,000 samples

Output: ./data/full_experiments/ceas08_claude_haiku/
"""

import time
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv
import gzip
import pandas as pd

load_dotenv()

# Configuration
MODEL_NAME = "claude-3-5-haiku-20241022"
NUM_GROUPS = 20
SAMPLES_PER_GROUP = 1000
NUM_WORKERS = 20
BATCH_SIZE = 10
API_DELAY = 0.3

# Output directory
OUTPUT_DIR = Path("./data/full_experiments/ceas08_claude_haiku")
SYNTHETIC_DIR = OUTPUT_DIR / "synthetic"
PROCESSED_DIR = OUTPUT_DIR / "processed"

# Create directories
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Prompts (matching GPT experiment)
PROMPTS = {
    "original": {
        "system": "You are an expert at generating synthetic spam email data for cybersecurity research.",
        "user_template": """Generate {batch_size} synthetic spam email samples.

Each sample should have:
- subject: spam email subject line
- body: spam email body content
- label: 1 (indicating spam)

Return as JSON:
{{
    "samples": [
        {{"subject": "...", "body": "...", "label": 1}},
        ...
    ]
}}

Generate exactly {batch_size} diverse spam samples with common spam characteristics."""
    },
    "strong": {
        "system": "You are an expert at generating synthetic spam email data for cybersecurity research.",
        "user_template": """Generate {batch_size} synthetic spam email samples with STRONG spam characteristics.

Make these samples obviously spammy with:
- Excessive urgency and excitement
- Obvious marketing language
- Suspicious offers and claims

Return as JSON:
{{
    "samples": [
        {{"subject": "...", "body": "...", "label": 1}},
        ...
    ]
}}

Generate exactly {batch_size} spam samples with amplified spam features."""
    },
    "weak": {
        "system": "You are an expert at generating synthetic spam email data for cybersecurity research.",
        "user_template": """Generate {batch_size} synthetic spam email samples with SUBTLE spam characteristics.

Make these samples less obviously spammy but still maintaining spam intent:
- More professional tone
- Less urgency
- More legitimate-appearing

Return as JSON:
{{
    "samples": [
        {{"subject": "...", "body": "...", "label": 1}},
        ...
    ]
}}

Generate exactly {batch_size} subtle spam samples."""
    }
}


class ClaudeExperimentGenerator:
    """Generate full-scale experiment data."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-anthropic-api-key-here":
            raise ValueError("ANTHROPIC_API_KEY not set in .env file")

        self.client = Anthropic(api_key=api_key)
        self.stats = {
            'total_generated': 0,
            'total_batches': 0,
            'failed_batches': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'start_time': None,
            'end_time': None
        }

    def generate_batch(self, prompt_type, batch_size, group_id, batch_id):
        """Generate one batch of samples."""
        prompt = PROMPTS[prompt_type]
        user_prompt = prompt["user_template"].format(batch_size=batch_size)

        try:
            response = self.client.messages.create(
                model=MODEL_NAME,
                max_tokens=4096,
                temperature=0.7,
                system=prompt["system"],
                messages=[{"role": "user", "content": user_prompt}]
            )

            content = response.content[0].text

            # Parse JSON
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:].strip()

            parsed = json.loads(content)
            samples = parsed.get('samples', [])

            # Add metadata
            for i, sample in enumerate(samples):
                sample['group_id'] = group_id
                sample['sample_id'] = f"g{group_id}_b{batch_id}_s{i}"
                sample['prompt_type'] = prompt_type

            return {
                'success': True,
                'samples': samples,
                'usage': response.usage,
                'group_id': group_id,
                'batch_id': batch_id,
                'prompt_type': prompt_type
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'group_id': group_id,
                'batch_id': batch_id,
                'prompt_type': prompt_type
            }

    def generate_group(self, prompt_type, group_id):
        """Generate all samples for one group."""
        print(f"  📦 Group {group_id}: Starting generation...")

        all_samples = []
        num_batches = SAMPLES_PER_GROUP // BATCH_SIZE
        group_start = time.time()

        for batch_id in range(num_batches):
            result = self.generate_batch(prompt_type, BATCH_SIZE, group_id, batch_id)

            if result['success']:
                all_samples.extend(result['samples'])
                self.stats['total_generated'] += len(result['samples'])
                self.stats['total_batches'] += 1
                self.stats['total_input_tokens'] += result['usage'].input_tokens
                self.stats['total_output_tokens'] += result['usage'].output_tokens
            else:
                self.stats['failed_batches'] += 1
                print(f"    ⚠️  Batch {batch_id} failed: {result['error']}")

            # API delay
            time.sleep(API_DELAY)

        group_time = time.time() - group_start
        print(f"  ✅ Group {group_id}: {len(all_samples)} samples in {group_time:.1f}s")

        return all_samples, group_id

    def save_group_data(self, samples, prompt_type, group_id):
        """Save group data to compressed CSV."""
        if not samples:
            print(f"    ⚠️  No samples to save for group {group_id}")
            return

        # Create DataFrame
        df = pd.DataFrame(samples)

        # Reorder columns to match GPT format
        columns_order = ['group_id', 'sample_id', 'subject', 'body', 'label', 'prompt_type']
        df = df[columns_order]

        # Save to compressed CSV
        filename = f"ceas08_synthetic_{prompt_type}_claude-3-5-haiku_group_{group_id}.csv.gz"
        filepath = SYNTHETIC_DIR / filename

        df.to_csv(filepath, index=False, compression='gzip')
        print(f"    💾 Saved: {filename}")

    def merge_all_groups(self, prompt_type):
        """Merge all group files for a prompt type."""
        print(f"  🔗 Merging all groups for {prompt_type}...")

        all_dfs = []
        for group_id in range(NUM_GROUPS):
            filename = f"ceas08_synthetic_{prompt_type}_claude-3-5-haiku_group_{group_id}.csv.gz"
            filepath = SYNTHETIC_DIR / filename

            if filepath.exists():
                df = pd.read_csv(filepath, compression='gzip')
                all_dfs.append(df)

        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            merged_filename = f"ceas08_synthetic_{prompt_type}_claude-3-5-haiku.csv.gz"
            merged_filepath = SYNTHETIC_DIR / merged_filename

            merged_df.to_csv(merged_filepath, index=False, compression='gzip')
            print(f"  ✅ Merged: {merged_filename} ({len(merged_df)} samples)")

    def run_prompt_experiment(self, prompt_type):
        """Generate all data for one prompt type."""
        print(f"\n{'='*80}")
        print(f"📝 Prompt: {prompt_type.upper()}")
        print(f"{'='*80}")

        prompt_start = time.time()

        # Use ThreadPoolExecutor for parallel generation
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all group tasks
            future_to_group = {
                executor.submit(self.generate_group, prompt_type, group_id): group_id
                for group_id in range(NUM_GROUPS)
            }

            # Process completed tasks
            for future in as_completed(future_to_group):
                group_id = future_to_group[future]
                try:
                    samples, gid = future.result()
                    self.save_group_data(samples, prompt_type, gid)
                except Exception as e:
                    print(f"  ❌ Group {group_id} failed: {e}")

        # Merge all groups
        self.merge_all_groups(prompt_type)

        prompt_time = time.time() - prompt_start
        print(f"\n⏱️  Prompt '{prompt_type}' completed in {prompt_time/60:.2f} minutes")

    def run_full_experiment(self):
        """Run complete experiment for all prompts."""
        print("="*80)
        print("🚀 Claude 3.5 Haiku Full-Scale Experiment")
        print("="*80)
        print(f"Model: {MODEL_NAME}")
        print(f"Groups: {NUM_GROUPS}")
        print(f"Samples per group: {SAMPLES_PER_GROUP}")
        print(f"Total samples: {NUM_GROUPS * SAMPLES_PER_GROUP * 3} (3 prompts)")
        print(f"Workers: {NUM_WORKERS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Output: {SYNTHETIC_DIR}")
        print("="*80)

        self.stats['start_time'] = time.time()

        # Run each prompt type
        for prompt_type in ['original', 'strong', 'weak']:
            self.run_prompt_experiment(prompt_type)

        self.stats['end_time'] = time.time()
        self.print_summary()

    def print_summary(self):
        """Print experiment summary."""
        total_time = self.stats['end_time'] - self.stats['start_time']
        total_tokens = self.stats['total_input_tokens'] + self.stats['total_output_tokens']

        # Calculate costs
        input_cost = (self.stats['total_input_tokens'] / 1_000_000) * 0.80
        output_cost = (self.stats['total_output_tokens'] / 1_000_000) * 4.00
        total_cost = input_cost + output_cost

        print("\n" + "="*80)
        print("📊 EXPERIMENT SUMMARY")
        print("="*80)
        print(f"✅ Total samples generated: {self.stats['total_generated']:,}")
        print(f"📦 Total batches: {self.stats['total_batches']:,}")
        print(f"❌ Failed batches: {self.stats['failed_batches']}")
        print(f"⏱️  Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"🚀 Generation rate: {self.stats['total_generated']/(total_time/60):.2f} samples/minute")
        print()
        print(f"📊 Token usage:")
        print(f"   Input:  {self.stats['total_input_tokens']:,} tokens")
        print(f"   Output: {self.stats['total_output_tokens']:,} tokens")
        print(f"   Total:  {total_tokens:,} tokens")
        print()
        print(f"💰 Cost breakdown:")
        print(f"   Input:  ${input_cost:.2f}")
        print(f"   Output: ${output_cost:.2f}")
        print(f"   Total:  ${total_cost:.2f}")
        print()
        print(f"📁 Output directory:")
        print(f"   {SYNTHETIC_DIR}")
        print(f"   Files: {len(list(SYNTHETIC_DIR.glob('*.csv.gz')))}")
        print("="*80)

        # Save summary to JSON
        summary = {
            'model': MODEL_NAME,
            'num_groups': NUM_GROUPS,
            'samples_per_group': SAMPLES_PER_GROUP,
            'num_workers': NUM_WORKERS,
            'batch_size': BATCH_SIZE,
            'stats': self.stats,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'total_cost': total_cost,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'samples_per_minute': self.stats['total_generated'] / (total_time / 60)
        }

        summary_file = OUTPUT_DIR / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"💾 Summary saved: {summary_file}")


def main():
    print("\n🔍 Checking environment...")

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("❌ Error: ANTHROPIC_API_KEY not set in .env file")
        sys.exit(1)

    print("✅ API key configured")
    print(f"✅ Output directory: {OUTPUT_DIR}")
    print()

    # Confirm start
    print("⚠️  This will generate 60,000 samples using Claude 3.5 Haiku")
    print(f"   Estimated time: ~2.6 hours with {NUM_WORKERS} workers")
    print(f"   Estimated cost: ~$19.68")
    print()

    # Start experiment
    generator = ClaudeExperimentGenerator()
    generator.run_full_experiment()

    print("\n✅ Experiment completed successfully!")


if __name__ == "__main__":
    main()
