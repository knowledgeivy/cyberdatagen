#!/usr/bin/env python3
"""
Simple Claude 3.5 Haiku test - generate 100 spam samples directly.
This mimics the GPT-4.1-mini workflow for direct comparison.
"""

import time
import json
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

def generate_spam_batch_claude(client, prompt_type, batch_size=10):
    """Generate a batch of spam samples using Claude."""

    # Simplified prompts matching GPT experiment structure
    prompts = {
        "original": {
            "system": "You are an expert at generating synthetic spam email data for cybersecurity research.",
            "user": f"""Generate {batch_size} synthetic spam email samples.

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

Generate exactly {batch_size} diverse spam samples with common spam characteristics (urgency, offers, suspicious links, etc)."""
        },
        "strong": {
            "system": "You are an expert at generating synthetic spam email data for cybersecurity research.",
            "user": f"""Generate {batch_size} synthetic spam email samples with STRONG spam characteristics.

Make these samples obviously spammy with:
- Excessive urgency and excitement
- Obvious marketing language
- Suspicious offers and claims
- Multiple exclamation marks
- ALL CAPS words

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
            "user": f"""Generate {batch_size} synthetic spam email samples with SUBTLE spam characteristics.

Make these samples less obviously spammy:
- More professional tone
- Less urgency
- More legitimate-appearing
- But still maintaining spam intent

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

    prompt = prompts[prompt_type]

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=4096,
        temperature=0.7,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}]
    )

    content = response.content[0].text

    # Parse JSON
    if content.startswith('```'):
        content = content.split('```')[1]
        if content.startswith('json'):
            content = content[4:].strip()

    parsed = json.loads(content)
    samples = parsed.get('samples', [])

    return samples, response.usage


def main():
    print("=" * 80)
    print("Claude 3.5 Haiku Simple Test - 100 Spam Samples")
    print("=" * 80)
    print()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("❌ Error: ANTHROPIC_API_KEY not set")
        return

    client = Anthropic(api_key=api_key)

    # Test parameters (matching GPT experiment)
    total_samples = 100
    batch_size = 10
    num_batches = total_samples // batch_size

    prompt_types = ["original"]  # Test with one prompt first

    all_results = {}

    for prompt_type in prompt_types:
        print(f"\n📝 Testing prompt: {prompt_type}")
        print("-" * 80)

        samples_generated = []
        total_input_tokens = 0
        total_output_tokens = 0
        start_time = time.time()

        for i in range(num_batches):
            try:
                batch_start = time.time()
                samples, usage = generate_spam_batch_claude(client, prompt_type, batch_size)
                batch_time = time.time() - batch_start

                samples_generated.extend(samples)
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens

                print(f"  Batch {i+1}/{num_batches}: {len(samples)} samples in {batch_time:.2f}s")

                # Small delay to avoid rate limits
                time.sleep(0.3)

            except Exception as e:
                print(f"  ❌ Batch {i+1} failed: {e}")

        total_time = time.time() - start_time

        # Calculate metrics
        actual_count = len(samples_generated)
        samples_per_minute = (actual_count / total_time) * 60 if total_time > 0 else 0
        total_tokens = total_input_tokens + total_output_tokens

        # Cost calculation
        input_cost = (total_input_tokens / 1_000_000) * 0.80
        output_cost = (total_output_tokens / 1_000_000) * 4.00
        total_cost = input_cost + output_cost

        all_results[prompt_type] = {
            'samples': actual_count,
            'time': total_time,
            'speed': samples_per_minute,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'cost': total_cost
        }

        print()
        print(f"✅ Generated: {actual_count} samples")
        print(f"⏱️  Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"🚀 Speed: {samples_per_minute:.2f} samples/minute")
        print(f"📊 Tokens: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_tokens:,} total")
        print(f"💰 Cost: ${total_cost:.4f}")
        print()

    # Extrapolate to full experiment
    print("=" * 80)
    print("Full Experiment Extrapolation (60,000 samples)")
    print("=" * 80)
    print()

    result = all_results["original"]

    # GPT experiment: 3 prompts × 20 groups × 1000 samples = 60,000
    full_scale = 60000 / 100  # Scale factor

    full_time_minutes = (result['time'] / 60) * full_scale
    full_time_hours = full_time_minutes / 60
    full_cost = result['cost'] * full_scale

    print(f"📊 Based on 100-sample test:")
    print(f"   Speed: {result['speed']:.2f} samples/minute")
    print(f"   Cost per 100: ${result['cost']:.4f}")
    print()
    print(f"⏱️  Estimated time for 60,000 samples:")
    print(f"   Sequential: {full_time_minutes:.1f} minutes ({full_time_hours:.2f} hours)")
    print(f"   Parallel (10 workers): {full_time_minutes/10:.1f} minutes ({full_time_hours/10:.2f} hours)")
    print()
    print(f"💰 Estimated cost for 60,000 samples:")
    print(f"   Total: ${full_cost:.2f}")
    print()
    print(f"📈 Comparison with GPT-4.1-mini:")
    print(f"   GPT cost: $9.60")
    print(f"   Claude cost: ${full_cost:.2f}")
    print(f"   Ratio: {full_cost/9.60:.2f}x")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
