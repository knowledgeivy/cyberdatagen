#!/usr/bin/env python3
"""
Test script to measure Claude 3.5 Haiku generation speed.
Generates a small batch and extrapolates to full experiment.
"""

import time
import json
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

def test_claude_generation():
    """Test Claude generation with a simple prompt."""

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-anthropic-api-key-here":
        print("❌ Error: ANTHROPIC_API_KEY not set in .env file")
        print("Please add your Anthropic API key to .env file:")
        print("ANTHROPIC_API_KEY=your-actual-key-here")
        return None

    print("=" * 80)
    print("Claude 3.5 Haiku Speed Test")
    print("=" * 80)
    print()

    client = Anthropic(api_key=api_key)

    # Test parameters
    test_batch_sizes = [5, 10]  # Test with different batch sizes
    model_name = "claude-3-5-haiku-20241022"

    results = []

    for batch_size in test_batch_sizes:
        print(f"Testing batch size: {batch_size}")
        print("-" * 40)

        # Create a simple spam generation prompt
        system_prompt = """You are an expert at generating synthetic spam email data for cybersecurity research.
Your task is to generate realistic spam emails based on the provided examples and context."""

        user_prompt = f"""Generate {batch_size} synthetic spam email samples.

Each sample should be a JSON object with the following structure:
{{
    "subject": "email subject",
    "body": "email body content",
    "label": 1
}}

Return the samples in this format:
{{
    "samples": [
        {{"subject": "...", "body": "...", "label": 1}},
        ...
    ]
}}

Generate exactly {batch_size} diverse spam email samples."""

        # Measure generation time
        start_time = time.time()

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                temperature=0.8,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            generation_time = time.time() - start_time

            # Parse response
            content = response.content[0].text

            # Try to parse JSON
            try:
                # Clean markdown code blocks if present
                if content.startswith('```'):
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:].strip()

                parsed = json.loads(content)
                samples = parsed.get('samples', [])
                actual_count = len(samples)
            except:
                actual_count = 0

            # Calculate metrics
            samples_per_second = actual_count / generation_time if generation_time > 0 else 0
            samples_per_minute = samples_per_second * 60
            time_per_sample = generation_time / actual_count if actual_count > 0 else 0

            # Token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            result = {
                'batch_size': batch_size,
                'actual_samples': actual_count,
                'generation_time': generation_time,
                'samples_per_second': samples_per_second,
                'samples_per_minute': samples_per_minute,
                'time_per_sample': time_per_sample,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'tokens_per_sample': total_tokens / actual_count if actual_count > 0 else 0
            }

            results.append(result)

            print(f"✅ Generated: {actual_count} samples")
            print(f"⏱️  Time: {generation_time:.2f}s")
            print(f"🚀 Speed: {samples_per_minute:.2f} samples/minute")
            print(f"📊 Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total")
            print(f"📈 Per sample: {time_per_sample:.2f}s, {total_tokens/actual_count:.0f} tokens")
            print()

            # Small delay between tests
            time.sleep(2)

        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    return results


def extrapolate_full_experiment(results):
    """Extrapolate time and cost for full GPT-4.1-mini equivalent experiment."""

    if not results:
        return

    print()
    print("=" * 80)
    print("Full Experiment Extrapolation (GPT-4.1-mini equivalent)")
    print("=" * 80)
    print()

    # Use the best performing batch size
    best_result = max(results, key=lambda x: x['samples_per_minute'])

    batch_size = best_result['batch_size']
    samples_per_minute = best_result['samples_per_minute']
    tokens_per_sample = best_result['tokens_per_sample']

    print(f"Using batch size: {batch_size}")
    print(f"Measured speed: {samples_per_minute:.2f} samples/minute")
    print()

    # GPT-4.1-mini experiment parameters
    total_samples = 60000  # 3 prompts × 20 groups × 1000 samples

    # Time estimation
    estimated_minutes = total_samples / samples_per_minute
    estimated_hours = estimated_minutes / 60

    print(f"📊 Target: {total_samples:,} samples (3 prompts × 20 groups × 1,000)")
    print()
    print("⏱️  Estimated Time:")
    print(f"   {estimated_minutes:.1f} minutes ({estimated_hours:.2f} hours)")
    print()

    # Cost estimation (Claude 3.5 Haiku pricing)
    input_cost_per_mtok = 0.80
    output_cost_per_mtok = 4.00

    # Estimate input/output token split (rough estimate: 30% input, 70% output)
    avg_input_ratio = best_result['input_tokens'] / best_result['total_tokens']
    avg_output_ratio = best_result['output_tokens'] / best_result['total_tokens']

    total_tokens_estimate = total_samples * tokens_per_sample
    input_tokens_estimate = total_tokens_estimate * avg_input_ratio
    output_tokens_estimate = total_tokens_estimate * avg_output_ratio

    input_cost = (input_tokens_estimate / 1_000_000) * input_cost_per_mtok
    output_cost = (output_tokens_estimate / 1_000_000) * output_cost_per_mtok
    total_cost = input_cost + output_cost

    print("💰 Estimated Cost:")
    print(f"   Input:  {input_tokens_estimate/1_000_000:.2f}M tokens × ${input_cost_per_mtok} = ${input_cost:.2f}")
    print(f"   Output: {output_tokens_estimate/1_000_000:.2f}M tokens × ${output_cost_per_mtok} = ${output_cost:.2f}")
    print(f"   Total:  ${total_cost:.2f}")
    print()

    # Comparison with GPT-4.1-mini
    gpt_cost_estimate = 9.60  # From earlier calculation
    cost_ratio = total_cost / gpt_cost_estimate

    print("📈 Comparison with GPT-4.1-mini:")
    print(f"   GPT-4.1-mini estimated: ${gpt_cost_estimate:.2f}")
    print(f"   Claude 3.5 Haiku:       ${total_cost:.2f}")
    print(f"   Cost ratio:             {cost_ratio:.2f}x")
    print()

    # Parallel execution estimate
    max_workers = 10  # From config
    parallel_time_minutes = estimated_minutes / max_workers
    parallel_time_hours = parallel_time_minutes / 60

    print("⚡ With Parallel Execution (10 workers):")
    print(f"   Estimated time: {parallel_time_minutes:.1f} minutes ({parallel_time_hours:.2f} hours)")
    print()

    print("=" * 80)


if __name__ == "__main__":
    results = test_claude_generation()
    if results:
        extrapolate_full_experiment(results)
        print()
        print("✅ Speed test completed!")
        print()
        print("Next steps:")
        print("1. Review the estimates above")
        print("2. Run small-scale test: python scripts/run_claude_scale_generation.py --scale-count 100")
        print("3. Run full experiment: python scripts/run_claude_scale_generation.py")
    else:
        print()
        print("❌ Test failed. Please fix the API key issue and try again.")
