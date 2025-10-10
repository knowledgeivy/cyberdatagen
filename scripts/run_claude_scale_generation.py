#!/usr/bin/env python3
"""
Parallel execution script for Claude 3.5 Haiku scale generation.
This script runs the Claude scale generation with configurable parameters.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberdata.process.claude_scale_generation import main

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Claude 3.5 Haiku scale generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config settings
  python scripts/run_claude_scale_generation.py

  # Run with custom parameters
  python scripts/run_claude_scale_generation.py --scale-count 1000 --malicious-ratio 0.5

  # Run with specific problems
  python scripts/run_claude_scale_generation.py --problems phishing malware

  # Run with custom workers and batch size
  python scripts/run_claude_scale_generation.py --max-workers 8 --batch-size 10

  # Full custom run
  python scripts/run_claude_scale_generation.py \\
      --scale-count 2000 \\
      --malicious-ratio 0.6 \\
      --max-workers 10 \\
      --batch-size 5 \\
      --problems spam phishing
        """
    )

    parser.add_argument(
        '--scale-count',
        type=int,
        help='Number of samples to generate per problem (default: from config)'
    )

    parser.add_argument(
        '--malicious-ratio',
        type=float,
        help='Ratio of malicious samples, e.g., 0.5 for 50%% (default: from config)'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        help='Maximum number of parallel workers (default: from config, recommended: 8-10 for Claude)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for each generation task (default: from config, recommended: 5 for Claude)'
    )

    parser.add_argument(
        '--problems',
        nargs='+',
        help='Specific problem natures to generate (e.g., phishing spam malware)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Claude 3.5 Haiku Scale Generation")
    print("=" * 80)
    print(f"Model: claude-3-5-haiku-20241022")
    print(f"Configuration: config/claude_scale_config.yaml")
    print()

    if args.scale_count:
        print(f"Scale count: {args.scale_count}")
    if args.malicious_ratio:
        print(f"Malicious ratio: {args.malicious_ratio:.1%}")
    if args.max_workers:
        print(f"Max workers: {args.max_workers}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    if args.problems:
        print(f"Specific problems: {', '.join(args.problems)}")

    print()
    print("Starting generation...")
    print()

    # Run the main generation function
    try:
        main(
            scale_count=args.scale_count,
            malicious_ratio=args.malicious_ratio,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            problems=args.problems
        )
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during generation: {e}")
        sys.exit(1)
