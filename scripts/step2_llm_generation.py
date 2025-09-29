#!/usr/bin/env python3
"""
Step 2: LLM Generation Script
Responsible for using LLM to generate synthetic spam email data
"""

import argparse
import sys
import os
import asyncio
import json
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.llm_generation.generator import SyntheticDataGenerator
from src.data_processing.preprocessor import EmailDataPreprocessor


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_generation.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def load_processed_groups(processed_data_dir: str, dataset_name: str) -> dict:
    """Load preprocessed grouped data"""
    logger.info("Loading preprocessed grouped data")

    # Load metadata
    metadata_file = os.path.join(processed_data_dir, f"{dataset_name}_metadata.json")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load grouped data
    groups = {}
    groups_dir = os.path.join(processed_data_dir, 'groups')

    for group_name in metadata['groups'].keys():
        group_file = os.path.join(groups_dir, f"{dataset_name}_{group_name}.csv.gz")
        if os.path.exists(group_file):
            import pandas as pd
            groups[group_name] = pd.read_csv(group_file, compression='gzip')
        else:
            logger.warning(f"Group file does not exist: {group_file}")

    logger.info(f"Loading completed: {len(groups)} groups")
    return groups


async def main():
    parser = argparse.ArgumentParser(description='LLM generation script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Prompt name (original, strong, weak)'
    )
    parser.add_argument(
        '--llm_engine',
        type=str,
        help='LLM engine name (overrides config file setting)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size (overrides config file setting)'
    )
    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        help='Checkpoint saving interval'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume generation from checkpoint'
    )
    parser.add_argument(
        '--processed_data_dir',
        type=str,
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration file")
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)

        # Override configuration parameters
        if args.llm_engine:
            config.llm_engines = [args.llm_engine]
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.save_checkpoint_every:
            config.save_checkpoint_every = args.save_checkpoint_every

        llm_engine = config.llm_engines[0]  # Use the first LLM engine
        processed_data_dir = args.processed_data_dir or config.processed_data_path
        output_dir = args.output_dir or config.synthetic_data_path

        logger.info(f"Starting synthetic data generation")
        logger.info(f"Dataset: {config.dataset}")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"LLM engine: {llm_engine}")
        logger.info(f"Batch size: {config.batch_size}")

        # Validate prompt
        if args.prompt not in config.prompts:
            logger.error(f"Prompt not found: {args.prompt}")
            logger.error(f"Available prompts: {list(config.prompts.keys())}")
            return

        # Check if output file already exists
        dataset_name = config.dataset.lower().replace('-', '')
        output_file = os.path.join(
            output_dir,
            f"{dataset_name}_synthetic_{args.prompt}_{llm_engine}.csv.gz"
        )

        if os.path.exists(output_file) and not args.resume:
            logger.info(f"Output file already exists: {output_file}")
            logger.info("Use --resume parameter to resume from checkpoint")
            return

        # Load preprocessed data
        groups_data = load_processed_groups(processed_data_dir, dataset_name)

        if not groups_data:
            logger.error("Preprocessed data not found")
            return

        # Create generator
        generator = SyntheticDataGenerator(config)

        # Execute generation
        logger.info("Starting synthetic data generation")
        generated_files = await generator.generate_complete_dataset(
            groups_data,
            args.prompt,
            llm_engine,
            output_dir
        )

        # Output generation results statistics
        logger.info("=" * 50)
        logger.info("Synthetic data generation completed")
        logger.info("=" * 50)

        if generated_files:
            logger.info("Generated files:")
            for file_type, file_path in generated_files.items():
                logger.info(f"  {file_type}: {file_path}")

            # Statistics of generated data
            if 'complete' in generated_files:
                import pandas as pd
                complete_data = pd.read_csv(generated_files['complete'], compression='gzip')
                logger.info(f"Total generated: {len(complete_data)} synthetic data samples")

                # Statistics by group
                if 'group_id' in complete_data.columns:
                    group_counts = complete_data.groupby('group_id').size()
                    logger.info("Statistics by group:")
                    for group_id, count in group_counts.items():
                        logger.info(f"  Group {group_id}: {count} samples")

            logger.success("Synthetic data generation completed successfully")
        else:
            logger.error("Failed to generate any synthetic data")

    except Exception as e:
        logger.error(f"Synthetic data generation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())