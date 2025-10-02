#!/usr/bin/env python3
"""
Step 3: Dataset Construction Script
Responsible for building training and testing datasets based on different synthetic ratios
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.data_processing.dataset_builder import DatasetBuilder


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_dataset_construction.log"
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', "{time} | {level} | {message}"),
        rotation=log_config.get('rotation', "100 MB"),
        retention=log_config.get('retention', "30 days")
    )


def main():
    parser = argparse.ArgumentParser(description='Dataset construction script')
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
        help='Prompt name'
    )
    parser.add_argument(
        '--llm_engine',
        type=str,
        help='LLM engine name'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='within_group',
        choices=['within_group', 'cross_group', 'real_fixed_random_synthetic', 'full_random'],
        help='Mixing strategy'
    )
    parser.add_argument(
        '--processed_data_dir',
        type=str,
        help='Preprocessed data directory'
    )
    parser.add_argument(
        '--synthetic_data_dir',
        type=str,
        help='Synthetic data directory'
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

        # Set parameters
        llm_engine = args.llm_engine or config.llm_engines[0]
        processed_data_dir = args.processed_data_dir or config.processed_data_path
        synthetic_data_dir = args.synthetic_data_dir or config.synthetic_data_path
        output_dir = args.output_dir or config.datasets_path

        logger.info(f"Starting dataset construction")
        logger.info(f"Dataset: {config.dataset}")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"LLM engine: {llm_engine}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Synthetic ratios: {config.synthetic_ratios}")

        # Create dataset builder
        builder = DatasetBuilder(config)

        # Load preprocessed data
        logger.info("Loading preprocessed data")
        processed_data = builder.load_processed_data(processed_data_dir)

        # Load synthetic data
        logger.info("Loading synthetic data")
        synthetic_data = builder.load_synthetic_data(synthetic_data_dir, args.prompt, llm_engine)

        # Build datasets
        logger.info("Building datasets")
        build_results = builder.build_datasets_for_ratios(
            processed_data,
            synthetic_data,
            args.prompt,
            llm_engine,
            args.strategy,
            output_dir
        )

        # Output construction results statistics
        logger.info("=" * 50)
        logger.info("Dataset construction completed")
        logger.info("=" * 50)

        logger.info(f"Strategy: {build_results['strategy']}")
        logger.info(f"Prompt: {build_results['prompt_name']}")
        logger.info(f"LLM engine: {build_results['llm_engine']}")

        # Statistics of datasets per ratio
        total_datasets = 0
        for ratio_key, ratio_data in build_results['datasets'].items():
            ratio_datasets = sum(len(group_data) for group_data in ratio_data.values())
            total_datasets += ratio_datasets
            ratio = ratio_key.replace('ratio_', '')
            logger.info(f"Ratio {ratio}%: {ratio_datasets} datasets")

        logger.info(f"Total constructed: {total_datasets} training datasets")

        # Display save location
        logger.info(f"Dataset save directory: {os.path.join(output_dir, args.strategy)}")
        logger.info(f"Metadata file: {build_results['metadata_file']}")

        logger.success("Dataset construction completed successfully")

    except Exception as e:
        logger.error(f"Dataset construction failed: {e}")
        raise


if __name__ == "__main__":
    main()