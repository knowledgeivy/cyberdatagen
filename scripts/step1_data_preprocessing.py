#!/usr/bin/env python3
"""
Step 1: Data Preprocessing Script
Responsible for loading, cleaning, and grouping raw email data
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.config_manager import load_config
from src.data_processing.preprocessor import EmailDataPreprocessor


def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.logging
    log_file = os.path.join(
        config.output.get('logs_path', './logs/'),
        f"{config.name}_preprocessing.log"
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
    parser = argparse.ArgumentParser(description='Data preprocessing script')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Configuration file path'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name (overrides config file setting)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory (overrides config file setting)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output files exist'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        logger.info("Loading configuration file")
        config = load_config(args.config)

        # Setup logging
        setup_logging(config)

        # Override configuration parameters
        if args.dataset:
            config.dataset = args.dataset

        output_dir = args.output_dir or config.processed_data_path

        logger.info(f"Starting data preprocessing for dataset: {config.dataset}")
        logger.info(f"Output directory: {output_dir}")

        # Check if output files already exist
        dataset_name = config.dataset.lower().replace('-', '')
        output_file = os.path.join(output_dir, f"{dataset_name}_processed.csv.gz")

        if os.path.exists(output_file) and not args.force:
            logger.info(f"Output file already exists: {output_file}")
            logger.info("Use --force parameter to force reprocessing")
            return

        # Create preprocessor
        preprocessor = EmailDataPreprocessor(config)

        # Execute preprocessing
        processed_data = preprocessor.process_dataset(config.dataset, output_dir)

        # Output processing results statistics
        logger.info("=" * 50)
        logger.info("Data preprocessing completed")
        logger.info("=" * 50)

        logger.info(f"Original data size: {processed_data['raw_data_stats']['original_size']}")
        logger.info(f"Cleaned data size: {processed_data['raw_data_stats']['cleaned_size']}")
        logger.info(f"Data size with IDs: {processed_data['raw_data_stats']['with_ids_size']}")
        logger.info(f"Number of groups: {len(processed_data['groups'])}")
        logger.info(f"Test set size: {len(processed_data['test_set'])}")

        # Display per-group statistics
        for group_name, group_data in processed_data['groups'].items():
            spam_count = (group_data['label'] == 1).sum()
            non_spam_count = (group_data['label'] == 0).sum()
            logger.info(f"{group_name}: total={len(group_data)}, spam={spam_count}, non-spam={non_spam_count}")

        # Display saved files
        logger.info("Saved files:")
        for file_type, file_path in processed_data['saved_files'].items():
            logger.info(f"  {file_type}: {file_path}")

        logger.success("Data preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()