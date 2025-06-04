# cyberdata/main.py

"""
Main entry point for CyberData synthetic data generation.

This script provides a command-line interface for running the complete pipeline
or individual stages of cybersecurity synthetic data generation.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR))

from cyberdata.stages.stage_1_problems import run_stage_1
from cyberdata.stages.stage_2_seeds import run_stage_2, run_stage_2_for_problem
from cyberdata.crews.cyberdata_crew import CyberDataCrew
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.config_manager import get_config_manager

# Set up logger
logger = setup_logger("cyberdata.main")

# Load environment variables
load_dotenv()


def run_pipeline(stages: list = None) -> dict:
    """
    Run the complete CyberData pipeline or specific stages.
    
    Args:
        stages: List of stage numbers to run (default: [1, 2] for all stages)
        
    Returns:
        dict: Pipeline execution results
    """
    if stages is None:
        stages = [1, 2]
    
    logger.info("="*80)
    logger.info("CYBERDATA SYNTHETIC DATA GENERATION PIPELINE")
    logger.info("Current Configuration:")
    logger.info("- Problems Generated: 3 (1 per operational area)")
    logger.info("- Seed Examples: 10 per problem")
    logger.info("- Model: GPT-4.1-mini")
    logger.info("- Architecture: Simplified CrewAI tools")
    logger.info("="*80)
    logger.info(f"Stages to execute: {stages}")
    
    results = {
        "stages_requested": stages,
        "stages_completed": [],
        "stages_failed": [],
        "overall_status": "not_started"
    }
    
    try:
        # Stage 1: Problem Generation
        if 1 in stages:
            logger.info("\n" + "="*60)
            logger.info("EXECUTING STAGE 1: PROBLEM GENERATION")
            logger.info("="*60)
            
            stage_1_result = run_stage_1()
            results["stage_1"] = stage_1_result
            
            if stage_1_result["status"] == "success":
                results["stages_completed"].append(1)
                logger.info("Stage 1 completed successfully")
            else:
                results["stages_failed"].append(1)
                logger.error("Stage 1 failed")
                
                # If Stage 1 fails and Stage 2 is requested, we might still try Stage 2
                # if there are existing problems from a previous run
                if 2 in stages:
                    logger.warning("Stage 1 failed, but will attempt Stage 2 with existing problems")
        
        # Stage 2: Seed Generation
        if 2 in stages:
            logger.info("\n" + "="*60)
            logger.info("EXECUTING STAGE 2: SEED GENERATION")
            logger.info("="*60)
            
            stage_2_result = run_stage_2()
            results["stage_2"] = stage_2_result
            
            if stage_2_result["status"] in ["success", "partial_success"]:
                results["stages_completed"].append(2)
                logger.info("Stage 2 completed successfully")
            else:
                results["stages_failed"].append(2)
                logger.error("Stage 2 failed")
        
        # Determine overall status
        if len(results["stages_completed"]) == len(stages):
            results["overall_status"] = "success"
        elif len(results["stages_completed"]) > 0:
            results["overall_status"] = "partial_success"
        else:
            results["overall_status"] = "failed"
        
        logger.info(f"\nPipeline completed with status: {results['overall_status']}")
        return results
        
    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        results["overall_status"] = "error"
        results["error"] = error_msg
        return results


def print_pipeline_summary(results: dict):
    """Print a summary of pipeline execution results."""
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Stages Requested: {results['stages_requested']}")
    print(f"Stages Completed: {results['stages_completed']}")
    
    if results['stages_failed']:
        print(f"Stages Failed: {results['stages_failed']}")
    
    # Stage 1 details
    if 'stage_1' in results:
        stage_1 = results['stage_1']
        print(f"\nStage 1 Status: {stage_1['status'].upper()}")
        if stage_1['status'] == 'success':
            print(f"  Problems Generated: {stage_1.get('problems_generated', 'Unknown')}")
            print(f"  Output File: {stage_1.get('output_file', 'Unknown')}")
        elif 'error' in stage_1:
            print(f"  Error: {stage_1['error']}")
    
    # Stage 2 details
    if 'stage_2' in results:
        stage_2 = results['stage_2']
        print(f"\nStage 2 Status: {stage_2['status'].upper()}")
        if stage_2['status'] in ['success', 'partial_success']:
            print(f"  Total Problems: {stage_2.get('total_problems', 'Unknown')}")
            print(f"  Successful: {stage_2.get('successful', 'Unknown')}")
            print(f"  Failed: {stage_2.get('failed', 'Unknown')}")
        elif 'error' in stage_2:
            print(f"  Error: {stage_2['error']}")
    
    if 'error' in results:
        print(f"\nPipeline Error: {results['error']}")
    
    print("="*80)


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="CyberData: CrewAI-powered Synthetic Cybersecurity Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run complete pipeline (Stage 1 + 2)
  %(prog)s --stage 1                 # Run only Stage 1 (problem generation)
  %(prog)s --stage 2                 # Run only Stage 2 (seed generation)
  %(prog)s --stage 1 2               # Run both stages (same as default)
  %(prog)s --problem Enterprise phishing  # Generate seeds for specific problem
        """
    )
    
    parser.add_argument(
        '--stage', 
        type=int, 
        nargs='+', 
        choices=[1, 2],
        default=[1, 2],
        help='Stages to run (1=problems, 2=seeds). Default: 1 2'
    )
    
    parser.add_argument(
        '--problem',
        nargs=2,
        metavar=('AREA', 'NATURE'),
        help='Generate seeds for specific problem (area nature)'
    )
    
    parser.add_argument(
        '--config-check',
        action='store_true',
        help='Check configuration and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration check
    if args.config_check:
        try:
            config_manager = get_config_manager()
            print("Configuration Check:")
            print(f"  Project Root: {config_manager.project_root}")
            print(f"  Config Directory: {config_manager.config_dir}")
            print(f"  Data Directory: {config_manager.data_dir}")
            print(f"  Problems File: {config_manager.problems_file}")
            print(f"  Problems Init File: {config_manager.problems_init_file}")
            
            # Check if problems_init.json exists
            if config_manager.problems_init_file.exists():
                print("  ✓ problems_init.json found")
            else:
                print("  ✗ problems_init.json not found")
                return 1
            
            # Check environment variables
            import os
            if os.getenv("OPENAI_API_KEY"):
                print("  ✓ OPENAI_API_KEY is set")
            else:
                print("  ✗ OPENAI_API_KEY is not set")
                return 1
            
            print("Configuration check passed!")
            return 0
            
        except Exception as e:
            print(f"Configuration check failed: {str(e)}")
            return 1
    
    # Specific problem processing
    if args.problem:
        area, nature = args.problem
        logger.info(f"Generating seeds for specific problem: {area}/{nature}")
        
        result = run_stage_2_for_problem(area, nature)
        
        print(f"\nResult: {result['status'].upper()}")
        print(f"Message: {result['message']}")
        
        if 'examples_generated' in result:
            print(f"Examples Generated: {result['examples_generated']}")
        
        return 0 if result['status'] in ['success', 'partial_success'] else 1
    
    # Run pipeline
    try:
        results = run_pipeline(args.stage)
        print_pipeline_summary(results)
        
        return 0 if results['overall_status'] in ['success', 'partial_success'] else 1
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        logger.error(f"Main execution error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())