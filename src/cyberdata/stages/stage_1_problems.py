# cyberdata/stages/stage_1_problems.py

"""
Stage 1: Problem Generation

This stage uses the TaxonomyAgent to generate expanded cybersecurity problem 
definitions based on initial examples from problems_init.json.
"""

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directories to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))  # Add cyberdata package to path

from cyberdata.crews.cyberdata_crew import CyberDataCrew
from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.config_manager import get_config_manager

# Set up logger
logger = setup_logger("cyberdata.stages.stage_1_problems")

# Load environment variables
load_dotenv()


def extract_json_from_result(result_text: str) -> dict:
    """
    Extract JSON from CrewAI result text.
    
    Args:
        result_text: Raw result text from CrewAI
        
    Returns:
        dict: Extracted JSON data
    """
    try:
        # Try direct JSON parsing
        return json.loads(result_text)
    except json.JSONDecodeError:
        # Try to find JSON in the text
        import re
        
        # Look for JSON object pattern
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, result_text)
        
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If no JSON found, return error structure
        logger.warning("Could not extract JSON from result")
        return {"problems": [], "error": "Could not extract JSON from result"}


def validate_and_save_problems(problems_data: dict) -> bool:
    """
    Validate and save the generated problems.
    
    Args:
        problems_data: Dictionary containing problems
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config_manager = get_config_manager()
        problems = problems_data.get("problems", [])
        
        if not problems:
            logger.error("No problems found in generated data")
            return False
        
        # Basic validation
        valid_problems = []
        for problem in problems:
            required_fields = ["area", "nature", "description", "risk_reduction"]
            if all(field in problem for field in required_fields):
                valid_problems.append(problem)
            else:
                logger.warning(f"Invalid problem missing required fields: {problem.get('nature', 'Unknown')}")
        
        if not valid_problems:
            logger.error("No valid problems found after validation")
            return False
        
        # Save to problems_updated.json
        config_manager.save_problems(valid_problems, "problems_updated")
        logger.info(f"Successfully saved {len(valid_problems)} problems to problems_updated.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating and saving problems: {str(e)}", exc_info=True)
        return False


def run_stage_1(context: dict = None) -> dict:
    """
    Execute Stage 1: Problem Generation.
    
    Args:
        context: Additional context for the task
        
    Returns:
        dict: Stage execution results
    """
    logger.info("="*60)
    logger.info("STAGE 1: CYBERSECURITY PROBLEM GENERATION")
    logger.info("="*60)
    
    try:
        # Initialize the crew
        logger.info("Initializing CyberDataCrew...")
        crew = CyberDataCrew()
        
        # Execute problem generation
        logger.info("Executing problem generation task...")
        result = crew.generate_problems(context)
        
        # Extract and validate results
        logger.info("Processing results...")
        problems_data = extract_json_from_result(result)
        
        # Save results
        success = validate_and_save_problems(problems_data)
        
        if success:
            problems_count = len(problems_data.get("problems", []))
            logger.info(f"Stage 1 completed successfully: Generated {problems_count} problems")
            
            return {
                "status": "success",
                "problems_generated": problems_count,
                "message": f"Successfully generated {problems_count} cybersecurity problems",
                "output_file": "problems_updated.json"
            }
        else:
            logger.error("Stage 1 failed: Could not save valid problems")
            return {
                "status": "failed",
                "message": "Could not save valid problems",
                "raw_result": str(result)
            }
    
    except Exception as e:
        error_msg = f"Stage 1 failed with error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "error": str(e)
        }


def main():
    """Main function for standalone execution."""
    logger.info("Starting Stage 1: Problem Generation")
    
    result = run_stage_1()
    
    # Print results
    print("\n" + "="*60)
    print("STAGE 1 RESULTS")
    print("="*60)
    print(f"Status: {result['status'].upper()}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'success':
        print(f"Problems Generated: {result['problems_generated']}")
        print(f"Output File: {result['output_file']}")
    elif 'error' in result:
        print(f"Error: {result['error']}")
    
    print("="*60)
    
    return result['status'] == 'success'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)