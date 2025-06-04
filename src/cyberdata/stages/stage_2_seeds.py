# cyberdata/stages/stage_2_seeds.py

"""
Stage 2: Seed Generation

This stage uses the SeedGeneratorAgent to create detailed technical seed examples
for each cybersecurity problem generated in Stage 1.
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
logger = setup_logger("cyberdata.stages.stage_2_seeds")

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
        return {"examples": [], "error": "Could not extract JSON from result"}


def validate_and_save_examples(problem: dict, examples_data: dict) -> bool:
    """
    Validate and save the generated examples for a specific problem.
    
    Args:
        problem: The problem dictionary
        examples_data: Dictionary containing examples
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        config_manager = get_config_manager()
        examples = examples_data.get("examples", [])
        
        if not examples:
            logger.error(f"No examples found for {problem['area']}/{problem['nature']}")
            return False
        
        # Basic validation
        valid_examples = []
        for example in examples:
            required_fields = ["scenario", "technical_data", "indicators", "detection_method"]
            if all(field in example for field in required_fields):
                valid_examples.append(example)
            else:
                logger.warning(f"Invalid example missing required fields for {problem['nature']}")
        
        if not valid_examples:
            logger.error(f"No valid examples found for {problem['area']}/{problem['nature']}")
            return False
        
        # Save examples using config manager
        file_path = config_manager.get_seeds_file(problem['area'], problem['nature'])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with file_path.open('w', encoding='utf-8') as f:
            json.dump({"examples": valid_examples}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(valid_examples)} examples for {problem['area']}/{problem['nature']}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving examples for {problem['area']}/{problem['nature']}: {str(e)}", exc_info=True)
        return False


def load_problems() -> list:
    """
    Load problems from configuration files.
    
    Returns:
        list: List of problems to generate seeds for
    """
    try:
        config_manager = get_config_manager()
        problems = config_manager.load_problems(prefer_updated=True)
        logger.info(f"Loaded {len(problems)} problems for seed generation")
        return problems
    except Exception as e:
        logger.error(f"Error loading problems: {str(e)}", exc_info=True)
        return []


def run_stage_2(problems: list = None, context: dict = None) -> dict:
    """
    Execute Stage 2: Seed Generation.
    
    Args:
        problems: List of problems to generate seeds for (if None, loads from config)
        context: Additional context for the tasks
        
    Returns:
        dict: Stage execution results
    """
    logger.info("="*60)
    logger.info("STAGE 2: SEED EXAMPLE GENERATION")
    logger.info("="*60)
    
    try:
        # Load problems if not provided
        if problems is None:
            problems = load_problems()
        
        if not problems:
            logger.error("No problems found for seed generation")
            return {
                "status": "failed",
                "message": "No problems found for seed generation"
            }
        
        # Initialize the crew
        logger.info("Initializing CyberDataCrew...")
        crew = CyberDataCrew()
        
        # Track results
        results = {
            "total_problems": len(problems),
            "successful": 0,
            "failed": 0,
            "problem_results": []
        }
        
        # Generate seeds for each problem
        for i, problem in enumerate(problems):
            area = problem.get('area', 'Unknown')
            nature = problem.get('nature', 'Unknown')
            
            logger.info(f"Processing problem {i+1}/{len(problems)}: {area}/{nature}")
            
            try:
                # Execute seed generation for this problem
                result = crew.generate_seeds_for_problem(problem, context)
                
                # Extract and validate results
                examples_data = extract_json_from_result(result)
                
                # Save examples
                success = validate_and_save_examples(problem, examples_data)
                
                if success:
                    examples_count = len(examples_data.get("examples", []))
                    results["problem_results"].append({
                        "problem": {"area": area, "nature": nature},
                        "status": "success",
                        "examples_generated": examples_count
                    })
                    results["successful"] += 1
                    logger.info(f"Successfully generated {examples_count} examples for {area}/{nature}")
                else:
                    results["problem_results"].append({
                        "problem": {"area": area, "nature": nature},
                        "status": "failed",
                        "error": "Could not save valid examples"
                    })
                    results["failed"] += 1
                
            except Exception as e:
                error_msg = f"Error generating seeds for {area}/{nature}: {str(e)}"
                logger.error(error_msg)
                
                results["problem_results"].append({
                    "problem": {"area": area, "nature": nature},
                    "status": "error",
                    "error": error_msg
                })
                results["failed"] += 1
        
        # Determine overall status
        if results["successful"] > 0 and results["failed"] == 0:
            status = "success"
            message = f"Successfully generated seeds for all {results['successful']} problems"
        elif results["successful"] > 0:
            status = "partial_success"
            message = f"Generated seeds for {results['successful']}/{results['total_problems']} problems"
        else:
            status = "failed"
            message = "Failed to generate seeds for any problems"
        
        results["status"] = status
        results["message"] = message
        
        logger.info(f"Stage 2 completed: {message}")
        return results
        
    except Exception as e:
        error_msg = f"Stage 2 failed with error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "error": str(e)
        }


def run_stage_2_for_problem(problem_area: str, problem_nature: str, context: dict = None) -> dict:
    """
    Execute Stage 2 for a specific problem only.
    
    Args:
        problem_area: Area of the problem
        problem_nature: Nature of the problem
        context: Additional context for the task
        
    Returns:
        dict: Stage execution results
    """
    logger.info(f"Starting Stage 2 for specific problem: {problem_area}/{problem_nature}")
    
    try:
        # Load all problems and find the specific one
        problems = load_problems()
        target_problem = None
        
        for problem in problems:
            if (problem.get('area') == problem_area and 
                problem.get('nature') == problem_nature):
                target_problem = problem
                break
        
        if not target_problem:
            return {
                "status": "failed",
                "message": f"Problem {problem_area}/{problem_nature} not found"
            }
        
        # Run stage 2 for just this problem
        result = run_stage_2([target_problem], context)
        
        # Adjust result format for single problem
        if result["status"] != "error":
            problem_result = result["problem_results"][0] if result["problem_results"] else {}
            result["examples_generated"] = problem_result.get("examples_generated", 0)
        
        return result
        
    except Exception as e:
        error_msg = f"Error running Stage 2 for {problem_area}/{problem_nature}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "error": str(e)
        }


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate seed examples for cybersecurity problems")
    parser.add_argument('--area', help='Specific problem area to process')
    parser.add_argument('--nature', help='Specific problem nature to process')
    
    args = parser.parse_args()
    
    logger.info("Starting Stage 2: Seed Generation")
    
    # Run for specific problem or all problems
    if args.area and args.nature:
        result = run_stage_2_for_problem(args.area, args.nature)
    else:
        result = run_stage_2()
    
    # Print results
    print("\n" + "="*60)
    print("STAGE 2 RESULTS")
    print("="*60)
    print(f"Status: {result['status'].upper()}")
    print(f"Message: {result['message']}")
    
    if result['status'] in ['success', 'partial_success']:
        if 'total_problems' in result:
            print(f"Total Problems: {result['total_problems']}")
            print(f"Successful: {result['successful']}")
            print(f"Failed: {result['failed']}")
        elif 'examples_generated' in result:
            print(f"Examples Generated: {result['examples_generated']}")
    elif 'error' in result:
        print(f"Error: {result['error']}")
    
    print("="*60)
    
    return result['status'] in ['success', 'partial_success']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)