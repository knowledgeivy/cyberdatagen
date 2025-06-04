# cyberdata/tools/validation_tool.py

from crewai.tools import BaseTool
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
import json

from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tools.validation_tool")


class ValidateProblemInput(BaseModel):
    """Input schema for problem validation."""
    problem_json: str = Field(..., description="JSON string containing the problem to validate")


class ValidateExampleInput(BaseModel):
    """Input schema for example validation."""
    example_json: str = Field(..., description="JSON string containing the example to validate")


class ValidationTool(BaseTool):
    """
    Tool for data quality and schema validation.
    
    This tool provides validation capabilities for problems and examples
    to ensure they meet the required schema and quality standards.
    """
    
    name: str = "validation_tool"
    description: str = (
        "Validate cybersecurity problems and examples for schema compliance "
        "and data quality. Use this to ensure generated data meets standards."
    )

    def _run(self, operation: str, **kwargs) -> str:
        """
        Execute validation operations.
        
        Args:
            operation: Type of validation to perform
            **kwargs: Operation-specific arguments
            
        Returns:
            str: Validation result
        """
        try:
            if operation == "validate_problem":
                return self._validate_problem(**kwargs)
            elif operation == "validate_example":
                return self._validate_example(**kwargs)
            elif operation == "validate_problems_batch":
                return self._validate_problems_batch(**kwargs)
            else:
                return f"Error: Unknown validation operation '{operation}'"
                
        except Exception as e:
            error_msg = f"Validation error in '{operation}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _validate_problem(self, problem_json: str) -> str:
        """Validate a single problem against the schema."""
        try:
            problem = json.loads(problem_json)
            issues = []
            
            # Required fields
            required_fields = ["area", "nature", "description", "risk_reduction"]
            for field in required_fields:
                if field not in problem:
                    issues.append(f"Missing required field: {field}")
                elif not problem[field]:
                    issues.append(f"Empty required field: {field}")
            
            # Validate area values
            valid_areas = ["Enterprise", "Cloud", "EDTC", "Social Engineering", "Supply Chain", "Endpoint Security"]
            if "area" in problem and problem["area"] not in valid_areas:
                issues.append(f"Invalid area '{problem['area']}'. Must be one of: {valid_areas}")
            
            # Validate risk_reduction is a list
            if "risk_reduction" in problem and not isinstance(problem["risk_reduction"], list):
                issues.append("risk_reduction must be a list")
            
            # Validate description length
            if "description" in problem and len(problem["description"]) < 50:
                issues.append("Description should be at least 50 characters long")
            
            # Check for minimum number of risk reduction strategies
            if "risk_reduction" in problem and isinstance(problem["risk_reduction"], list):
                if len(problem["risk_reduction"]) < 3:
                    issues.append("Should have at least 3 risk reduction strategies")
            
            if issues:
                result = {
                    "valid": False,
                    "issues": issues,
                    "problem": problem
                }
            else:
                result = {
                    "valid": True,
                    "issues": [],
                    "problem": problem
                }
            
            logger.info(f"Problem validation: {'PASS' if result['valid'] else 'FAIL'}")
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "valid": False,
                "issues": [f"Invalid JSON: {str(e)}"],
                "problem": None
            })

    def _validate_example(self, example_json: str) -> str:
        """Validate a single example against the schema."""
        try:
            example = json.loads(example_json)
            issues = []
            
            # Required fields for examples
            required_fields = ["scenario", "technical_data", "indicators", "detection_method"]
            for field in required_fields:
                if field not in example:
                    issues.append(f"Missing required field: {field}")
                elif not example[field]:
                    issues.append(f"Empty required field: {field}")
            
            # Validate indicators is a list
            if "indicators" in example and not isinstance(example["indicators"], list):
                issues.append("indicators must be a list")
            
            # Validate technical_data has sufficient detail
            if "technical_data" in example and len(str(example["technical_data"])) < 100:
                issues.append("technical_data should contain substantial detail (at least 100 characters)")
            
            # Check for MITRE techniques if provided
            if "relevant_mitre_techniques" in example:
                if not isinstance(example["relevant_mitre_techniques"], list):
                    issues.append("relevant_mitre_techniques must be a list")
                else:
                    # Basic validation of MITRE technique format
                    for technique in example["relevant_mitre_techniques"]:
                        if not isinstance(technique, str) or not technique.startswith('T'):
                            issues.append(f"Invalid MITRE technique format: {technique}")
            
            if issues:
                result = {
                    "valid": False,
                    "issues": issues,
                    "example": example
                }
            else:
                result = {
                    "valid": True,
                    "issues": [],
                    "example": example
                }
            
            logger.info(f"Example validation: {'PASS' if result['valid'] else 'FAIL'}")
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "valid": False,
                "issues": [f"Invalid JSON: {str(e)}"],
                "example": None
            })

    def _validate_problems_batch(self, problems_json: str) -> str:
        """Validate a batch of problems."""
        try:
            data = json.loads(problems_json)
            problems = data.get("problems", [])
            
            results = {
                "total_problems": len(problems),
                "valid_problems": 0,
                "invalid_problems": 0,
                "validation_results": []
            }
            
            for i, problem in enumerate(problems):
                problem_json = json.dumps(problem)
                validation_result = json.loads(self._validate_problem(problem_json))
                
                validation_result["index"] = i
                results["validation_results"].append(validation_result)
                
                if validation_result["valid"]:
                    results["valid_problems"] += 1
                else:
                    results["invalid_problems"] += 1
            
            results["validation_rate"] = results["valid_problems"] / results["total_problems"] if results["total_problems"] > 0 else 0
            
            logger.info(f"Batch validation: {results['valid_problems']}/{results['total_problems']} problems valid")
            return json.dumps(results, indent=2)
            
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid JSON: {str(e)}",
                "total_problems": 0,
                "valid_problems": 0,
                "invalid_problems": 0
            })


# Convenience tools for specific validation operations
class ValidateProblemTool(BaseTool):
    """Simplified tool for validating a single problem."""
    
    name: str = "validate_problem"
    description: str = "Validate a cybersecurity problem against the required schema"
    args_schema: Type[BaseModel] = ValidateProblemInput

    def _run(self, problem_json: str) -> str:
        """Validate a problem."""
        validation_tool = ValidationTool()
        return validation_tool._run("validate_problem", problem_json=problem_json)


class ValidateExampleTool(BaseTool):
    """Simplified tool for validating a single example."""
    
    name: str = "validate_example"
    description: str = "Validate a cybersecurity example against the required schema"
    args_schema: Type[BaseModel] = ValidateExampleInput

    def _run(self, example_json: str) -> str:
        """Validate an example."""
        validation_tool = ValidationTool()
        return validation_tool._run("validate_example", example_json=example_json)