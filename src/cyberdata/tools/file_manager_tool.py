# cyberdata/tools/archive/file_manager_tool.py

"""
ARCHIVED: Complex Pydantic-based file manager tool.

This tool uses advanced Pydantic schemas which caused compatibility issues
with the current CrewAI version. Kept for future reference when Pydantic
compatibility improves.
"""

from crewai.tools import BaseTool
from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tools.archive.file_manager_tool")


class LoadProblemsInput(BaseModel):
    """Input schema for loading problems."""
    prefer_updated: bool = Field(default=True, description="Whether to prefer problems_updated.json over problems.json")


class SaveProblemsInput(BaseModel):
    """Input schema for saving problems."""
    problems: List[Dict[str, Any]] = Field(description="List of problems to save")
    filename: str = Field(default="problems_updated", description="Filename (without .json extension)")


class LoadExamplesInput(BaseModel):
    """Input schema for loading examples."""
    area: str = Field(description="Problem area")
    nature: str = Field(description="Problem nature")


class SaveExamplesInput(BaseModel):
    """Input schema for saving examples."""
    area: str = Field(description="Problem area")
    nature: str = Field(description="Problem nature")
    examples: List[Dict[str, Any]] = Field(description="List of examples to save")


class FileManagerTool(BaseTool):
    """
    Tool for file operations and data management.
    
    This tool provides standardized access to the project's file system
    using the existing ConfigManager infrastructure.
    """
    
    name: str = "file_manager_tool"
    description: str = (
        "Manage files and data operations including loading/saving problems, "
        "examples, and other project data. Uses the existing ConfigManager "
        "for consistent file handling."
    )

    def _run(self, operation: str, **kwargs) -> str:
        """
        Execute file management operations.
        
        Args:
            operation: Type of operation to perform
            **kwargs: Operation-specific arguments
            
        Returns:
            str: Operation result or error message
        """
        try:
            config_manager = get_config_manager()
            
            if operation == "load_problems":
                return self._load_problems(config_manager, **kwargs)
            elif operation == "save_problems":
                return self._save_problems(config_manager, **kwargs)
            elif operation == "load_examples":
                return self._load_examples(config_manager, **kwargs)
            elif operation == "save_examples":
                return self._save_examples(config_manager, **kwargs)
            elif operation == "load_initial_problems":
                return self._load_initial_problems(config_manager)
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            error_msg = f"FileManager error in '{operation}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _load_problems(self, config_manager, prefer_updated: bool = True) -> str:
        """Load problems from configuration files."""
        try:
            problems = config_manager.load_problems(prefer_updated=prefer_updated)
            logger.info(f"Loaded {len(problems)} problems")
            return json.dumps({"problems": problems}, indent=2)
        except Exception as e:
            logger.error(f"Error loading problems: {str(e)}")
            return f"Error loading problems: {str(e)}"

    def _save_problems(self, config_manager, problems: List[Dict[str, Any]], filename: str = "problems_updated") -> str:
        """Save problems to configuration file."""
        try:
            saved_path = config_manager.save_problems(problems, filename)
            logger.info(f"Saved {len(problems)} problems to {saved_path}")
            return f"Successfully saved {len(problems)} problems to {saved_path}"
        except Exception as e:
            logger.error(f"Error saving problems: {str(e)}")
            return f"Error saving problems: {str(e)}"

    def _load_examples(self, config_manager, area: str, nature: str) -> str:
        """Load examples for a specific problem."""
        try:
            # Try to find the examples file
            examples_file = config_manager.find_existing_file(
                config_manager.seeds_dir, 
                nature, 
                "_examples.json"
            )
            
            if not examples_file:
                return f"No examples found for {area}/{nature}"
            
            with examples_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                examples = data.get('examples', [])
            
            logger.info(f"Loaded {len(examples)} examples for {area}/{nature}")
            return json.dumps({"examples": examples}, indent=2)
            
        except Exception as e:
            error_msg = f"Error loading examples for {area}/{nature}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _save_examples(self, config_manager, area: str, nature: str, examples: List[Dict[str, Any]]) -> str:
        """Save examples for a specific problem."""
        try:
            file_path = config_manager.get_seeds_file(area, nature)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save examples
            with file_path.open('w', encoding='utf-8') as f:
                json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(examples)} examples for {area}/{nature} to {file_path}")
            return f"Successfully saved {len(examples)} examples to {file_path}"
            
        except Exception as e:
            error_msg = f"Error saving examples for {area}/{nature}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _load_initial_problems(self, config_manager) -> str:
        """Load initial problems from problems_init.json."""
        try:
            data = config_manager.load_config_file("problems_init")
            problems = data.get("problems", [])
            logger.info(f"Loaded {len(problems)} initial problems")
            return json.dumps({"problems": problems}, indent=2)
        except Exception as e:
            error_msg = f"Error loading initial problems: {str(e)}"
            logger.error(error_msg)
            return error_msg


# Convenience tools for specific operations
class LoadProblemsToolInput(BaseModel):
    """Input for load problems tool."""
    prefer_updated: bool = Field(default=True, description="Prefer updated problems file")


class LoadProblemsTool(BaseTool):
    """Simplified tool for loading problems."""
    
    name: str = "load_problems"
    description: str = "Load cybersecurity problems from configuration files"
    args_schema: Type[BaseModel] = LoadProblemsToolInput

    def _run(self, prefer_updated: bool = True) -> str:
        """Load problems."""
        file_tool = FileManagerTool()
        return file_tool._run("load_problems", prefer_updated=prefer_updated)


class SaveProblemsToolInput(BaseModel):
    """Input for save problems tool."""
    problems_json: str = Field(description="JSON string containing problems to save")
    filename: str = Field(default="problems_updated", description="Output filename")


class SaveProblemsTool(BaseTool):
    """Simplified tool for saving problems."""
    
    name: str = "save_problems"
    description: str = "Save cybersecurity problems to configuration files"
    args_schema: Type[BaseModel] = SaveProblemsToolInput

    def _run(self, problems_json: str, filename: str = "problems_updated") -> str:
        """Save problems."""
        try:
            # Parse JSON string
            data = json.loads(problems_json)
            problems = data.get("problems", [])
            
            file_tool = FileManagerTool()
            return file_tool._run("save_problems", problems=problems, filename=filename)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {str(e)}"
        except Exception as e:
            return f"Error saving problems: {str(e)}"


class SaveExamplesToolInput(BaseModel):
    """Input for save examples tool."""
    area: str = Field(description="Problem area")
    nature: str = Field(description="Problem nature")
    examples_json: str = Field(description="JSON string containing examples to save")


class SaveExamplesTool(BaseTool):
    """Simplified tool for saving examples."""
    
    name: str = "save_examples"
    description: str = "Save seed examples for a specific cybersecurity problem"
    args_schema: Type[BaseModel] = SaveExamplesToolInput

    def _run(self, area: str, nature: str, examples_json: str) -> str:
        """Save examples."""
        try:
            # Parse JSON string
            data = json.loads(examples_json)
            examples = data.get("examples", [])
            
            file_tool = FileManagerTool()
            return file_tool._run("save_examples", area=area, nature=nature, examples=examples)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {str(e)}"
        except Exception as e:
            return f"Error saving examples: {str(e)}"


class LoadInitialProblemsToolInput(BaseModel):
    """Input for load initial problems tool."""
    pass  # No parameters needed


class LoadInitialProblemsTool(BaseTool):
    """Tool for loading initial problems."""
    
    name: str = "load_initial_problems"
    description: str = "Load initial cybersecurity problems from problems_init.json"
    args_schema: Type[BaseModel] = LoadInitialProblemsToolInput

    def _run(self) -> str:
        """Load initial problems."""
        file_tool = FileManagerTool()
        return file_tool._run("load_initial_problems")