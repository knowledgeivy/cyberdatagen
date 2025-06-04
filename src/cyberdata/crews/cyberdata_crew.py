# cyberdata/crews/cyberdata_crew.py

from crewai import Crew, Process
from typing import List, Dict, Any

from cyberdata.agents.taxonomy_agent import TaxonomyAgent
from cyberdata.agents.seed_generator_agent import SeedGeneratorAgent
from cyberdata.tasks.problem_generation_task import ProblemGenerationTask
from cyberdata.tasks.seed_generation_task import SeedGenerationTask
from cyberdata.tools.simple_tools import (
    load_initial_problems, save_problems, save_examples, generate_with_llm,
    validate_json, get_problems_prompt, get_seed_generation_prompt
)
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.crews.cyberdata_crew")


class CyberDataCrew:
    """
    Main crew for orchestrating cybersecurity synthetic data generation.
    
    This crew manages the complete pipeline from initial problem definitions
    to seed example generation using specialized agents and tasks.
    """
    
    def __init__(self):
        """Initialize the CyberData crew."""
        logger.info("Initializing CyberDataCrew")
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize agents
        self.taxonomy_agent = TaxonomyAgent.create_agent(self.tools)
        self.seed_generator_agent = SeedGeneratorAgent.create_agent(self.tools)
        
        logger.info("CyberDataCrew initialized successfully")
    
    def _setup_tools(self) -> List:
        """Set up tools available to agents."""
        logger.info("Setting up tools for agents")
        
        tools = [
            load_initial_problems,
            save_problems,
            save_examples,
            generate_with_llm,
            validate_json,
            get_problems_prompt,
            get_seed_generation_prompt
        ]
        
        logger.info(f"Set up {len(tools)} tools for agents")
        return tools
    
    def generate_problems(self, context: Dict[str, Any] = None) -> str:
        """
        Execute Stage 1: Generate expanded cybersecurity problems.
        
        Args:
            context: Additional context for the task
            
        Returns:
            str: Result of the problem generation
        """
        logger.info("Starting Stage 1: Problem Generation")
        
        try:
            # Create problem generation task
            task = ProblemGenerationTask.create_task(
                agent=self.taxonomy_agent,
                context=context
            )
            
            # Create crew for this stage
            crew = Crew(
                agents=[self.taxonomy_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            # Execute the crew
            logger.info("Executing problem generation crew")
            result = crew.kickoff()
            
            logger.info("Stage 1: Problem Generation completed successfully")
            return str(result)
            
        except Exception as e:
            error_msg = f"Error in problem generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def generate_seeds_for_problem(self, problem: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """
        Execute Stage 2: Generate seed examples for a specific problem.
        
        Args:
            problem: The cybersecurity problem to generate examples for
            context: Additional context for the task
            
        Returns:
            str: Result of the seed generation
        """
        area = problem.get('area', 'Unknown')
        nature = problem.get('nature', 'Unknown')
        logger.info(f"Starting seed generation for {area}/{nature}")
        
        try:
            # Create seed generation task
            task = SeedGenerationTask.create_task(
                agent=self.seed_generator_agent,
                problem=problem,
                context=context
            )
            
            # Create crew for this stage
            crew = Crew(
                agents=[self.seed_generator_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                memory=True
            )
            
            # Execute the crew
            logger.info(f"Executing seed generation crew for {area}/{nature}")
            result = crew.kickoff()
            
            logger.info(f"Seed generation for {area}/{nature} completed successfully")
            return str(result)
            
        except Exception as e:
            error_msg = f"Error in seed generation for {area}/{nature}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def generate_all_seeds(self, problems: List[Dict[str, Any]] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Stage 2: Generate seed examples for all problems.
        
        Args:
            problems: List of problems to generate seeds for (if None, loads from config)
            context: Additional context for the tasks
            
        Returns:
            Dict[str, Any]: Summary of seed generation results
        """
        logger.info("Starting Stage 2: Seed Generation for all problems")
        
        # Load problems if not provided
        if problems is None:
            try:
                from cyberdata.tools.file_manager_tool import LoadProblemsTool
                load_tool = LoadProblemsTool()
                problems_json = load_tool._run(prefer_updated=True)
                import json
                problems_data = json.loads(problems_json)
                problems = problems_data.get("problems", [])
                logger.info(f"Loaded {len(problems)} problems for seed generation")
            except Exception as e:
                error_msg = f"Error loading problems: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg, "results": []}
        
        results = {
            "total_problems": len(problems),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        # Generate seeds for each problem
        for i, problem in enumerate(problems):
            area = problem.get('area', 'Unknown')
            nature = problem.get('nature', 'Unknown')
            
            logger.info(f"Processing problem {i+1}/{len(problems)}: {area}/{nature}")
            
            try:
                result = self.generate_seeds_for_problem(problem, context)
                
                results["results"].append({
                    "problem": {"area": area, "nature": nature},
                    "status": "success",
                    "result": result
                })
                results["successful"] += 1
                
            except Exception as e:
                error_msg = f"Failed to generate seeds for {area}/{nature}: {str(e)}"
                logger.error(error_msg)
                
                results["results"].append({
                    "problem": {"area": area, "nature": nature},
                    "status": "failed",
                    "error": error_msg
                })
                results["failed"] += 1
        
        logger.info(f"Stage 2 completed: {results['successful']}/{results['total_problems']} successful")
        return results
    
    def run_full_pipeline(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the complete pipeline: Stage 1 + Stage 2.
        
        Args:
            context: Additional context for the tasks
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        logger.info("Starting full CyberData pipeline")
        
        pipeline_results = {
            "stage_1": None,
            "stage_2": None,
            "overall_status": "failed"
        }
        
        try:
            # Stage 1: Generate problems
            logger.info("Executing Stage 1: Problem Generation")
            stage_1_result = self.generate_problems(context)
            pipeline_results["stage_1"] = stage_1_result
            
            # Stage 2: Generate seeds for all problems
            logger.info("Executing Stage 2: Seed Generation")
            stage_2_result = self.generate_all_seeds(context=context)
            pipeline_results["stage_2"] = stage_2_result
            
            # Determine overall status
            if (stage_2_result.get("successful", 0) > 0 and 
                stage_2_result.get("failed", 0) < stage_2_result.get("total_problems", 1)):
                pipeline_results["overall_status"] = "success"
            elif stage_2_result.get("successful", 0) > 0:
                pipeline_results["overall_status"] = "partial_success"
            
            logger.info(f"Full pipeline completed with status: {pipeline_results['overall_status']}")
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            pipeline_results["error"] = error_msg
            return pipeline_results