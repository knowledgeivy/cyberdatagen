# cyberdata/tasks/seed_generation_task.py

from crewai import Task
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tasks.seed_generation_task")


class SeedGenerationTask:
    """
    Task for generating technical seed examples for cybersecurity problems.
    
    This task creates detailed, realistic examples with technical artifacts
    that security professionals would encounter in real incidents.
    """
    
    @staticmethod
    def create_task(agent, problem: dict, context: dict = None) -> Task:
        """
        Create the seed generation task for a specific problem.
        
        Args:
            agent: The agent that will execute this task
            problem (dict): The cybersecurity problem to generate examples for
            context (dict): Additional context for the task
            
        Returns:
            Task: Configured CrewAI task
        """
        logger.info(f"Creating SeedGenerationTask for {problem.get('area', 'Unknown')}/{problem.get('nature', 'Unknown')}")
        
        area = problem.get('area', 'Unknown')
        nature = problem.get('nature', 'Unknown')
        description = problem.get('description', 'No description provided')
        risk_reduction = ', '.join(problem.get('risk_reduction', []))
        
        task_description = f"""
        You are tasked with generating realistic seed examples for a specific cybersecurity problem.

        **Problem Details:**
        - Area: {area}
        - Nature: {nature}
        - Description: {description}
        - Risk Reduction: {risk_reduction}

        **Your Process:**
        1. Use the 'get_seed_generation_prompt' tool with the problem details to get detailed generation instructions
        2. Use the 'generate_with_llm' tool to create exactly 10 highly detailed technical examples following the prompt
        3. Use the 'validate_json' tool to ensure your output is valid JSON
        4. Use the 'save_examples' tool to save the examples with area="{area}" and nature="{nature}"

        **Requirements:**
        - Generate exactly 10 detailed examples with realistic technical artifacts
        - Include scenario, technical_data, indicators, detection_method, and relevant_mitre_techniques
        - Ensure all technical details are accurate and realistic
        - Examples should be indistinguishable from real-world security incidents

        **Output:** Successfully saved seed examples for the specified problem.
        """

        expected_output = f"""
        Successfully saved seed examples file for {area}/{nature} containing exactly 10 detailed examples.
        Each example includes realistic technical data, indicators of compromise, detection methods,
        and relevant MITRE ATT&CK techniques. The examples are technically accurate and could be
        used for training security analysts or testing detection systems.
        """
        
        return Task(
            description=task_description,
            expected_output=expected_output,
            agent=agent,
            context=context
        )