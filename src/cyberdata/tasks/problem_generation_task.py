# cyberdata/tasks/problem_generation_task.py

from crewai import Task
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tasks.problem_generation_task")


class ProblemGenerationTask:
    """
    Task for generating expanded cybersecurity problem definitions.
    
    This task takes initial problem examples and generates a comprehensive
    set of problems within the same operational areas but with different
    attack vectors and techniques.
    """
    
    @staticmethod
    def create_task(agent, context: dict = None) -> Task:
        """
        Create the problem generation task.
        
        Args:
            agent: The agent that will execute this task
            context (dict): Additional context for the task
            
        Returns:
            Task: Configured CrewAI task
        """
        logger.info("Creating ProblemGenerationTask")
        
        description = """
        You are tasked with expanding the initial cybersecurity problem set into a comprehensive collection of threat scenarios.

        **Your Process:**
        1. Use the 'load_initial_problems' tool to get the starting problems from problems_init.json
        2. Use the 'get_problems_prompt' tool to get detailed generation instructions
        3. Use the 'generate_with_llm' tool to create expanded problems following the prompt instructions
        4. Use the 'validate_json' tool to ensure your output is valid JSON
        5. Use the 'save_problems' tool to save the expanded problems to problems_updated.json

        **Requirements:**
        - Generate exactly 3 NEW problems (1 per area: Enterprise, Cloud, EDTC)
        - Each problem must include: area, nature, description, risk_reduction
        - Focus on current and emerging threats (2023-2025)
        - Ensure technical accuracy and realistic scenarios
        - Use different attack vectors from the initial set

        **Output:** Save a comprehensive JSON file with 3 additional cybersecurity problems.
        """
        
        expected_output = """
        A successfully saved problems_updated.json file containing exactly 3 NEW cybersecurity problems.
        Each problem includes area, nature, description, and risk_reduction fields.
        The problems should cover diverse attack vectors and current threat scenarios.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context
        )