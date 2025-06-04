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
        Analyze the initial cybersecurity problem definitions from problems_init.json and generate 
        an expanded set of problems within the same operational areas (Enterprise, Cloud, EDTC).
        
        For each area in the initial problems, create 3-5 additional problems that:
        1. Use different attack vectors and techniques
        2. Target different assets or systems
        3. Represent realistic threat scenarios from 2023-2025
        4. Follow the same schema structure as the initial problems
        5. Include comprehensive risk reduction strategies
        
        Focus on current and emerging threats such as:
        - AI-generated attacks and deepfakes
        - Supply chain compromises
        - Cloud misconfigurations and container escapes
        - Advanced persistent threats (APTs)
        - Ransomware with lateral movement
        - IoT and edge computing vulnerabilities
        
        Ensure each problem includes:
        - area: Operational environment (Enterprise, Cloud, EDTC)
        - nature: Specific attack type or vulnerability
        - description: Detailed technical scenario
        - risk_reduction: List of 3-4 concrete mitigation strategies
        
        Output the results as a properly formatted JSON structure that can be saved to problems_updated.json.
        """
        
        expected_output = """
        A JSON object with a 'problems' key containing an array of problem definitions.
        Each problem must include: area, nature, description, and risk_reduction fields.
        The output should contain 15-25 total problems across all operational areas.
        Ensure the JSON is valid and properly formatted for file saving.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context
        )