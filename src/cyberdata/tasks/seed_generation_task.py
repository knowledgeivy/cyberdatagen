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
        risk_reduction = problem.get('risk_reduction', [])
        
        task_description = f"""
        Generate 2-3 highly detailed, technically accurate seed examples for the following cybersecurity problem:

        **Problem Details:**
        - Area: {area}
        - Nature: {nature}
        - Description: {description}
        - Risk Reduction: {', '.join(risk_reduction)}

        For each example, create realistic technical data that includes:

        **For network-based attacks:**
        - Raw HTTP request/response data with headers and payloads
        - Network packet captures (text format similar to Wireshark output)
        - Log entries from web servers, WAFs, or IDS/IPS systems
        - Actual exploit code or injection strings

        **For phishing and social engineering:**
        - Complete email content with realistic headers (including X-headers)
        - SMTP transaction logs
        - Domain registration details for suspicious domains
        - URL structures with obfuscation techniques

        **For malware and system compromise:**
        - File hashes (MD5, SHA-1, SHA-256)
        - Registry changes or file system artifacts
        - Memory dump analysis snippets
        - Command-and-control traffic patterns
        - Process creation and execution chains

        **For cloud security issues:**
        - API call sequences demonstrating the attack
        - IAM policy definitions showing misconfigurations
        - CloudTrail or equivalent logs showing suspicious activity
        - Container escape proof-of-concept details

        Each example must include:
        1. **scenario**: Brief description of this specific attack instance
        2. **technical_data**: Detailed technical information as described above
        3. **indicators**: Specific technical indicators of compromise
        4. **detection_method**: How this would be detected in practice
        5. **relevant_mitre_techniques**: MITRE ATT&CK techniques relevant to this example
        """

        expected_output = f"""
        A JSON object with an 'examples' key containing an array of 2-3 detailed examples.
        Each example must be technically accurate and contain realistic data artifacts
        that a security professional would encounter during an actual security incident.
        
        The examples should be for: {area} / {nature}
        
        Structure each example as:
        {{
            "scenario": "Brief description of the attack instance",
            "technical_data": "Detailed technical information with realistic artifacts",
            "indicators": ["List", "of", "specific", "IoCs"],
            "detection_method": "How this would be detected",
            "relevant_mitre_techniques": ["T1234", "T5678"]
        }}
        
        Ensure the JSON is valid and ready for file saving.
        """
        
        return Task(
            description=task_description,
            expected_output=expected_output,
            agent=agent,
            context=context
        )