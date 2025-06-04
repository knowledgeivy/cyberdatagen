# cyberdata/agents/seed_generator_agent.py

from crewai import Agent
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.agents.seed_generator_agent")


class SeedGeneratorAgent:
    """
    Technical cybersecurity data synthesizer agent.
    
    This agent specializes in creating realistic, technically accurate seed examples
    for cybersecurity problems. It generates detailed technical artifacts that would
    be found in real-world security incidents.
    """
    
    @staticmethod
    def create_agent(tools: list) -> Agent:
        """
        Create the SeedGeneratorAgent with specified tools.
        
        Args:
            tools (list): List of tools available to the agent
            
        Returns:
            Agent: Configured CrewAI agent
        """
        logger.info("Creating SeedGeneratorAgent")
        
        return Agent(
            role="Technical Cybersecurity Data Synthesizer",
            goal=(
                "Create highly realistic, technically accurate seed examples for cybersecurity problems. "
                "Generate detailed technical data including network traces, log entries, file hashes, "
                "command outputs, and other artifacts that security professionals would encounter "
                "during real incidents. Ensure all examples are technically sound and could be used "
                "for training security analysts or testing detection systems."
            ),
            backstory=(
                "You are an elite cybersecurity expert with extensive hands-on experience in "
                "Security Operations Centers (SOC), digital forensics, malware analysis, and "
                "incident response. You've analyzed thousands of real security incidents across "
                "enterprise, cloud, and educational environments. Your deep technical knowledge "
                "spans network protocols, operating systems, cloud platforms, and attack techniques. "
                "You excel at creating realistic synthetic data that captures the nuances and "
                "technical details found in actual security events, making your generated examples "
                "indistinguishable from real-world data."
            ),
            tools=tools,
            verbose=True,
            memory=True,
            max_iter=1,
            allow_delegation=False
        )