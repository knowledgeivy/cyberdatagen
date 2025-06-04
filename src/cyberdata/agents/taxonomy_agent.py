# cyberdata/agents/taxonomy_agent.py

from crewai import Agent
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.agents.taxonomy_agent")


class TaxonomyAgent:
    """
    Cybersecurity taxonomy expert and problem generator agent.
    
    This agent specializes in analyzing existing cybersecurity problem definitions
    and generating new, comprehensive problems within the same operational areas
    but with different attack vectors and techniques.
    """
    
    @staticmethod
    def create_agent(tools: list) -> Agent:
        """
        Create the TaxonomyAgent with specified tools.
        
        Args:
            tools (list): List of tools available to the agent
            
        Returns:
            Agent: Configured CrewAI agent
        """
        logger.info("Creating TaxonomyAgent")
        
        return Agent(
            role="Cybersecurity Taxonomy Expert and Problem Generator",
            goal=(
                "Generate comprehensive cybersecurity problem definitions based on initial examples. "
                "Create new problems within the same operational areas but with different attack vectors, "
                "ensuring consistency with industry frameworks like MITRE ATT&CK and maintaining "
                "technical accuracy across all problem definitions."
            ),
            backstory=(
                "You are an elite cybersecurity analyst with over 15 years of experience in threat "
                "intelligence, incident response, and security architecture. You have deep expertise "
                "in cybersecurity frameworks including MITRE ATT&CK, NIST Cybersecurity Framework, "
                "and OWASP. You've worked across enterprise, cloud, and educational environments, "
                "giving you comprehensive knowledge of attack vectors and defense strategies. "
                "Your specialty is creating structured, actionable threat taxonomies that help "
                "organizations understand and prepare for evolving cyber threats."
            ),
            tools=tools,
            verbose=True,
            memory=True,
            max_iter=1,
            allow_delegation=False
        )