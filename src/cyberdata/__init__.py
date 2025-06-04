# cyberdata/__init__.py

"""
CyberData: CrewAI-powered Synthetic Cybersecurity Data Generation

A modular Python toolkit for generating and validating synthetic cybersecurity datasets 
using CrewAI agents and tasks.
"""

# Import main components for easy access
from cyberdata.crews.cyberdata_crew import CyberDataCrew
from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.logger_config import setup_logger

__all__ = [
    "CyberDataCrew",
    "get_config_manager", 
    "setup_logger"
]