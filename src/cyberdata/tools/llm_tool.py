# cyberdata/tools/llm_tool.py

from crewai.tools import BaseTool
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field
import json

from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.prompt_loader import load_system_prompt, load_user_prompt
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tools.llm_tool")


class LLMToolInput(BaseModel):
    """Input schema for LLM Tool."""
    prompt_file: str = Field(description="Name of the YAML prompt file to use")
    prompt_type: str = Field(description="Type of prompt: 'system' or 'user'")
    prompt_name: Optional[str] = Field(default=None, description="Specific prompt name if file has multiple prompts")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variables to substitute in the prompt")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration (temperature, max_tokens, etc.)")


class LLMTool(BaseTool):
    """
    Tool for standardized LLM interaction using existing prompt infrastructure.
    
    This tool leverages the existing prompt loading system and LLM invocation
    to provide consistent LLM interactions for CrewAI agents.
    """
    
    name: str = "llm_tool"
    description: str = (
        "Invoke LLM with prompts from YAML files. Use this tool when you need to "
        "generate content using the existing prompt templates. Supports variable "
        "substitution and model configuration."
    )
    args_schema: Type[BaseModel] = LLMToolInput

    def _run(self, 
             prompt_file: str, 
             prompt_type: str, 
             prompt_name: Optional[str] = None,
             variables: Dict[str, Any] = None,
             model_config: Dict[str, Any] = None) -> str:
        """
        Execute LLM request using existing prompt infrastructure.
        
        Args:
            prompt_file: Name of the YAML prompt file
            prompt_type: 'system' or 'user' prompt type
            prompt_name: Specific prompt name if file has multiple prompts
            variables: Variables to substitute in prompts
            model_config: Model configuration overrides
            
        Returns:
            str: LLM response
        """
        try:
            logger.info(f"LLM Tool invoked: {prompt_file} / {prompt_type}")
            logger.debug(f"Variables: {variables}")
            
            # Handle None values
            if variables is None:
                variables = {}
            if model_config is None:
                model_config = {}
            
            # Default model configuration
            default_config = {
                "model_name": "gpt-4.1-mini",
                "temperature": 0.7,
                "max_tokens": 16384
            }
            default_config.update(model_config)
            
            # Load prompts based on type
            if prompt_type.lower() == "system":
                prompt_content = load_system_prompt(
                    prompt_file, 
                    prompt_name=prompt_name,
                    **variables
                )
                # For system prompts, create a simple user prompt
                user_prompt = "Please complete the task as described in the system prompt."
                
            elif prompt_type.lower() == "user":
                # For user prompts, we need both system and user
                system_content = load_system_prompt(
                    prompt_file,
                    prompt_name=prompt_name,
                    **variables
                )
                user_prompt = load_user_prompt(
                    prompt_file,
                    prompt_name=prompt_name, 
                    **variables
                )
                prompt_content = system_content
                
            else:
                raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be 'system' or 'user'")
            
            # Invoke LLM
            response = process_llm_request(
                system_prompt=prompt_content,
                user_prompt=user_prompt,
                **default_config
            )
            
            logger.info(f"LLM Tool response received, length: {len(response)} characters")
            return response
            
        except Exception as e:
            error_msg = f"LLM Tool error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"


class DirectLLMToolInput(BaseModel):
    """Input schema for Direct LLM Tool."""
    system_prompt: str = Field(description="System prompt content")
    user_prompt: str = Field(description="User prompt content")
    model_config: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")


class DirectLLMTool(BaseTool):
    """
    Tool for direct LLM interaction without prompt files.
    
    Use this when you need to make direct LLM calls with custom prompts
    not stored in YAML files.
    """
    
    name: str = "direct_llm_tool"
    description: str = (
        "Make direct LLM calls with custom system and user prompts. Use this when "
        "you need to generate content with prompts not stored in YAML files."
    )
    args_schema: Type[BaseModel] = DirectLLMToolInput

    def _run(self, 
             system_prompt: str, 
             user_prompt: str,
             model_config: Dict[str, Any] = None) -> str:
        """
        Execute direct LLM request.
        
        Args:
            system_prompt: System prompt content
            user_prompt: User prompt content
            model_config: Model configuration overrides
            
        Returns:
            str: LLM response
        """
        try:
            logger.info("Direct LLM Tool invoked")
            
            # Handle None values
            if model_config is None:
                model_config = {}
            
            # Default model configuration
            default_config = {
                "model_name": "gpt-4.1-mini",
                "temperature": 0.7,
                "max_tokens": 16384
            }
            default_config.update(model_config)
            
            # Invoke LLM
            response = process_llm_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **default_config
            )
            
            logger.info(f"Direct LLM Tool response received, length: {len(response)} characters")
            return response
            
        except Exception as e:
            error_msg = f"Direct LLM Tool error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"