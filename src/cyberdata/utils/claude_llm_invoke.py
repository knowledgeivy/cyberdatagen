# cyberdata/utils/claude_llm_invoke.py

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.utils.claude_llm_invoke")

load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Global configuration variables
MODEL_NAME = "claude-3-5-haiku-20241022"
MODEL_TOKEN_SIZE = 8192  # Claude 3.5 Haiku max output tokens
LOG_LEVEL = logging.INFO

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)


def process_llm_request(
    system_prompt,
    user_prompt,
    model_name=MODEL_NAME,
    max_tokens=MODEL_TOKEN_SIZE,
    temperature=0.1,
):
    """
    Process a request to Claude LLM.

    Args:
        system_prompt (str): The system prompt for the LLM
        user_prompt (str): The user prompt for the LLM
        model_name (str): The model to use (default: claude-3-5-haiku-20241022)
        max_tokens (int): Maximum tokens in the response (default: 8192)
        temperature (float): Temperature for the response (default: 0.1)

    Returns:
        str: The response from the LLM
    """
    try:
        # Log the prompts for debugging
        logger.info(f"Calling {model_name} with system prompt: {system_prompt[:100]}...")
        logger.debug(f"Full system prompt: {system_prompt}")
        logger.info(f"User prompt: {user_prompt[:100]}...")
        logger.debug(f"Full user prompt: {user_prompt}")

        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        result = response.content[0].text
        logger.info(f"LLM response received, length: {len(result)} characters")
        logger.debug(f"LLM response: {result[:500]}...")

        return result
    except Exception as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"[Error: {str(e)}]"
