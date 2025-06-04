# cyberdata/tools/simple_tools.py

"""
Simplified tools for CrewAI that avoid complex Pydantic schema issues.
These tools provide basic functionality for the agents.
"""

from crewai.tools import tool
import json
from typing import Dict, Any, List

from cyberdata.utils.config_manager import get_config_manager
from cyberdata.utils.llm_invoke import process_llm_request
from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.tools.simple_tools")


@tool("load_initial_problems")
def load_initial_problems() -> str:
    """Load initial cybersecurity problems from problems_init.json file."""
    try:
        config_manager = get_config_manager()
        data = config_manager.load_config_file("problems_init")
        problems = data.get("problems", [])
        logger.info(f"Loaded {len(problems)} initial problems")
        return json.dumps({"problems": problems}, indent=2)
    except Exception as e:
        error_msg = f"Error loading initial problems: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool("save_problems")
def save_problems(problems_json: str, filename: str = "problems_updated") -> str:
    """Save cybersecurity problems to a configuration file."""
    try:
        # Parse JSON string
        data = json.loads(problems_json)
        problems = data.get("problems", [])
        
        config_manager = get_config_manager()
        saved_path = config_manager.save_problems(problems, filename)
        logger.info(f"Saved {len(problems)} problems to {saved_path}")
        return f"Successfully saved {len(problems)} problems to {saved_path}"
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    except Exception as e:
        return f"Error saving problems: {str(e)}"


@tool("save_examples")
def save_examples(area: str, nature: str, examples_json: str) -> str:
    """Save seed examples for a specific cybersecurity problem."""
    try:
        # Parse JSON string
        data = json.loads(examples_json)
        examples = data.get("examples", [])
        
        config_manager = get_config_manager()
        file_path = config_manager.get_seeds_file(area, nature)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save examples
        with file_path.open('w', encoding='utf-8') as f:
            json.dump({"examples": examples}, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(examples)} examples for {area}/{nature} to {file_path}")
        return f"Successfully saved {len(examples)} examples to {file_path}"
        
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: {str(e)}"
    except Exception as e:
        error_msg = f"Error saving examples for {area}/{nature}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool("generate_with_llm")
def generate_with_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Generate content using LLM with custom prompts."""
    try:
        logger.info("Generating content with LLM")
        response = process_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name="gpt-4.1-mini",
            temperature=temperature,
            max_tokens=16384
        )
        logger.info(f"LLM response received, length: {len(response)} characters")
        return response
    except Exception as e:
        error_msg = f"LLM generation error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"Error: {error_msg}"


@tool("validate_json")
def validate_json(json_string: str) -> str:
    """Validate if a string is valid JSON and return validation status."""
    try:
        data = json.loads(json_string)
        return f"Valid JSON with {len(data)} top-level items"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {str(e)}"


@tool("get_problems_prompt")
def get_problems_prompt() -> str:
    """Get the system and user prompts for problem generation."""
    
    system_prompt = """You are an expert cybersecurity analyst and synthetic data engineer with 15+ years of experience 
in threat intelligence, incident response, and security architecture. Your expertise spans:

- MITRE ATT&CK framework and tactics/techniques
- NIST Cybersecurity Framework 
- Enterprise, Cloud, and EDTC (Education/Data/Technology/Communications) security
- Emerging threats and attack vectors (2023-2025)
- Risk assessment and mitigation strategies

Your task is to analyze initial cybersecurity problem definitions and generate an expanded, 
comprehensive set of problems that covers current and emerging threats across different 
operational environments.

Generate problems that are:
1. Technically accurate and realistic
2. Based on actual threat intelligence and attack patterns
3. Relevant to current threat landscape (2023-2025)
4. Actionable with concrete mitigation strategies
5. Properly categorized by operational environment

Focus on emerging threats including:
- AI-generated attacks and deepfakes
- Supply chain compromises and software dependencies
- Cloud misconfigurations and container security
- Advanced persistent threats (APTs)
- Ransomware with lateral movement capabilities
- IoT and edge computing vulnerabilities
- Zero-day exploits and vulnerability management"""

    user_prompt = """Analyze the initial cybersecurity problems and generate an expanded 
set of exactly 3 NEW problems (in addition to the existing ones) across the operational areas.

For the operational areas (Enterprise, Cloud, EDTC), create 1 additional problem per area that:

1. **Use different attack vectors**: Don't repeat the same attack types from the initial set
2. **Target different assets**: Focus on various systems, data types, and infrastructure components  
3. **Represent current threats**: Include attack scenarios relevant to 2023-2025 threat landscape
4. **Follow consistent schema**: Each problem must include exactly these fields:
   - `area`: Operational environment (Enterprise, Cloud, EDTC)
   - `nature`: Specific attack type or vulnerability (use snake_case format)
   - `description`: Detailed technical scenario (150-300 words)
   - `risk_reduction`: List of 3-4 concrete, actionable mitigation strategies

**Output Format:**
Return a valid JSON object with this exact structure:
```json
{
  "problems": [
    {
      "area": "Enterprise",
      "nature": "advanced_persistent_threat",
      "description": "Detailed technical description of the threat scenario...",
      "risk_reduction": [
        "Specific mitigation strategy 1",
        "Specific mitigation strategy 2", 
        "Specific mitigation strategy 3"
      ]
    }
  ]
}
```

**Important**: Return ONLY the JSON object with exactly 3 problems, no additional text or formatting."""

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"


@tool("get_seed_generation_prompt")
def get_seed_generation_prompt(area: str, nature: str, description: str, risk_reduction: str) -> str:
    """Get the system and user prompts for seed example generation."""
    
    system_prompt = """You are an elite cybersecurity expert with extensive hands-on experience in:
- Security Operations Centers (SOC) and incident response
- Digital forensics and malware analysis
- Network security and protocol analysis
- Cloud security and container technologies
- Threat hunting and intelligence analysis

You have analyzed thousands of real security incidents and understand the technical 
artifacts and indicators that security professionals encounter during actual attacks.

Your task is to generate highly realistic, technically accurate seed examples for 
cybersecurity problems. These examples will be used to train security analysts and 
test detection systems, so they must be indistinguishable from real-world data.

**Technical Requirements:**
- Include actual technical artifacts (logs, packet captures, file hashes, etc.)
- Use realistic IP addresses, domains, and network configurations
- Generate plausible timestamps, file sizes, and system details
- Include proper protocol headers, API calls, and system commands
- Ensure all technical details are consistent and accurate"""

    user_prompt = f"""Generate exactly 10 highly detailed, technically accurate seed examples for this cybersecurity problem:

**Problem Area:** {area}
**Problem Nature:** {nature}
**Problem Description:** {description}
**Risk Reduction Strategies:** {risk_reduction}

**Example Requirements:**

Each example must include realistic technical data appropriate for the attack type:

**For network-based attacks:**
- Raw HTTP/HTTPS request/response data with headers and payloads
- Network packet captures in text format (similar to tcpdump/Wireshark)
- Web server, WAF, or IDS/IPS log entries
- Actual exploit code, injection strings, or malicious payloads

**For phishing/social engineering:**
- Complete email content with realistic headers (From, To, Subject, X-headers)
- SMTP transaction logs showing email routing
- Domain registration details for suspicious domains
- URL structures with obfuscation techniques and redirects

**For malware/system compromise:**
- File hashes (MD5, SHA-1, SHA-256) for malicious files
- Windows Registry changes or Linux filesystem artifacts
- Memory dump analysis snippets and process information
- Command-and-control traffic patterns and network beacons
- Process creation chains and execution artifacts

**For cloud security issues:**
- Cloud API call sequences demonstrating the attack
- IAM policies, bucket configurations, or access control lists
- CloudTrail, CloudWatch, or equivalent audit logs
- Container escape techniques or Kubernetes misconfigurations

**Required Fields for Each Example:**
1. **scenario**: Brief description of this specific attack instance (50-100 words)
2. **technical_data**: Detailed technical artifacts as described above (300+ words)
3. **indicators**: List of specific technical indicators of compromise (IoCs)
4. **detection_method**: Specific methods/tools for detecting this attack
5. **relevant_mitre_techniques**: MITRE ATT&CK technique IDs (e.g., ["T1566.002", "T1059.001"])

**Output Format:**
Return a valid JSON object with this exact structure:
```json
{{
  "examples": [
    {{
      "scenario": "Brief attack scenario description",
      "technical_data": "Detailed technical artifacts and data",
      "indicators": ["indicator1", "indicator2", "indicator3"],
      "detection_method": "How this would be detected in practice",
      "relevant_mitre_techniques": ["T1234", "T5678"]
    }}
  ]
}}
```

**Important**: 
- Return ONLY the JSON object with exactly 10 examples, no additional text
- Ensure all technical data is realistic and could appear in actual incidents
- Include enough detail for security professionals to recognize attack patterns
- Make examples unique but consistent with the problem type"""

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"