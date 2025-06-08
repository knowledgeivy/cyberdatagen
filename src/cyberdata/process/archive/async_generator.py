# cyberdata/process/async_generator_v2.py
# not working

import asyncio
import aiohttp
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, AsyncGenerator
from dataclasses import dataclass
import logging
import sys
import re
import random
from collections import defaultdict

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(CURRENT_DIR.parent))

from cyberdata.utils.logger_config import setup_logger
from cyberdata.utils.prompt_loader import load_prompt
from cyberdata.utils.config_manager import get_config_manager
from dotenv import load_dotenv
import os

# Set up logger
logger = setup_logger("cyberdata.scripts.async_generator_v2")
load_dotenv()

@dataclass
class AsyncGenerationTask:
    """Async generation task"""
    problem: Dict
    sample_type: str
    batch_size: int
    task_id: str
    examples: Optional[List] = None
    priority: int = 0

class ConfigurableAsyncDuplicateDetector:
    """Async-safe duplicate detector with YAML configuration"""
    
    def __init__(self, config: Dict):
        dup_config = config.get('duplicate_detection', {})
        self.enabled = dup_config.get('enabled', True)
        self.check_fields = dup_config.get('check_fields', ['scenario', 'technical_data'])
        
        # Normalization settings
        norm_config = dup_config.get('normalization', {})
        self.normalize_lowercase = norm_config.get('lowercase', True)
        self.normalize_whitespace = norm_config.get('remove_extra_whitespace', True)
        self.normalize_values = norm_config.get('remove_specific_values', True)
        
        self.content_hashes: Set[str] = set()
        self.lock = asyncio.Lock()
        
        logger.debug(f"Async duplicate detector initialized: enabled={self.enabled}")
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content based on configuration"""
        if self.normalize_lowercase:
            content = content.lower()
        
        if self.normalize_whitespace:
            content = re.sub(r'\s+', ' ', content.strip())
        
        if self.normalize_values:
            # Replace specific values with placeholders
            content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', content)
            content = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', 'DOMAIN', content)
            content = re.sub(r'\b[0-9a-fA-F]{8,64}\b', 'HASH', content)
            content = re.sub(r'\b\d{4}-\d{2}-\d{2}', 'DATE', content)
        
        return content
    
    async def _generate_content_hash(self, sample: Dict) -> str:
        """Generate content hash asynchronously"""
        if not self.enabled:
            return str(random.random())
        
        content_parts = []
        for field in self.check_fields:
            if field in sample:
                content = str(sample[field])
                normalized_content = self._normalize_content(content)
                content_parts.append(normalized_content)
        
        combined_content = "|".join(content_parts)
        return hashlib.sha256(combined_content.encode()).hexdigest()
    
    async def is_duplicate(self, sample: Dict) -> bool:
        """Check for duplicate asynchronously"""
        content_hash = await self._generate_content_hash(sample)
        
        async with self.lock:
            if content_hash in self.content_hashes:
                return True
            self.content_hashes.add(content_hash)
            return False
    
    async def add_existing_samples(self, samples: List[Dict]):
        """Add existing samples to duplicate detector"""
        if not self.enabled:
            return
        
        async with self.lock:
            for sample in samples:
                content_hash = await self._generate_content_hash(sample)
                self.content_hashes.add(content_hash)
        
        logger.debug(f"Added {len(samples)} existing samples to async duplicate detector")
    
    async def get_stats(self) -> Dict:
        """Get statistics about duplicate detection"""
        async with self.lock:
            return {
                "enabled": self.enabled,
                "total_hashes": len(self.content_hashes),
                "check_fields": self.check_fields
            }

class ConfigurableAsyncOpenAIClient:
    """Async OpenAI client with configuration and rate limiting"""
    
    def __init__(self, config: Dict, api_key: str):
        self.api_key = api_key
        
        # Load async configuration
        async_config = config.get('parallel_generation', {}).get('async_based', {})
        self.max_concurrent = async_config.get('max_concurrent', 15)
        self.requests_per_minute = async_config.get('requests_per_minute', 50)
        self.connection_timeout = async_config.get('connection_timeout', 30)
        self.max_retry_attempts = async_config.get('max_retry_attempts', 3)
        self.backoff_factor = async_config.get('backoff_factor', 2)
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.rate_limiter = asyncio.Semaphore(self.requests_per_minute)
        
        logger.info(f"Async OpenAI client initialized: {self.max_concurrent} concurrent, "
                   f"{self.requests_per_minute} req/min")
    
    async def _wait_for_rate_limit(self):
        """Rate limiting with async release"""
        await self.rate_limiter.acquire()
        asyncio.create_task(self._release_rate_limit())
    
    async def _release_rate_limit(self):
        """Release rate limit after delay"""
        await asyncio.sleep(60 / self.requests_per_minute)
        self.rate_limiter.release()
    
    async def generate_completion(self, system_prompt: str, user_prompt: str, 
                                model: str = "gpt-4.1-mini", temperature: float = 0.7) -> str:
        """Generate completion with async OpenAI call and retry logic"""
        
        for attempt in range(self.max_retry_attempts):
            try:
                async with self.semaphore:
                    await self._wait_for_rate_limit()
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": temperature,
                        "max_tokens": 16384
                    }
                    
                    timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return result['choices'][0]['message']['content']
                            elif response.status == 429:  # Rate limit
                                if attempt < self.max_retry_attempts - 1:
                                    wait_time = self.backoff_factor ** attempt
                                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    raise Exception("Rate limit exceeded, max retries reached")
                            else:
                                error_text = await response.text()
                                raise Exception(f"OpenAI API error {response.status}: {error_text}")
                                
            except asyncio.TimeoutError:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"Request timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Request timeout, max retries reached")
            except Exception as e:
                if attempt < self.max_retry_attempts - 1:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"Request failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise

class ConfigurableAsyncParallelGenerator:
    """Async parallel generator with YAML configuration"""
    
    def __init__(self, config_override: Optional[Dict] = None, environment: str = "production"):
        """Initialize with YAML configuration"""
        self.config_manager = get_config_manager()
        self.environment = environment
        
        # Load configuration
        self.config = self._load_configuration(config_override)
        
        # Initialize components
        self.duplicate_detector = ConfigurableAsyncDuplicateDetector(self.config)
        self.openai_client = ConfigurableAsyncOpenAIClient(
            self.config,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Get configuration values
        async_config = self.config.get('parallel_generation', {}).get('async_based', {})
        self.max_concurrent = async_config.get('max_concurrent', 15)
        self.default_batch_size = async_config.get('batch_size', 3)
        
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_samples": 0,
            "duplicates_filtered": 0,
            "generation_time": 0
        }
        
        logger.info(f"Async generator initialized: {self.max_concurrent} max concurrent, "
                   f"batch size {self.default_batch_size}")
    
    def _load_configuration(self, config_override: Optional[Dict] = None) -> Dict:
        """Load configuration from YAML (same as sync version)"""
        try:
            config_file = self.config_manager.prompts_dir / "parallel_generation_config.yaml"
            
            if not config_file.exists():
                logger.warning(f"Config not found, creating default: {config_file}")
                self._create_default_config(config_file)
            
            from cyberdata.utils.prompt_loader import get_prompt_loader
            loader = get_prompt_loader()
            config = loader.load_prompt_file("parallel_generation_config")
            
            # Apply environment overrides
            env_config = config.get("environments", {}).get(self.environment, {})
            if env_config:
                config = self._merge_configs(config, {"parallel_generation": {"defaults": env_config}})
            
            # Apply runtime overrides
            if config_override:
                config = self._merge_configs(config, config_override)
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_fallback_config()
    
    def _create_default_config(self, config_file: Path):
        """Create default config (same as sync version)"""
        default_config = """version: "1.0"
description: "Configuration for parallel synthetic data generation"

parallel_generation:
  async_based:
    max_concurrent: 15
    batch_size: 3
    requests_per_minute: 50
    max_retry_attempts: 3
  
  defaults:
    method: "async_based"
    temperature:
      malicious_generation: 0.7
      benign_generation: 0.7

duplicate_detection:
  enabled: true
  check_fields: 
    - "scenario"
    - "technical_data"
"""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(default_config)
    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration"""
        return {
            "parallel_generation": {
                "async_based": {
                    "max_concurrent": 15,
                    "batch_size": 3,
                    "requests_per_minute": 50
                }
            },
            "duplicate_detection": {
                "enabled": True,
                "check_fields": ["scenario", "technical_data"]
            }
        }
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Merge configurations recursively"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_problem_config(self, problem: Dict) -> Dict:
        """Get problem-specific configuration"""
        nature = problem.get('nature', '')
        problem_configs = self.config.get('problem_specific', {})
        
        if nature in problem_configs:
            return problem_configs[nature]
        
        return {
            'batch_size': self.default_batch_size,
            'max_concurrent': self.max_concurrent,
            'malicious_ratio': 0.5,
            'temperature': 0.7
        }
    
    async def load_problems(self) -> List[Dict]:
        """Load problem definitions async"""
        return self.config_manager.load_problems()
    
    async def load_existing_samples(self, problem: Dict) -> List[Dict]:
        """Load existing samples asynchronously"""
        try:
            samples_file = self.config_manager.get_large_samples_file(
                problem['area'], problem['nature']
            )
            
            if samples_file.exists():
                # Use sync file reading (async file I/O is often overkill)
                with open(samples_file, 'r') as f:
                    data = json.load(f)
                existing_samples = data.get('samples', [])
                
                await self.duplicate_detector.add_existing_samples(existing_samples)
                return existing_samples
        except Exception as e:
            logger.warning(f"Could not load existing samples: {e}")
        
        return []
    
    async def create_generation_tasks(self, problem: Dict, malicious_count: int, 
                                    benign_count: int) -> List[AsyncGenerationTask]:
        """Create async generation tasks"""
        problem_config = self.get_problem_config(problem)
        batch_size = problem_config.get('batch_size', self.default_batch_size)
        
        tasks = []
        task_id = 0
        
        # Load examples for malicious tasks
        examples = []
        if malicious_count > 0:
            try:
                examples_file = self.config_manager.get_seeds_file(problem['area'], problem['nature'])
                if examples_file.exists():
                    with open(examples_file, 'r') as f:
                        data = json.load(f)
                    examples = data.get('examples', [])[:2]
            except Exception as e:
                logger.warning(f"Could not load examples: {e}")
        
        # Create malicious tasks
        remaining_malicious = malicious_count
        while remaining_malicious > 0:
            current_batch = min(batch_size, remaining_malicious)
            task = AsyncGenerationTask(
                problem=problem,
                sample_type='malicious',
                batch_size=current_batch,
                task_id=f"{problem['nature']}_mal_{task_id}",
                examples=examples,
                priority=2
            )
            tasks.append(task)
            remaining_malicious -= current_batch
            task_id += 1
        
        # Create benign tasks
        remaining_benign = benign_count
        while remaining_benign > 0:
            current_batch = min(batch_size, remaining_benign)
            task = AsyncGenerationTask(
                problem=problem,
                sample_type='benign',
                batch_size=current_batch,
                task_id=f"{problem['nature']}_ben_{task_id}",
                priority=1
            )
            tasks.append(task)
            remaining_benign -= current_batch
            task_id += 1
        
        # Sort by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Created {len(tasks)} async tasks for {problem['nature']}")
        return tasks
    
    async def execute_generation_task(self, task: AsyncGenerationTask) -> Dict:
        """Execute async generation task"""
        try:
            logger.info(f"Executing async task {task.task_id}: {task.batch_size} {task.sample_type} samples")
            
            if task.sample_type == 'malicious':
                samples = await self._generate_malicious_batch_async(task)
            else:
                samples = await self._generate_benign_batch_async(task)
            
            # Filter duplicates asynchronously
            unique_samples = []
            duplicates_count = 0
            
            for sample in samples:
                is_duplicate = await self.duplicate_detector.is_duplicate(sample)
                
                if not is_duplicate:
                    sample['sample_type'] = task.sample_type
                    sample['is_attack'] = task.sample_type == 'malicious'
                    sample['generation_task_id'] = task.task_id
                    sample['generation_timestamp'] = time.time()
                    unique_samples.append(sample)
                else:
                    duplicates_count += 1
            
            # Update stats
            self.stats["completed_tasks"] += 1
            self.stats["total_samples"] += len(samples)
            self.stats["duplicates_filtered"] += duplicates_count
            
            logger.info(f"Async task {task.task_id} completed: {len(unique_samples)} unique samples")
            
            return {
                "task_id": task.task_id,
                "samples": unique_samples,
                "sample_type": task.sample_type,
                "success": True,
                "duplicates_filtered": duplicates_count
            }
            
        except Exception as e:
            self.stats["failed_tasks"] += 1
            logger.error(f"Async task {task.task_id} failed: {e}")
            return {
                "task_id": task.task_id,
                "samples": [],
                "sample_type": task.sample_type,
                "success": False,
                "error": str(e)
            }
    
    async def _generate_malicious_batch_async(self, task: AsyncGenerationTask) -> List[Dict]:
        """Generate malicious samples async"""
        problem_config = self.get_problem_config(task.problem)
        temperature = problem_config.get('temperature', 0.7)
        
        examples_json = json.dumps(task.examples, indent=2) if task.examples else "[]"
        
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.generation.user.template",
            nature=task.problem['nature'],
            area=task.problem['area'],
            description=task.problem.get('description', ''),
            examples_json=examples_json,
            count=task.batch_size
        )
        
        response_content = await self.openai_client.generate_completion(
            system_prompt=system_content,
            user_prompt=user_content,
            temperature=temperature
        )
        
        return self._extract_json_from_response(response_content).get('samples', [])
    
    async def _generate_benign_batch_async(self, task: AsyncGenerationTask) -> List[Dict]:
        """Generate benign samples async"""
        problem_config = self.get_problem_config(task.problem)
        temperature = problem_config.get('temperature', 0.7)
        
        system_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.system.template"
        )
        
        user_content = load_prompt(
            "large_generation_prompts",
            "prompts.benign_generation.user.template",
            area=task.problem['area'],
            count=task.batch_size,
            context=task.problem.get('description', '')
        )
        
        response_content = await self.openai_client.generate_completion(
            system_prompt=system_content,
            user_prompt=user_content,
            temperature=temperature
        )
        
        return self._extract_json_from_response(response_content).get('samples', [])
    
    def _extract_json_from_response(self, response_content: str) -> Dict:
        """Extract JSON from response (same as sync version)"""
        if '```' in response_content:
            pattern = r'```(?:json)?\s*([\s\S]*?)```'
            matches = re.findall(pattern, response_content)
            if matches:
                response_content = matches[0]
        
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            try:
                fixed_content = response_content.replace("'", '"')
                fixed_content = re.sub(r'}\s*{', '},{', fixed_content)
                fixed_content = re.sub(r',\s*}', '}', fixed_content)
                fixed_content = re.sub(r',\s*]', ']', fixed_content)
                return json.loads(fixed_content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response")
                return {"samples": []}
    
    async def generate_with_progress(self, problem: Dict, malicious_count: int, 
                                   benign_count: int) -> AsyncGenerator[Dict, None]:
        """Generate samples with real-time progress updates"""
        logger.info(f"Starting async generation for {problem['nature']}: "
                   f"{malicious_count} malicious, {benign_count} benign")
        
        start_time = time.time()
        
        # Load existing samples
        existing_samples = await self.load_existing_samples(problem)
        
        # Create tasks
        tasks = await self.create_generation_tasks(problem, malicious_count, benign_count)
        self.stats["total_tasks"] = len(tasks)
        
        if not tasks:
            return
        
        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_task(task):
            async with semaphore:
                return await self.execute_generation_task(task)
        
        all_samples = []
        
        # Process in chunks for progress updates
        chunk_size = min(self.max_concurrent, len(tasks))
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i + chunk_size]
            
            # Execute chunk
            results = await asyncio.gather(
                *[bounded_task(task) for task in chunk_tasks],
                return_exceptions=True
            )
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task exception: {result}")
                    continue
                
                if result["success"]:
                    all_samples.extend(result["samples"])
                    
                    # Yield progress
                    yield {
                        "type": "progress",
                        "completed_tasks": self.stats["completed_tasks"],
                        "total_tasks": self.stats["total_tasks"],
                        "samples_generated": len(all_samples),
                        "duplicates_filtered": self.stats["duplicates_filtered"]
                    }
        
        # Shuffle and yield final result
        random.shuffle(all_samples)
        generation_time = time.time() - start_time
        self.stats["generation_time"] = generation_time
        
        yield {
            "type": "complete",
            "samples": all_samples,
            "existing_samples": existing_samples,
            "stats": self.stats.copy(),
            "generation_time": generation_time
        }
    
    async def save_samples_async(self, problem: Dict, new_samples: List[Dict], 
                               existing_samples: List[Dict]):
        """Save samples with metadata"""
        if not new_samples:
            return
        
        out_file = self.config_manager.get_large_samples_file(problem['area'], problem['nature'])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_samples = existing_samples + new_samples
        
        # Calculate stats
        malicious_new = sum(1 for s in new_samples if s.get('sample_type') == 'malicious')
        benign_new = sum(1 for s in new_samples if s.get('sample_type') == 'benign')
        
        metadata = {
            'total_samples': len(all_samples),
            'new_samples_added': len(new_samples),
            'new_malicious_samples': malicious_new,
            'new_benign_samples': benign_new,
            'generation_method': 'async_parallel',
            'generation_stats': self.stats.copy(),
            'duplicate_detection_stats': await self.duplicate_detector.get_stats(),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': self.environment
        }
        
        output_data = {'samples': all_samples, 'metadata': metadata}
        
        # Use sync file I/O (simpler than aiofiles dependency)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Async saved {len(new_samples)} samples to {out_file}")

async def main_async(count: int = 100, malicious_ratio: float = 0.5, 
                    problem_filter: List[str] = None, max_concurrent: int = None,
                    config_override: Dict = None, environment: str = "production"):
    """Main async function with YAML configuration"""
    logger.info(f"Starting async generation: count={count}, ratio={malicious_ratio}")
    
    if not 0.0 <= malicious_ratio <= 1.0:
        logger.error("Invalid malicious ratio")
        return
    
    malicious_count = int(count * malicious_ratio)
    benign_count = count - malicious_count
    
    # Apply max_concurrent override
    if max_concurrent:
        if not config_override:
            config_override = {}
        config_override['parallel_generation'] = config_override.get('parallel_generation', {})
        config_override['parallel_generation']['async_based'] = config_override['parallel_generation'].get('async_based', {})
        config_override['parallel_generation']['async_based']['max_concurrent'] = max_concurrent
    
    # Initialize generator
    generator = ConfigurableAsyncParallelGenerator(
        config_override=config_override,
        environment=environment
    )
    
    # Load and filter problems
    problems = await generator.load_problems()
    if problem_filter:
        problems = [p for p in problems if p['nature'] in problem_filter]
    
    logger.info(f"Processing {len(problems)} problems asynchronously")
    
    # Process each problem
    for i, problem in enumerate(problems):
        logger.info(f"\nAsync processing {i+1}/{len(problems)}: {problem['nature']}")
        
        try:
            new_samples = []
            existing_samples = []
            
            async for update in generator.generate_with_progress(
                problem, malicious_count, benign_count
            ):
                if update["type"] == "progress":
                    logger.info(f"Progress: {update['completed_tasks']}/{update['total_tasks']} tasks")
                elif update["type"] == "complete":
                    new_samples = update["samples"]
                    existing_samples = update["existing_samples"]
                    logger.info(f"Generated {len(new_samples)} samples in {update['generation_time']:.2f}s")
            
            if new_samples:
                await generator.save_samples_async(problem, new_samples, existing_samples)
            
        except Exception as e:
            logger.error(f"Async error processing {problem['nature']}: {e}")

def run_async_main():
    """CLI wrapper for async main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Async parallel data generation with YAML config")
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--malicious-ratio', type=float, default=0.5)
    parser.add_argument('--concurrent', type=int, help='Max concurrent tasks')
    parser.add_argument('--problems', nargs='+', help='Specific problems')
    parser.add_argument('--environment', choices=['development', 'production', 'testing'], default='production')
    
    args = parser.parse_args()
    
    asyncio.run(main_async(
        count=args.count,
        malicious_ratio=args.malicious_ratio,
        problem_filter=args.problems,
        max_concurrent=args.concurrent,
        environment=args.environment
    ))

if __name__ == '__main__':
    run_async_main()