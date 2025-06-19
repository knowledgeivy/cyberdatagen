# cyberdata/utils/config_manager.py

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

from cyberdata.utils.logger_config import setup_logger

# Set up logger
logger = setup_logger("cyberdata.utils.config_manager")


class ConfigManager:
    """
    Enhanced configuration manager for the CyberData project with support for
    the new three-step real-world data analysis architecture.
    
    This class provides centralized access to all configuration files,
    data directories, and project paths, including the new enhanced architecture
    with domain discovery, contextual problems, and multi-tier seed validation.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            project_root (Optional[Path]): Override the auto-detected project root
        """
        # Auto-detect project root if not provided
        if project_root is None:
            # Start from this file and work upward to find pyproject.toml
            current = Path(__file__).parent
            while current.parent != current:  # Stop at filesystem root
                if (current / "pyproject.toml").exists():
                    project_root = current
                    break
                current = current.parent
            
            if project_root is None:
                # Fallback: assume standard structure
                project_root = Path(__file__).parent.parent.parent.parent
        
        self.project_root = Path(project_root).resolve()
        logger.info(f"ConfigManager initialized with project root: {self.project_root}")
        
        # Define all standard directories
        self._setup_directories()
        self._validate_structure()
    
    def _setup_directories(self):
        """Set up all standard project directories including new enhanced architecture."""
        # Core directories
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.src_dir = self.project_root / "src"
        
        # Enhanced config subdirectories
        self.prompts_dir = self.config_dir / "prompts"
        self.domain_discovery_dir = self.config_dir / "domain_discovery"
        self.contextual_problems_dir = self.config_dir / "contextual_problems"
        
        # Enhanced data subdirectories
        self.seeds_dir = self.data_dir / "seeds"  # Legacy seeds
        self.seeds_raw_dir = self.data_dir / "seeds-raw"  # Raw unvalidated seeds
        self.seeds_validated_dir = self.data_dir / "seeds-validated"  # High quality seeds
        self.seeds_filtered_dir = self.data_dir / "seeds-filtered"  # Filtered out seeds
        self.seed_validation_dir = self.data_dir / "seed_validation"  # Validation reports
        self.generation_analytics_dir = self.data_dir / "generation-analytics"  # Analytics
        
        # Existing directories
        self.large_samples_dir = self.data_dir / "large_samples"
        self.validation_reports_dir = self.data_dir / "validation_reports"
        self.quality_reports_dir = self.data_dir / "quality_reports"
        
        # Create all directories
        for dir_path in [
            self.config_dir, self.data_dir, self.logs_dir,
            self.prompts_dir, self.domain_discovery_dir, self.contextual_problems_dir,
            self.seeds_dir, self.seeds_raw_dir, self.seeds_validated_dir, 
            self.seeds_filtered_dir, self.seed_validation_dir, self.generation_analytics_dir,
            self.large_samples_dir, self.validation_reports_dir, self.quality_reports_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def _validate_structure(self):
        """Validate that the project structure looks correct."""
        required_files = [
            self.project_root / "pyproject.toml",
            self.project_root / "README.md"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.warning(f"Expected file not found: {file_path}")
    
    # === Enhanced Config File Methods ===
    
    @lru_cache(maxsize=32)
    def load_config_file(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """
        Load a JSON configuration file.
        
        Args:
            filename (str): Name of the config file (with or without .json extension)
            required (bool): Whether the file is required to exist
            
        Returns:
            Dict[str, Any]: Parsed JSON content
            
        Raises:
            FileNotFoundError: If required file doesn't exist
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            if required:
                logger.error(f"Required config file not found: {file_path}")
                raise FileNotFoundError(f"Required config file not found: {file_path}")
            else:
                logger.warning(f"Optional config file not found: {file_path}")
                return {}
        
        try:
            logger.debug(f"Loading config file: {file_path}")
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {file_path}: {str(e)}")
            raise
    
    def save_config_file(self, filename: str, data: Dict[str, Any]) -> Path:
        """
        Save data to a JSON configuration file.
        
        Args:
            filename (str): Name of the config file
            data (Dict[str, Any]): Data to save
            
        Returns:
            Path: Path to the saved file
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
        
        file_path = self.config_dir / filename
        
        try:
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved config file: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving config file {file_path}: {str(e)}")
            raise
    
    # === Enhanced Domain Discovery Methods ===
    
    def load_domain_discovery(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load domain discovery results for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Domain discovery data
        """
        try:
            domain_file = self.domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
            
            if not domain_file.exists():
                logger.warning(f"Domain discovery file not found: {domain_file}")
                return {}
            
            with domain_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded domain discovery for: {dataset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading domain discovery: {str(e)}")
            return {}
    
    def save_domain_discovery(self, dataset_name: str, data: Dict[str, Any]) -> Path:
        """
        Save domain discovery results.
        
        Args:
            dataset_name (str): Name of the dataset
            data (Dict[str, Any]): Domain discovery data
            
        Returns:
            Path: Path to saved file
        """
        domain_file = self.domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
        
        with domain_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved domain discovery for: {dataset_name}")
        return domain_file
    
    def load_technical_indicators(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load technical indicators for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Technical indicators data
        """
        try:
            indicators_file = self.domain_discovery_dir / f"{dataset_name}_technical_indicators.json"
            
            if not indicators_file.exists():
                logger.warning(f"Technical indicators file not found: {indicators_file}")
                return {}
            
            with indicators_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded technical indicators for: {dataset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading technical indicators: {str(e)}")
            return {}
    
    # === Enhanced Contextual Problems Methods ===
    
    def load_contextual_problems(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load contextual problems for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Contextual problems data
        """
        try:
            contextual_file = self.contextual_problems_dir / f"{dataset_name}_contextual_problems.json"
            
            if not contextual_file.exists():
                logger.warning(f"Contextual problems file not found: {contextual_file}")
                return {}
            
            with contextual_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded contextual problems for: {dataset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading contextual problems: {str(e)}")
            return {}
    
    def save_contextual_problems(self, dataset_name: str, data: Dict[str, Any]) -> Path:
        """
        Save contextual problems.
        
        Args:
            dataset_name (str): Name of the dataset
            data (Dict[str, Any]): Contextual problems data
            
        Returns:
            Path: Path to saved file
        """
        contextual_file = self.contextual_problems_dir / f"{dataset_name}_contextual_problems.json"
        
        with contextual_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved contextual problems for: {dataset_name}")
        return contextual_file
    
    def load_attack_taxonomy(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load attack taxonomy for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Attack taxonomy data
        """
        try:
            taxonomy_file = self.contextual_problems_dir / f"{dataset_name}_attack_taxonomy.json"
            
            if not taxonomy_file.exists():
                logger.warning(f"Attack taxonomy file not found: {taxonomy_file}")
                return {}
            
            with taxonomy_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded attack taxonomy for: {dataset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading attack taxonomy: {str(e)}")
            return {}
    
    # === Enhanced Seed Management Methods ===
    
    def get_seeds_raw_file(self, area: str, nature: str) -> Path:
        """Get the path for a raw seeds file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seeds_raw_dir / sanitized_area / f"{sanitized_nature}_examples.json"
    
    def get_seeds_validated_file(self, area: str, nature: str) -> Path:
        """Get the path for a validated seeds file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seeds_validated_dir / sanitized_area / f"{sanitized_nature}_examples.json"
    
    def get_seeds_filtered_file(self, area: str, nature: str) -> Path:
        """Get the path for a filtered seeds file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seeds_filtered_dir / sanitized_area / f"{sanitized_nature}_examples.json"
    
    def get_seed_validation_report_file(self, area: str, nature: str) -> Path:
        """Get the path for a seed validation report file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seed_validation_dir / sanitized_area / f"{sanitized_nature}_seed_validation_report.json"
    
    def get_quality_summary_file(self, area: str, nature: str) -> Path:
        """Get the path for a quality summary file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seed_validation_dir / sanitized_area / f"{sanitized_nature}_quality_summary.json"
    
    # === Legacy Problem Management (Enhanced) ===
    
    def load_problems(self, prefer_updated: bool = True) -> list:
        """
        Load problem definitions, with automatic fallback logic.
        
        Args:
            prefer_updated (bool): Whether to prefer problems_updated.json over problems.json
            
        Returns:
            list: List of problems
        """
        if prefer_updated:
            # Try updated problems first
            try:
                data = self.load_config_file("problems_updated", required=False)
                if data and 'problems' in data:
                    problems = data['problems']
                    logger.info(f"Loaded {len(problems)} problems from problems_updated.json")
                    return problems
            except Exception as e:
                logger.warning(f"Error loading problems_updated.json: {str(e)}")
        
        # Fallback to original problems
        try:
            data = self.load_config_file("problems", required=True)
            problems = data.get('problems', [])
            logger.info(f"Loaded {len(problems)} problems from problems.json")
            return problems
        except FileNotFoundError:
            # Try problems_init as final fallback
            logger.warning("problems.json not found, trying problems_init.json")
            data = self.load_config_file("problems_init", required=True)
            problems = data.get('problems', [])
            logger.info(f"Loaded {len(problems)} problems from problems_init.json")
            return problems
    
    def save_problems(self, problems: list, filename: str = "problems") -> Path:
        """
        Save problems to a configuration file.
        
        Args:
            problems (list): List of problems to save
            filename (str): Base filename (without .json extension)
            
        Returns:
            Path: Path to the saved file
        """
        data = {"problems": problems}
        return self.save_config_file(filename, data)
    
    # === Enhanced Data File Helpers ===
    
    def get_seeds_file(self, area: str, nature: str) -> Path:
        """Get the path for a legacy seeds file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.seeds_dir / sanitized_area / f"{sanitized_nature}_examples.json"
    
    def get_large_samples_file(self, area: str, nature: str) -> Path:
        """Get the path for a large samples file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.large_samples_dir / sanitized_area / f"{sanitized_nature}_large.json"
    
    def get_validation_report_file(self, area: str, nature: str) -> Path:
        """Get the path for a validation report file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.validation_reports_dir / sanitized_area / f"{sanitized_nature}_validation.json"
    
    def get_quality_report_file(self, area: str, nature: str) -> Path:
        """Get the path for a quality report file."""
        sanitized_area = self._sanitize_name(area)
        sanitized_nature = self._sanitize_name(nature)
        return self.quality_reports_dir / sanitized_area / f"{sanitized_nature}_quality_report.json"
    
    def find_existing_file(self, base_dir: Path, nature: str, suffix: str = "") -> Optional[Path]:
        """
        Find an existing file by searching through area subdirectories.
        
        Args:
            base_dir (Path): Base directory to search in
            nature (str): Nature identifier to search for
            suffix (str): File suffix (e.g., "_examples.json")
            
        Returns:
            Optional[Path]: Path to the file if found, None otherwise
        """
        sanitized_nature = self._sanitize_name(nature)
        
        # Search through all subdirectories
        for area_dir in base_dir.glob('*'):
            if area_dir.is_dir():
                for file_path in area_dir.glob('*.json'):
                    if sanitized_nature.lower() in file_path.name.lower():
                        if not suffix or suffix in file_path.name:
                            logger.debug(f"Found existing file: {file_path}")
                            return file_path
        
        return None
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in filenames and directory names."""
        return name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    # === Enhanced Analytics Methods ===
    
    def save_generation_analytics(self, dataset_name: str, analytics_data: Dict[str, Any]) -> Path:
        """
        Save generation analytics data.
        
        Args:
            dataset_name (str): Name of the dataset
            analytics_data (Dict[str, Any]): Analytics data
            
        Returns:
            Path: Path to saved file
        """
        analytics_file = self.generation_analytics_dir / f"{dataset_name}_generation_analytics.json"
        
        with analytics_file.open('w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger.info(f"Saved generation analytics for: {dataset_name}")
        return analytics_file
    
    def load_generation_analytics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load generation analytics data.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Analytics data
        """
        try:
            analytics_file = self.generation_analytics_dir / f"{dataset_name}_generation_analytics.json"
            
            if not analytics_file.exists():
                logger.warning(f"Generation analytics file not found: {analytics_file}")
                return {}
            
            with analytics_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded generation analytics for: {dataset_name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading generation analytics: {str(e)}")
            return {}
    
    # === Enhanced Sample Type Analysis ===
    
    def analyze_sample_types(self, samples: list) -> Dict[str, Any]:
        """
        Analyze the distribution of sample types in a dataset.
        
        Args:
            samples (list): List of samples to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including counts and ratios
        """
        total = len(samples)
        if total == 0:
            return {
                'total_samples': 0,
                'malicious_count': 0,
                'benign_count': 0,
                'other_count': 0,
                'malicious_ratio': 0.0,
                'benign_ratio': 0.0,
                'has_mixed_types': False
            }
        
        malicious_count = 0
        benign_count = 0
        
        for sample in samples:
            # Check various labeling schemes
            if (sample.get('label') == 1 or 
                sample.get('Label') == 1 or
                sample.get('_metadata', {}).get('sample_type') == 'malicious'):
                malicious_count += 1
            elif (sample.get('label') == 0 or 
                  sample.get('Label') == 0 or
                  sample.get('_metadata', {}).get('sample_type') == 'benign'):
                benign_count += 1
        
        other_count = total - malicious_count - benign_count
        
        return {
            'total_samples': total,
            'malicious_count': malicious_count,
            'benign_count': benign_count,
            'other_count': other_count,
            'malicious_ratio': malicious_count / total,
            'benign_ratio': benign_count / total,
            'has_mixed_types': malicious_count > 0 and benign_count > 0
        }
    
    # === Enhanced Ratio Configuration Management ===
    
    def validate_ratio(self, ratio: float) -> bool:
        """
        Validate that a ratio is between 0.0 and 1.0.
        
        Args:
            ratio (float): The ratio to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return 0.0 <= ratio <= 1.0
    
    def calculate_sample_distribution(self, total_count: int, malicious_ratio: float) -> Tuple[int, int]:
        """
        Calculate malicious and benign sample counts based on total and ratio.
        
        Args:
            total_count (int): Total number of samples
            malicious_ratio (float): Ratio of malicious samples (0.0 to 1.0)
            
        Returns:
            Tuple[int, int]: (malicious_count, benign_count)
            
        Raises:
            ValueError: If ratio is invalid
        """
        if not self.validate_ratio(malicious_ratio):
            raise ValueError(f"Invalid malicious ratio: {malicious_ratio}. Must be between 0.0 and 1.0")
        
        malicious_count = int(total_count * malicious_ratio)
        benign_count = total_count - malicious_count
        
        logger.debug(f"Sample distribution for {total_count} total: {malicious_count} malicious, {benign_count} benign")
        return malicious_count, benign_count
    
    # === Convenience Properties ===
    
    @property
    def problems_file(self) -> Path:
        """Path to the main problems.json file."""
        return self.config_dir / "problems.json"
    
    @property
    def problems_updated_file(self) -> Path:
        """Path to the problems_updated.json file."""
        return self.config_dir / "problems_updated.json"
    
    @property
    def problems_init_file(self) -> Path:
        """Path to the problems_init.json file."""
        return self.config_dir / "problems_init.json"
    
    @property
    def data_info_file(self) -> Path:
        """Path to the data_info.yaml file."""
        return self.config_dir / "data_info.yaml"
    
    def clear_cache(self):
        """Clear the configuration cache."""
        self.load_config_file.cache_clear()
        logger.info("Configuration cache cleared")
    
    # === Enhanced Batch Operations ===
    
    def find_all_raw_seeds(self) -> List[Tuple[str, str, Path]]:
        """
        Find all raw seed files in the seeds-raw directory.
        
        Returns:
            List[Tuple[str, str, Path]]: List of (area, nature, file_path) tuples
        """
        seed_files = []
        
        if not self.seeds_raw_dir.exists():
            logger.warning(f"Seeds-raw directory not found: {self.seeds_raw_dir}")
            return seed_files
        
        for area_dir in self.seeds_raw_dir.iterdir():
            if area_dir.is_dir():
                area = area_dir.name
                
                for seed_file in area_dir.glob("*_examples.json"):
                    nature = seed_file.stem.replace('_examples', '')
                    seed_files.append((area, nature, seed_file))
        
        logger.info(f"Found {len(seed_files)} raw seed files")
        return seed_files
    
    def find_all_validated_seeds(self) -> List[Tuple[str, str, Path]]:
        """
        Find all validated seed files in the seeds-validated directory.
        
        Returns:
            List[Tuple[str, str, Path]]: List of (area, nature, file_path) tuples
        """
        seed_files = []
        
        if not self.seeds_validated_dir.exists():
            logger.warning(f"Seeds-validated directory not found: {self.seeds_validated_dir}")
            return seed_files
        
        for area_dir in self.seeds_validated_dir.iterdir():
            if area_dir.is_dir():
                area = area_dir.name
                
                for seed_file in area_dir.glob("*_examples.json"):
                    nature = seed_file.stem.replace('_examples', '')
                    seed_files.append((area, nature, seed_file))
        
        logger.info(f"Found {len(seed_files)} validated seed files")
        return seed_files
    
    def get_dataset_pipeline_status(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get the status of the three-step pipeline for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Pipeline status information
        """
        status = {
            'dataset_name': dataset_name,
            'step_1_domain_discovery': {
                'completed': False,
                'file_exists': False,
                'file_path': None
            },
            'step_2_contextual_problems': {
                'completed': False,
                'file_exists': False,
                'file_path': None
            },
            'step_3_raw_seeds': {
                'completed': False,
                'file_exists': False,
                'files_found': []
            },
            'validation_completed': {
                'completed': False,
                'files_found': []
            },
            'overall_status': 'not_started'
        }
        
        # Check Step 1: Domain Discovery
        domain_file = self.domain_discovery_dir / f"{dataset_name}_domain_discovery.json"
        if domain_file.exists():
            status['step_1_domain_discovery']['completed'] = True
            status['step_1_domain_discovery']['file_exists'] = True
            status['step_1_domain_discovery']['file_path'] = str(domain_file)
        
        # Check Step 2: Contextual Problems
        contextual_file = self.contextual_problems_dir / f"{dataset_name}_contextual_problems.json"
        if contextual_file.exists():
            status['step_2_contextual_problems']['completed'] = True
            status['step_2_contextual_problems']['file_exists'] = True
            status['step_2_contextual_problems']['file_path'] = str(contextual_file)
        
        # Check Step 3: Raw Seeds
        raw_seeds = self.find_all_raw_seeds()
        dataset_raw_seeds = [f for area, nature, f in raw_seeds if dataset_name.lower() in str(f).lower()]
        if dataset_raw_seeds:
            status['step_3_raw_seeds']['completed'] = True
            status['step_3_raw_seeds']['file_exists'] = True
            status['step_3_raw_seeds']['files_found'] = [str(f) for f in dataset_raw_seeds]
        
        # Check Validation
        validated_seeds = self.find_all_validated_seeds()
        dataset_validated_seeds = [f for area, nature, f in validated_seeds if dataset_name.lower() in str(f).lower()]
        if dataset_validated_seeds:
            status['validation_completed']['completed'] = True
            status['validation_completed']['files_found'] = [str(f) for f in dataset_validated_seeds]
        
        # Determine overall status
        if status['validation_completed']['completed']:
            status['overall_status'] = 'fully_completed'
        elif status['step_3_raw_seeds']['completed']:
            status['overall_status'] = 'seeds_generated'
        elif status['step_2_contextual_problems']['completed']:
            status['overall_status'] = 'contextual_enrichment_completed'
        elif status['step_1_domain_discovery']['completed']:
            status['overall_status'] = 'domain_discovery_completed'
        else:
            status['overall_status'] = 'not_started'
        
        return status
    
    def get_all_datasets_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get pipeline status for all datasets that have been processed.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status for each dataset
        """
        all_datasets = set()
        
        # Find datasets from domain discovery
        if self.domain_discovery_dir.exists():
            for file in self.domain_discovery_dir.glob("*_domain_discovery.json"):
                dataset_name = file.stem.replace('_domain_discovery', '')
                all_datasets.add(dataset_name)
        
        # Find datasets from contextual problems
        if self.contextual_problems_dir.exists():
            for file in self.contextual_problems_dir.glob("*_contextual_problems.json"):
                dataset_name = file.stem.replace('_contextual_problems', '')
                all_datasets.add(dataset_name)
        
        # Find datasets from raw seeds (extract from file paths)
        raw_seeds = self.find_all_raw_seeds()
        for area, nature, file_path in raw_seeds:
            # Try to extract dataset name from metadata
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                    dataset_name = data.get('metadata', {}).get('dataset_name')
                    if dataset_name:
                        all_datasets.add(dataset_name)
            except:
                pass
        
        # Get status for each dataset
        status_report = {}
        for dataset_name in sorted(all_datasets):
            status_report[dataset_name] = self.get_dataset_pipeline_status(dataset_name)
        
        return status_report


# Global instance for convenience
_default_config_manager = None


def get_config_manager() -> ConfigManager:
    """
    Get the default ConfigManager instance.
    
    Returns:
        ConfigManager: The default config manager
    """
    global _default_config_manager
    if _default_config_manager is None:
        _default_config_manager = ConfigManager()
    return _default_config_manager


# Convenience functions
def get_project_root() -> Path:
    """Get the project root directory."""
    return get_config_manager().project_root


def get_config_dir() -> Path:
    """Get the config directory."""
    return get_config_manager().config_dir


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_config_manager().data_dir


def load_problems(prefer_updated: bool = True) -> list:
    """Load problems using the default config manager."""
    return get_config_manager().load_problems(prefer_updated)


def save_problems(problems: list, filename: str = "problems") -> Path:
    """Save problems using the default config manager."""
    return get_config_manager().save_problems(problems, filename)


def validate_ratio(ratio: float) -> bool:
    """Validate a ratio using the default config manager."""
    return get_config_manager().validate_ratio(ratio)


def calculate_sample_distribution(total_count: int, malicious_ratio: float) -> Tuple[int, int]:
    """Calculate sample distribution using the default config manager."""
    return get_config_manager().calculate_sample_distribution(total_count, malicious_ratio)


# Enhanced convenience functions for new architecture
def load_domain_discovery(dataset_name: str) -> Dict[str, Any]:
    """Load domain discovery using the default config manager."""
    return get_config_manager().load_domain_discovery(dataset_name)


def save_domain_discovery(dataset_name: str, data: Dict[str, Any]) -> Path:
    """Save domain discovery using the default config manager."""
    return get_config_manager().save_domain_discovery(dataset_name, data)


def load_contextual_problems(dataset_name: str) -> Dict[str, Any]:
    """Load contextual problems using the default config manager."""
    return get_config_manager().load_contextual_problems(dataset_name)


def save_contextual_problems(dataset_name: str, data: Dict[str, Any]) -> Path:
    """Save contextual problems using the default config manager."""
    return get_config_manager().save_contextual_problems(dataset_name, data)


def get_dataset_pipeline_status(dataset_name: str) -> Dict[str, Any]:
    """Get dataset pipeline status using the default config manager."""
    return get_config_manager().get_dataset_pipeline_status(dataset_name)


def get_all_datasets_status() -> Dict[str, Dict[str, Any]]:
    """Get all datasets status using the default config manager."""
    return get_config_manager().get_all_datasets_status()