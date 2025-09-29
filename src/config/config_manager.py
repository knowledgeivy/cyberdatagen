"""
配置管理模块
负责加载、验证和管理实验配置
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ExperimentConfig:
    """实验配置数据类"""

    # 基本信息
    name: str
    description: str
    dataset: str
    random_seed: int = 42

    # 数据配置
    raw_data_path: str = "./raw/"
    processed_data_path: str = "./data/paper/processed/"
    synthetic_data_path: str = "./data/paper/synthetic/"
    datasets_path: str = "./data/paper/datasets/"

    # 采样参数
    n_groups: int = 20
    sample_size_per_group: int = 9000
    spam_ratio: float = 0.1
    test_split: float = 0.2
    synthetic_ratios: List[int] = field(default_factory=lambda: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    trials_per_config: int = 20

    # LLM配置
    llm_engines: List[str] = field(default_factory=lambda: ["gpt-4.1-mini"])
    api_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    save_checkpoint_every: int = 100

    # Prompt配置
    prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # 分类器配置
    classifiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 特征提取配置
    feature_extraction: Dict[str, Any] = field(default_factory=dict)

    # 评估配置
    evaluation: Dict[str, Any] = field(default_factory=dict)

    # 输出配置
    output: Dict[str, Any] = field(default_factory=dict)

    # 并行处理配置
    parallel: Dict[str, Any] = field(default_factory=dict)

    # 质量控制配置
    quality_control: Dict[str, Any] = field(default_factory=dict)

    # 实验策略配置
    strategies: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # 日志配置
    logging: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Optional[ExperimentConfig] = None

    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            ExperimentConfig: 配置对象
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            # 验证配置
            self._validate_config(config_dict)

            # 创建配置对象
            self.config = self._create_config_object(config_dict)

            logger.info(f"配置加载成功: {config_path}")
            return self.config

        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    def _validate_config(self, config_dict: Dict[str, Any]) -> None:
        """
        验证配置有效性

        Args:
            config_dict: 配置字典
        """
        required_sections = ['experiment', 'data', 'llm', 'prompts', 'classifiers']

        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"缺少必需的配置节: {section}")

        # 验证实验配置
        exp_config = config_dict['experiment']
        required_exp_fields = ['name', 'dataset']
        for field in required_exp_fields:
            if field not in exp_config:
                raise ValueError(f"缺少必需的实验配置字段: {field}")

        # 验证数据配置
        data_config = config_dict['data']
        if data_config.get('spam_ratio', 0) <= 0 or data_config.get('spam_ratio', 0) >= 1:
            raise ValueError("spam_ratio必须在0到1之间")

        if data_config.get('n_groups', 0) <= 0:
            raise ValueError("n_groups必须大于0")

        if data_config.get('sample_size_per_group', 0) <= 0:
            raise ValueError("sample_size_per_group必须大于0")

        # 验证synthetic_ratios
        synthetic_ratios = data_config.get('synthetic_ratios', [])
        if not isinstance(synthetic_ratios, list) or len(synthetic_ratios) == 0:
            raise ValueError("synthetic_ratios必须是非空列表")

        for ratio in synthetic_ratios:
            if not isinstance(ratio, (int, float)) or ratio < 0 or ratio > 100:
                raise ValueError("synthetic_ratios中的值必须在0到100之间")

        # 验证LLM配置
        llm_config = config_dict['llm']
        if not llm_config.get('engines'):
            raise ValueError("必须指定至少一个LLM引擎")

        # 验证prompt配置
        prompts_config = config_dict['prompts']
        if not prompts_config:
            raise ValueError("必须配置至少一个prompt")

        for prompt_name, prompt_config in prompts_config.items():
            if 'template' not in prompt_config:
                raise ValueError(f"Prompt {prompt_name} 缺少template")

        # 验证分类器配置
        classifiers_config = config_dict['classifiers']
        enabled_classifiers = [name for name, config in classifiers_config.items()
                              if config.get('enabled', True)]
        if not enabled_classifiers:
            raise ValueError("必须启用至少一个分类器")

    def _create_config_object(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        从配置字典创建配置对象

        Args:
            config_dict: 配置字典

        Returns:
            ExperimentConfig: 配置对象
        """
        # 展平配置字典
        flattened_config = {}

        # 实验配置
        exp_config = config_dict['experiment']
        flattened_config.update(exp_config)

        # 数据配置
        data_config = config_dict['data']
        for key, value in data_config.items():
            if key.endswith('_path'):
                flattened_config[key] = value
            else:
                flattened_config[key] = value

        # LLM配置
        llm_config = config_dict['llm']
        flattened_config['llm_engines'] = llm_config.get('engines', ['gpt-4.1-mini'])
        flattened_config['api_config'] = llm_config.get('api_config', {})
        flattened_config['batch_size'] = llm_config.get('batch_size', 10)
        flattened_config['max_retries'] = llm_config.get('max_retries', 3)
        flattened_config['retry_delay'] = llm_config.get('retry_delay', 1.0)
        flattened_config['save_checkpoint_every'] = llm_config.get('save_checkpoint_every', 100)

        # 其他配置
        flattened_config['prompts'] = config_dict.get('prompts', {})
        flattened_config['classifiers'] = config_dict.get('classifiers', {})
        flattened_config['feature_extraction'] = config_dict.get('feature_extraction', {})
        flattened_config['evaluation'] = config_dict.get('evaluation', {})
        flattened_config['output'] = config_dict.get('output', {})
        flattened_config['parallel'] = config_dict.get('parallel', {})
        flattened_config['quality_control'] = config_dict.get('quality_control', {})
        flattened_config['strategies'] = config_dict.get('strategies', {})
        flattened_config['logging'] = config_dict.get('logging', {})

        return ExperimentConfig(**flattened_config)

    def get_prompt_template(self, prompt_name: str) -> str:
        """
        获取prompt模板

        Args:
            prompt_name: prompt名称

        Returns:
            str: prompt模板
        """
        if self.config is None:
            raise ValueError("配置未加载")

        if prompt_name not in self.config.prompts:
            raise ValueError(f"未找到prompt: {prompt_name}")

        return self.config.prompts[prompt_name]['template']

    def get_classifier_config(self, classifier_name: str) -> Dict[str, Any]:
        """
        获取分类器配置

        Args:
            classifier_name: 分类器名称

        Returns:
            Dict[str, Any]: 分类器配置
        """
        if self.config is None:
            raise ValueError("配置未加载")

        if classifier_name not in self.config.classifiers:
            raise ValueError(f"未找到分类器: {classifier_name}")

        return self.config.classifiers[classifier_name]

    def get_enabled_classifiers(self) -> List[str]:
        """
        获取启用的分类器列表

        Returns:
            List[str]: 启用的分类器名称列表
        """
        if self.config is None:
            raise ValueError("配置未加载")

        return [name for name, config in self.config.classifiers.items()
                if config.get('enabled', True)]

    def get_output_path(self, output_type: str) -> str:
        """
        获取输出路径

        Args:
            output_type: 输出类型 ('results', 'plots', 'reports', 'logs')

        Returns:
            str: 输出路径
        """
        if self.config is None:
            raise ValueError("配置未加载")

        path_key = f"{output_type}_path"
        if path_key not in self.config.output:
            raise ValueError(f"未找到输出路径配置: {output_type}")

        path = self.config.output[path_key]
        os.makedirs(path, exist_ok=True)
        return path

    def save_config(self, save_path: str) -> None:
        """
        保存当前配置到文件

        Args:
            save_path: 保存路径
        """
        if self.config is None:
            raise ValueError("配置未加载")

        # 将配置对象转换为字典
        config_dict = {
            'experiment': {
                'name': self.config.name,
                'description': self.config.description,
                'dataset': self.config.dataset,
                'random_seed': self.config.random_seed
            },
            'data': {
                'raw_data_path': self.config.raw_data_path,
                'processed_data_path': self.config.processed_data_path,
                'synthetic_data_path': self.config.synthetic_data_path,
                'datasets_path': self.config.datasets_path,
                'n_groups': self.config.n_groups,
                'sample_size_per_group': self.config.sample_size_per_group,
                'spam_ratio': self.config.spam_ratio,
                'test_split': self.config.test_split,
                'synthetic_ratios': self.config.synthetic_ratios,
                'trials_per_config': self.config.trials_per_config
            },
            'llm': {
                'engines': self.config.llm_engines,
                'api_config': self.config.api_config,
                'batch_size': self.config.batch_size,
                'max_retries': self.config.max_retries,
                'retry_delay': self.config.retry_delay,
                'save_checkpoint_every': self.config.save_checkpoint_every
            },
            'prompts': self.config.prompts,
            'classifiers': self.config.classifiers,
            'feature_extraction': self.config.feature_extraction,
            'evaluation': self.config.evaluation,
            'output': self.config.output,
            'parallel': self.config.parallel,
            'quality_control': self.config.quality_control,
            'strategies': self.config.strategies,
            'logging': self.config.logging
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置已保存到: {save_path}")


def load_config(config_path: str) -> ExperimentConfig:
    """
    便捷函数：加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        ExperimentConfig: 配置对象
    """
    manager = ConfigManager()
    return manager.load_config(config_path)