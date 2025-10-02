"""
LLM生成模块
负责调用不同LLM API生成synthetic spam email data
"""

import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import asyncio
import aiohttp
from tqdm import tqdm
from loguru import logger
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv

from ..config.config_manager import ExperimentConfig

# 加载环境变量
load_dotenv()


class LLMGenerator:
    """LLM生成器基类"""

    def __init__(self, config: ExperimentConfig, engine_name: str):
        """
        初始化生成器

        Args:
            config: 实验配置
            engine_name: LLM引擎名称
        """
        self.config = config
        self.engine_name = engine_name
        self.api_config = config.api_config.get(engine_name.split('-')[0], {})

    async def generate_single(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        生成单个文本

        Args:
            prompt: 输入prompt
            max_retries: 最大重试次数

        Returns:
            Optional[str]: 生成的文本
        """
        raise NotImplementedError

    async def generate_single_with_system(self, user_prompt: str, system_prompt: str = '', max_retries: int = 3) -> Optional[str]:
        """
        使用system prompt生成单个文本

        Args:
            user_prompt: 用户输入prompt
            system_prompt: 系统prompt
            max_retries: 最大重试次数

        Returns:
            Optional[str]: 生成的文本
        """
        raise NotImplementedError

    def parse_response(self, response: str) -> Dict[str, str]:
        """
        解析LLM响应，提取主题和内容

        Args:
            response: LLM响应

        Returns:
            Dict[str, str]: 包含'subject'和'body'的字典
        """
        import json
        import re

        result = {'subject': '', 'body': ''}

        try:
            # 首先尝试直接解析JSON
            json_data = json.loads(response.strip())
            result['subject'] = json_data.get('rewritten_subject', '')
            result['body'] = json_data.get('rewritten_body', '')
            return result
        except json.JSONDecodeError:
            pass

        # 如果直接解析失败，尝试提取JSON块
        try:
            # 寻找JSON块
            json_match = re.search(r'\{[^{}]*"rewritten_subject"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_data = json.loads(json_str)
                result['subject'] = json_data.get('rewritten_subject', '')
                result['body'] = json_data.get('rewritten_body', '')
                return result
        except json.JSONDecodeError:
            pass

        # 如果JSON解析都失败，退回到文本解析
        lines = response.strip().split('\n')
        current_section = None
        content_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 识别主题部分
            if any(keyword in line.lower() for keyword in ['rewritten_subject', 'subject', '重写后的主题', '生成的主题', '主题']):
                if current_section == 'body':
                    result['body'] = '\n'.join(content_lines).strip()
                current_section = 'subject'
                content_lines = []
                # 尝试提取同一行的内容
                colon_pos = line.find(':')
                if colon_pos != -1 and colon_pos < len(line) - 1:
                    content = line[colon_pos + 1:].strip().strip('"')
                    if content:
                        content_lines.append(content)
                continue

            # 识别内容部分
            if any(keyword in line.lower() for keyword in ['rewritten_body', 'body', '重写后的内容', '生成的内容', '内容']):
                if current_section == 'subject':
                    result['subject'] = '\n'.join(content_lines).strip()
                current_section = 'body'
                content_lines = []
                # 尝试提取同一行的内容
                colon_pos = line.find(':')
                if colon_pos != -1 and colon_pos < len(line) - 1:
                    content = line[colon_pos + 1:].strip().strip('"')
                    if content:
                        content_lines.append(content)
                continue

            # 收集内容
            if current_section:
                content_lines.append(line.strip('"'))

        # 处理最后一个部分
        if current_section == 'subject':
            result['subject'] = '\n'.join(content_lines).strip()
        elif current_section == 'body':
            result['body'] = '\n'.join(content_lines).strip()

        # 如果解析失败，尝试简单分割
        if not result['subject'] and not result['body']:
            if lines:
                result['subject'] = lines[0].strip()
                if len(lines) > 1:
                    result['body'] = '\n'.join(lines[1:]).strip()

        return result


class OpenAIGenerator(LLMGenerator):
    """OpenAI GPT生成器"""

    def __init__(self, config: ExperimentConfig, engine_name: str = "gpt-4.1-mini"):
        super().__init__(config, engine_name)
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.model = self.api_config.get('model', 'gpt-4o-mini')

    async def generate_single(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """生成单个文本"""
        return await self.generate_single_with_system(prompt, "", max_retries)

    async def generate_single_with_system(self, user_prompt: str, system_prompt: str = '', max_retries: int = 3) -> Optional[str]:
        """使用system prompt生成单个文本"""
        for attempt in range(max_retries):
            try:
                messages = []
                if system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.api_config.get('max_tokens', 1000),
                    temperature=self.api_config.get('temperature', 0.7),
                    top_p=self.api_config.get('top_p', 1.0),
                    frequency_penalty=self.api_config.get('frequency_penalty', 0.0),
                    presence_penalty=self.api_config.get('presence_penalty', 0.0)
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"OpenAI API 错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        return None


class AnthropicGenerator(LLMGenerator):
    """Anthropic Claude生成器"""

    def __init__(self, config: ExperimentConfig, engine_name: str = "claude-3-sonnet"):
        super().__init__(config, engine_name)
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.model = self.api_config.get('model', 'claude-3-sonnet-20240229')

    async def generate_single(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """生成单个文本"""
        for attempt in range(max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.api_config.get('max_tokens', 1000),
                    temperature=self.api_config.get('temperature', 0.7),
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            except Exception as e:
                logger.warning(f"Anthropic API 错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        return None


class GeminiGenerator(LLMGenerator):
    """Google Gemini生成器"""

    def __init__(self, config: ExperimentConfig, engine_name: str = "gemini-pro"):
        super().__init__(config, engine_name)
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
            self.api_config.get('model', 'gemini-pro')
        )

    async def generate_single(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """生成单个文本"""
        for attempt in range(max_retries):
            try:
                # Gemini的异步调用
                generation_config = {
                    'temperature': self.api_config.get('temperature', 0.7),
                    'top_p': self.api_config.get('top_p', 1.0),
                    'max_output_tokens': self.api_config.get('max_tokens', 1000),
                }

                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                )
                return response.text

            except Exception as e:
                logger.warning(f"Gemini API 错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        return None


class SyntheticDataGenerator:
    """Synthetic数据生成器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化生成器

        Args:
            config: 实验配置
        """
        self.config = config
        self.generators = self._initialize_generators()

    def _initialize_generators(self) -> Dict[str, LLMGenerator]:
        """初始化LLM生成器"""
        generators = {}

        for engine_name in self.config.llm_engines:
            if engine_name.startswith('gpt'):
                generators[engine_name] = OpenAIGenerator(self.config, engine_name)
            elif engine_name.startswith('claude'):
                generators[engine_name] = AnthropicGenerator(self.config, engine_name)
            elif engine_name.startswith('gemini'):
                generators[engine_name] = GeminiGenerator(self.config, engine_name)
            else:
                logger.warning(f"未知的LLM引擎: {engine_name}")

        return generators

    def create_prompt(self, prompt_template: str, subject: str, body: str) -> str:
        """
        创建完整的prompt

        Args:
            prompt_template: prompt模板
            subject: 邮件主题
            body: 邮件内容

        Returns:
            str: 完整的prompt
        """
        return prompt_template.format(subject=subject, body=body)

    async def generate_batch(
        self,
        spam_data: pd.DataFrame,
        prompt_name: str,
        llm_engine: str,
        batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        批量生成synthetic data

        Args:
            spam_data: 原始spam数据
            prompt_name: prompt名称
            llm_engine: LLM引擎名称
            batch_size: 批处理大小

        Returns:
            pd.DataFrame: 生成的synthetic data
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if llm_engine not in self.generators:
            raise ValueError(f"未找到LLM生成器: {llm_engine}")

        generator = self.generators[llm_engine]
        prompt_config = self.config.prompts[prompt_name]
        prompt_template = prompt_config['template']
        system_prompt = prompt_config.get('system_prompt', '')

        logger.info(f"开始批量生成: {len(spam_data)} 条数据, prompt={prompt_name}, llm={llm_engine}")

        results = []
        failed_count = 0

        # 创建进度条
        pbar = tqdm(total=len(spam_data), desc=f"生成 {prompt_name}")

        # 分批处理
        for i in range(0, len(spam_data), batch_size):
            batch_data = spam_data.iloc[i:i + batch_size]
            batch_tasks = []

            # 创建批次任务
            for _, row in batch_data.iterrows():
                user_prompt = self.create_prompt(prompt_template, row['subject'], row['body'])
                task = generator.generate_single_with_system(user_prompt, system_prompt, self.config.max_retries)
                batch_tasks.append((row, task))

            # 并行执行批次
            batch_results = await asyncio.gather(
                *[task for _, task in batch_tasks],
                return_exceptions=True
            )

            # 处理批次结果
            for (row, _), result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception) or result is None:
                    failed_count += 1
                    logger.warning(f"生成失败: {row['unique_id']}")
                    pbar.update(1)
                    continue

                # 解析生成结果
                parsed_result = generator.parse_response(result)

                # 检查质量
                if not self._check_quality(row, parsed_result):
                    failed_count += 1
                    logger.warning(f"质量检查失败: {row['unique_id']}")
                    pbar.update(1)
                    continue

                # 添加到结果
                result_row = {
                    'original_id': row['unique_id'],
                    'original_subject': row['subject'],
                    'original_body': row['body'],
                    'synthetic_subject': parsed_result['subject'],
                    'synthetic_body': parsed_result['body'],
                    'prompt_name': prompt_name,
                    'llm_engine': llm_engine,
                    'generation_timestamp': pd.Timestamp.now(),
                    'original_label': row['label'],
                    'group_id': row.get('group_id', -1)
                }
                results.append(result_row)
                pbar.update(1)

            # 添加延迟避免API限制
            if i + batch_size < len(spam_data):
                await asyncio.sleep(0.1)

        pbar.close()

        logger.info(f"生成完成: 成功{len(results)}, 失败{failed_count}")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _check_quality(self, original: pd.Series, generated: Dict[str, str]) -> bool:
        """
        检查生成质量

        Args:
            original: 原始数据
            generated: 生成数据

        Returns:
            bool: 质量是否合格
        """
        quality_config = self.config.quality_control

        # 检查长度比例
        orig_length = len(original['subject']) + len(original['body'])
        gen_length = len(generated['subject']) + len(generated['body'])

        if orig_length == 0:
            return False

        length_ratio = gen_length / orig_length

        min_ratio = quality_config.get('min_length_ratio', 0.3)
        max_ratio = quality_config.get('max_length_ratio', 3.0)

        if length_ratio < min_ratio or length_ratio > max_ratio:
            return False

        # 检查内容不为空
        if not generated['subject'].strip() and not generated['body'].strip():
            return False

        # 可以添加更多质量检查...

        return True

    async def generate_for_group(
        self,
        group_data: pd.DataFrame,
        prompt_name: str,
        llm_engine: str,
        output_dir: str,
        checkpoint_interval: Optional[int] = None
    ) -> str:
        """
        为单个组生成synthetic data

        Args:
            group_data: 组数据
            prompt_name: prompt名称
            llm_engine: LLM引擎
            output_dir: 输出目录
            checkpoint_interval: checkpoint间隔

        Returns:
            str: 输出文件路径
        """
        if checkpoint_interval is None:
            checkpoint_interval = self.config.save_checkpoint_every

        # 过滤spam数据
        spam_data = group_data[group_data['label'] == 1].copy()
        group_id = group_data['group_id'].iloc[0] if 'group_id' in group_data.columns else 'unknown'

        logger.info(f"为组 {group_id} 生成synthetic data: {len(spam_data)} 条spam")

        # 生成数据
        synthetic_data = await self.generate_batch(spam_data, prompt_name, llm_engine)

        if synthetic_data.empty:
            logger.warning(f"组 {group_id} 生成失败")
            return None

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = self.config.dataset.lower().replace('-', '')
        filename = f"{dataset_name}_synthetic_{prompt_name}_{llm_engine}_group_{group_id}.csv.gz"
        output_path = os.path.join(output_dir, filename)

        synthetic_data.to_csv(output_path, compression='gzip', index=False)

        logger.info(f"组 {group_id} synthetic data 保存到: {output_path}")
        return output_path

    def load_checkpoint(self, checkpoint_path: str) -> pd.DataFrame:
        """加载checkpoint"""
        if os.path.exists(checkpoint_path):
            return pd.read_csv(checkpoint_path, compression='gzip')
        return pd.DataFrame()

    def save_checkpoint(self, data: pd.DataFrame, checkpoint_path: str):
        """保存checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        data.to_csv(checkpoint_path, compression='gzip', index=False)

    async def generate_complete_dataset(
        self,
        groups_data: Dict[str, pd.DataFrame],
        prompt_name: str,
        llm_engine: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        生成完整数据集的synthetic data

        Args:
            groups_data: 分组数据
            prompt_name: prompt名称
            llm_engine: LLM引擎
            output_dir: 输出目录

        Returns:
            Dict[str, str]: 生成的文件路径
        """
        if output_dir is None:
            output_dir = self.config.synthetic_data_path

        logger.info(f"开始生成完整数据集: {len(groups_data)} 组")

        generated_files = {}
        all_synthetic_data = []

        # 为每个组生成数据
        for group_name, group_data in groups_data.items():
            try:
                file_path = await self.generate_for_group(
                    group_data, prompt_name, llm_engine, output_dir
                )
                if file_path:
                    generated_files[group_name] = file_path
                    # 加载生成的数据用于合并
                    group_synthetic = pd.read_csv(file_path, compression='gzip')
                    all_synthetic_data.append(group_synthetic)

            except Exception as e:
                logger.error(f"组 {group_name} 生成失败: {e}")
                continue

        # 合并所有生成的数据
        if all_synthetic_data:
            complete_synthetic = pd.concat(all_synthetic_data, ignore_index=True)

            # 保存完整的synthetic数据集
            dataset_name = self.config.dataset.lower().replace('-', '')
            complete_filename = f"{dataset_name}_synthetic_{prompt_name}_{llm_engine}.csv.gz"
            complete_path = os.path.join(output_dir, complete_filename)

            complete_synthetic.to_csv(complete_path, compression='gzip', index=False)
            generated_files['complete'] = complete_path

            logger.info(f"完整synthetic数据集保存到: {complete_path}")
            logger.info(f"总计生成: {len(complete_synthetic)} 条synthetic data")

        return generated_files


# 便捷函数
async def generate_synthetic_data(
    config: ExperimentConfig,
    groups_data: Dict[str, pd.DataFrame],
    prompt_name: str,
    llm_engine: str,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    便捷函数：生成synthetic data

    Args:
        config: 实验配置
        groups_data: 分组数据
        prompt_name: prompt名称
        llm_engine: LLM引擎
        output_dir: 输出目录

    Returns:
        Dict[str, str]: 生成的文件路径
    """
    generator = SyntheticDataGenerator(config)
    return await generator.generate_complete_dataset(
        groups_data, prompt_name, llm_engine, output_dir
    )