#!/usr/bin/env python3
"""
批次4实验 - Phase 2: 分层合成数据生成

基于Phase 1的4层分层数据，使用3个prompt变体生成合成数据。
每层100个种子 × 3个prompt = 1,200次API调用

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import openai
from dotenv import load_dotenv

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger
from src.cyberdata.utils.data_loader import load_csv_data, save_csv_data
from src.cyberdata.utils.config_manager import ConfigManager

# 加载环境变量
load_dotenv()

class Batch4Phase2Generator:
    """批次4 Phase 2: 分层合成数据生成器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase2")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4_fresh"
        
        # 创建输出目录
        self.synthetic_dir = self.batch4_dir / "synthetic"
        self.analysis_dir = self.batch4_dir / "phase2_analysis"
        
        for dir_path in [self.synthetic_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config_manager = ConfigManager()
        self.setup_openai_client()
        
        # 实验参数
        self.seeds_per_layer = 100  # 每层采样100个种子
        self.prompt_variants = ['rewrite', 'rewrite_strong', 'rewrite_weak']
        self.layers = ['core', 'inner', 'outer', 'edge']
        
        # 并发控制
        self.max_workers = 5
        self.api_delay = 0.1  # API调用间隔
        self.progress_lock = Lock()
        self.progress_counter = 0
        self.total_calls = len(self.layers) * self.seeds_per_layer * len(self.prompt_variants)
        
        # 随机种子
        self.random_state = 2025
        
    def setup_openai_client(self):
        """设置OpenAI客户端"""
        try:
            # 从环境变量获取API密钥
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY环境变量未设置")
            
            openai.api_key = api_key
            self.model_name = 'gpt-4.1-mini'  # 使用新的高效模型
            
            self.logger.info(f"OpenAI客户端初始化完成，模型: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            raise
    
    def load_stratified_layers(self) -> Dict[str, pd.DataFrame]:
        """加载分层数据"""
        try:
            self.logger.info("加载分层数据...")
            
            stratified_dir = self.batch4_dir / "stratified_layers"
            layers_data = {}
            
            for layer in self.layers:
                layer_file = stratified_dir / f"{layer}_samples.csv"
                df = load_csv_data(layer_file, logger=self.logger)
                layers_data[layer] = df
                self.logger.info(f"加载{layer}层: {len(df)} 个样本")
            
            return layers_data
            
        except Exception as e:
            self.logger.error(f"加载分层数据失败: {str(e)}")
            raise
    
    def get_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """获取3个prompt变体模板"""
        return {
            'rewrite': {
                'system': "You are an expert in email phishing simulation. Rewrite phishing emails to maintain the malicious intent while changing language, structure, and specific details.",
                'user': "Rewrite this phishing email. Keep the malicious intent and core attack vector, but change the wording, structure, and specific details. Return only the rewritten email in the same format.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nRewritten email:"
            },
            'rewrite_strong': {
                'system': "You are an expert in advanced phishing email generation. Create sophisticated variants that are harder to detect while maintaining the original attack intent.",
                'user': "Create a sophisticated variant of this phishing email. Make it more convincing and harder to detect while keeping the same attack goal. Use professional language and realistic scenarios.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nSophisticated variant:"
            },
            'rewrite_weak': {
                'system': "You are generating basic phishing email variants. Create simpler versions that are easier to detect while maintaining the core malicious intent.",
                'user': "Create a simpler, more obvious variant of this phishing email. Make it easier to detect with clear warning signs while keeping the same basic attack intent.\n\nOriginal email:\nSubject: {subject}\nBody: {body}\n\nSimpler variant:"
            }
        }
    
    def sample_seeds_from_layer(self, layer_df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
        """从层中采样种子样本"""
        try:
            n_seeds = min(self.seeds_per_layer, len(layer_df))
            
            # 随机采样
            seeds = layer_df.sample(n=n_seeds, random_state=self.random_state)
            
            self.logger.info(f"从{layer_name}层采样 {n_seeds} 个种子")
            
            return seeds
            
        except Exception as e:
            self.logger.error(f"从{layer_name}层采样失败: {str(e)}")
            raise
    
    def call_openai_api(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """调用OpenAI API"""
        for attempt in range(max_retries):
            try:
                time.sleep(self.api_delay)  # API调用限制
                
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"API调用失败 (最终尝试): {str(e)}")
                    raise
                else:
                    self.logger.warning(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2 ** attempt)  # 指数退避
    
    def parse_email_response(self, response_text: str, original_subject: str) -> Tuple[str, str]:
        """解析API响应，提取subject和body"""
        try:
            # 尝试多种解析方式
            response_text = response_text.strip()
            
            # 方式1: 查找Subject和Body标记
            if "Subject:" in response_text and "Body:" in response_text:
                lines = response_text.split('\n')
                subject = ""
                body = ""
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Subject:"):
                        subject = line.replace("Subject:", "").strip()
                        current_section = "subject"
                    elif line.startswith("Body:"):
                        current_section = "body"
                        body_content = line.replace("Body:", "").strip()
                        if body_content:
                            body = body_content
                    elif current_section == "body" and line:
                        if body:
                            body += "\n" + line
                        else:
                            body = line
                
                if subject and body:
                    return subject, body
            
            # 方式2: 假设第一行是subject，其余是body
            lines = response_text.split('\n')
            if len(lines) >= 2:
                subject = lines[0].replace("Subject:", "").strip()
                body = '\n'.join(lines[1:]).strip()
                
                if subject and body:
                    return subject, body
            
            # 方式3: 使用原始subject，整个响应作为body
            return f"Re: {original_subject}", response_text
            
        except Exception as e:
            self.logger.warning(f"解析响应失败: {e}")
            return f"Re: {original_subject}", response_text
    
    def generate_single_synthetic(self, seed_row: pd.Series, layer_name: str, 
                                prompt_variant: str, seed_index: int) -> Dict[str, Any]:
        """生成单个合成样本"""
        try:
            # 获取原始数据
            original_subject = seed_row['subject']
            original_body = seed_row['body']
            
            # 获取prompt模板
            prompts = self.get_prompt_templates()
            prompt_config = prompts[prompt_variant]
            
            # 构建prompt
            system_prompt = prompt_config['system']
            user_prompt = prompt_config['user'].format(
                subject=original_subject,
                body=original_body
            )
            
            # 调用API
            response = self.call_openai_api(system_prompt, user_prompt)
            
            # 解析响应
            new_subject, new_body = self.parse_email_response(response, original_subject)
            
            # 更新进度
            with self.progress_lock:
                self.progress_counter += 1
                if self.progress_counter % 50 == 0:
                    progress_pct = (self.progress_counter / self.total_calls) * 100
                    self.logger.info(f"API调用进度: {self.progress_counter}/{self.total_calls} ({progress_pct:.1f}%)")
            
            # 构建合成样本
            synthetic_sample = {
                'unique_id': f"synth_{layer_name}_{seed_index:03d}_{prompt_variant}",
                'original_id': seed_row['unique_id'],
                'layer': layer_name,
                'prompt_variant': prompt_variant,
                'seed_index': seed_index,
                'subject': new_subject,
                'body': new_body,
                'text': new_subject + " " + new_body,
                'label': 1,  # 恶意标签
                'sender': f"synthetic_{layer_name}@phishing.com",
                'receiver': "target@company.com",
                'date': "2025-07-24",
                'urls': 0,
                'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'original_subject': original_subject,
                'original_body': original_body
            }
            
            return synthetic_sample
            
        except Exception as e:
            self.logger.error(f"生成合成样本失败 ({layer_name}, {prompt_variant}, {seed_index}): {str(e)}")
            raise
    
    def generate_layer_synthetic_data(self, layer_name: str, seeds_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """为单个层生成所有合成数据"""
        try:
            self.logger.info(f"开始生成{layer_name}层合成数据...")
            
            synthetic_samples = []
            
            # 创建所有任务
            tasks = []
            for seed_idx, (_, seed_row) in enumerate(seeds_df.iterrows()):
                for prompt_variant in self.prompt_variants:
                    tasks.append((seed_row, layer_name, prompt_variant, seed_idx))
            
            self.logger.info(f"{layer_name}层: 准备并发执行 {len(tasks)} 个API调用任务")
            
            # 使用并发处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(self.generate_single_synthetic, *task): task 
                    for task in tasks
                }
                
                # 收集结果
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        synthetic_sample = future.result()
                        synthetic_samples.append(synthetic_sample)
                    except Exception as e:
                        self.logger.error(f"跳过失败的样本: {task[1]}, {task[2]}, {task[3]} - {str(e)}")
                        continue
            
            self.logger.info(f"{layer_name}层生成完成: {len(synthetic_samples)} 个合成样本")
            return synthetic_samples
            
        except Exception as e:
            self.logger.error(f"生成{layer_name}层合成数据失败: {str(e)}")
            raise
    
    def generate_all_synthetic_data(self, layers_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """生成所有层的合成数据"""
        try:
            self.logger.info("开始生成所有层的合成数据...")
            self.logger.info(f"预计API调用次数: {self.total_calls}")
            
            all_synthetic_data = {}
            
            for layer_name in self.layers:
                # 采样种子
                layer_df = layers_data[layer_name]
                seeds_df = self.sample_seeds_from_layer(layer_df, layer_name)
                
                # 生成合成数据
                synthetic_samples = self.generate_layer_synthetic_data(layer_name, seeds_df)
                all_synthetic_data[layer_name] = synthetic_samples
            
            return all_synthetic_data
            
        except Exception as e:
            self.logger.error(f"生成所有合成数据失败: {str(e)}")
            raise
    
    def save_synthetic_data(self, all_synthetic_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """保存合成数据"""
        try:
            self.logger.info("保存合成数据...")
            
            for layer_name, synthetic_samples in all_synthetic_data.items():
                if not synthetic_samples:
                    self.logger.warning(f"{layer_name}层没有合成数据")
                    continue
                
                # 转换为DataFrame
                df = pd.DataFrame(synthetic_samples)
                
                # 保存CSV文件
                output_file = self.synthetic_dir / f"{layer_name}_synthetic.csv"
                save_csv_data(df, output_file, compress=False, logger=self.logger)
                
                self.logger.info(f"保存{layer_name}层合成数据: {len(df)} 个样本")
            
        except Exception as e:
            self.logger.error(f"保存合成数据失败: {str(e)}")
            raise
    
    def generate_generation_summary(self, all_synthetic_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """生成合成数据统计摘要"""
        try:
            self.logger.info("生成合成数据统计摘要...")
            
            summary = {
                'generation_info': {
                    'date': '2025-07-24',
                    'model': self.model_name,
                    'seeds_per_layer': self.seeds_per_layer,
                    'prompt_variants': self.prompt_variants,
                    'total_api_calls': self.progress_counter,
                    'expected_api_calls': self.total_calls
                },
                'layer_statistics': {},
                'prompt_statistics': {},
                'generation_success_rate': self.progress_counter / self.total_calls if self.total_calls > 0 else 0,
                'id_format': {
                    'pattern': 'synth_{layer}_{seed_index:03d}_{prompt_variant}',
                    'example': 'synth_core_001_rewrite'
                }
            }
            
            # 统计各层数据
            for layer_name, synthetic_samples in all_synthetic_data.items():
                layer_stats = {
                    'sample_count': len(synthetic_samples),
                    'seeds_used': len(set(sample['seed_index'] for sample in synthetic_samples)),
                    'variants_per_seed': len(self.prompt_variants)
                }
                summary['layer_statistics'][layer_name] = layer_stats
            
            # 统计各prompt变体
            for variant in self.prompt_variants:
                variant_count = sum(
                    len([s for s in samples if s['prompt_variant'] == variant])
                    for samples in all_synthetic_data.values()
                )
                summary['prompt_statistics'][variant] = variant_count
            
            # 保存摘要
            summary_file = self.analysis_dir / "generation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"合成数据统计摘要已保存: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"生成统计摘要失败: {str(e)}")
            raise
    
    def run_phase2_complete(self) -> bool:
        """执行完整的Phase 2流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行批次4 Phase 2: 分层合成数据生成")
            self.logger.info("="*50)
            
            start_time = time.time()
            
            # Step 1: 加载分层数据
            self.logger.info("Step 1: 加载分层数据")
            layers_data = self.load_stratified_layers()
            
            # Step 2: 生成所有合成数据
            self.logger.info("Step 2: 生成所有合成数据")
            all_synthetic_data = self.generate_all_synthetic_data(layers_data)
            
            # Step 3: 保存合成数据
            self.logger.info("Step 3: 保存合成数据")
            self.save_synthetic_data(all_synthetic_data)
            
            # Step 4: 生成统计摘要
            self.logger.info("Step 4: 生成统计摘要")
            self.generate_generation_summary(all_synthetic_data)
            
            elapsed_time = time.time() - start_time
            
            self.logger.info("="*50)
            self.logger.info("批次4 Phase 2 执行完成!")
            self.logger.info(f"总耗时: {elapsed_time/60:.1f} 分钟")
            self.logger.info(f"API调用次数: {self.progress_counter}/{self.total_calls}")
            
            total_samples = sum(len(samples) for samples in all_synthetic_data.values())
            self.logger.info(f"生成合成样本总数: {total_samples}")
            
            for layer_name, samples in all_synthetic_data.items():
                self.logger.info(f"- {layer_name}层: {len(samples)} 个样本")
            
            self.logger.info("合成数据已保存到: data/batch4_fresh/synthetic/")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 2执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(2025)
    np.random.seed(2025)
    
    # 设置日志
    logger = setup_logger("batch4_phase2", log_level=logging.INFO)
    
    try:
        # 创建生成器
        generator = Batch4Phase2Generator(logger)
        
        # 执行Phase 2
        success = generator.run_phase2_complete()
        
        if success:
            logger.info("批次4 Phase 2 successfully completed!")
            return 0
        else:
            logger.error("批次4 Phase 2 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())