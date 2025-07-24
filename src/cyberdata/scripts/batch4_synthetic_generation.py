#!/usr/bin/env python3
"""
阶段4实验 - Phase 2: 分层种子采样和合成数据生成

基于Phase 1的分层结果，从每层采样种子并使用LLM生成合成数据。

作者: Claude
创建时间: 2025-07-24
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import concurrent.futures
import random
from typing import Dict, List, Any, Optional
import logging
from openai import OpenAI
from dotenv import load_dotenv

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

# 加载环境变量
load_dotenv()
client = OpenAI()

class Batch4SyntheticGenerator:
    """阶段4合成数据生成器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch4_phase2")
        self.project_root = PROJECT_ROOT
        self.batch4_dir = self.project_root / "data" / "batch4"
        self.stratification_dir = self.batch4_dir / "layer_stratification"
        self.synthetic_dir = self.batch4_dir / "synthetic"
        
        # 确保输出目录存在
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.layered_samples = None
        self.seeds_by_layer = {}
        self.synthetic_results = {}
        
        # 生成配置
        self.seeds_per_layer = 100  # 每层采样种子数量
        self.model_name = "gpt-4o-mini"
        self.max_workers = 10
        
        # Rewrite prompts (复用batch1的设计)
        self.prompts = {
            'rewrite': {
                'system': "You are an expert in email phishing simulation. Rewrite phishing emails to maintain the malicious intent while changing language, structure, and specific details.",
                'user': "Rewrite this phishing email. Keep the malicious intent and core attack vector, but change the wording, structure, and specific details. Return only the rewritten email in the same format.\\n\\nOriginal email:\\nSubject: {subject}\\nBody: {body}\\n\\nRewritten email:"
            },
            'rewrite_strong': {
                'system': "You are an expert in advanced phishing email generation. Create sophisticated variants that are harder to detect while maintaining the original attack intent.",
                'user': "Create a sophisticated variant of this phishing email. Make it more convincing and harder to detect while keeping the same attack goal. Use professional language and realistic scenarios.\\n\\nOriginal email:\\nSubject: {subject}\\nBody: {body}\\n\\nSophisticated variant:"
            },
            'rewrite_weak': {
                'system': "You are generating basic phishing email variants. Create simpler versions that are easier to detect while maintaining the core malicious intent.",
                'user': "Create a simpler, more obvious variant of this phishing email. Make it easier to detect with clear warning signs while keeping the same basic attack intent.\\n\\nOriginal email:\\nSubject: {subject}\\nBody: {body}\\n\\nSimpler variant:"
            }
        }
        
    def load_stratified_data(self) -> bool:
        """加载Phase 1的分层结果"""
        try:
            # 加载分层数据
            layered_file = self.stratification_dir / "layered_samples.json"
            self.logger.info(f"加载分层数据: {layered_file}")
            
            with open(layered_file, 'r', encoding='utf-8') as f:
                self.layered_samples = json.load(f)
            
            # 验证数据完整性
            total_samples = 0
            for layer_name, samples in self.layered_samples.items():
                self.logger.info(f"{layer_name}层: {len(samples)} 个样本")
                total_samples += len(samples)
            
            self.logger.info(f"总计加载 {total_samples} 个分层样本")
            return True
            
        except Exception as e:
            self.logger.error(f"加载分层数据失败: {str(e)}")
            return False
    
    def sample_seeds_from_layers(self) -> Dict[str, List[Dict]]:
        """从每层采样种子样本"""
        try:
            self.logger.info(f"从每层采样 {self.seeds_per_layer} 个种子...")
            
            self.seeds_by_layer = {}
            
            for layer_name, layer_samples in self.layered_samples.items():
                available_samples = len(layer_samples)
                
                if available_samples < self.seeds_per_layer:
                    self.logger.warning(f"{layer_name}层样本不足: {available_samples} < {self.seeds_per_layer}")
                    # 使用全部样本
                    selected_seeds = layer_samples
                else:
                    # 随机采样
                    selected_seeds = random.sample(layer_samples, self.seeds_per_layer)
                
                self.seeds_by_layer[layer_name] = selected_seeds
                self.logger.info(f"{layer_name}层采样了 {len(selected_seeds)} 个种子")
            
            # 保存种子采样结果
            seeds_file = self.synthetic_dir / "sampled_seeds.json"
            with open(seeds_file, 'w', encoding='utf-8') as f:
                json.dump(self.seeds_by_layer, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"种子采样结果保存到: {seeds_file}")
            
            return self.seeds_by_layer
            
        except Exception as e:
            self.logger.error(f"种子采样失败: {str(e)}")
            raise
    
    def call_llm(self, system_prompt, user_prompt, max_retries=3):
        """调用OpenAI API with retry logic (复用batch1逻辑)"""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
    
    def parse_email_response(self, response_text, original_subject, original_body):
        """解析LLM响应提取subject和body (复用batch1逻辑)"""
        try:
            lines = response_text.strip().split('\\n')
            subject = ""
            body = ""
            
            current_section = None
            body_lines = []
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('subject:'):
                    current_section = 'subject'
                    subject = line[8:].strip()  # Remove 'Subject:' prefix
                elif line.lower().startswith('body:'):
                    current_section = 'body'
                    body_content = line[5:].strip()  # Remove 'Body:' prefix
                    if body_content:
                        body_lines.append(body_content)
                elif current_section == 'body' and line:
                    body_lines.append(line)
            
            if body_lines:
                body = '\\n'.join(body_lines)
            
            # 如果解析失败，使用备用方法
            if not subject and not body:
                if 'Subject:' in response_text and 'Body:' in response_text:
                    parts = response_text.split('Body:', 1)
                    subject_part = parts[0].replace('Subject:', '').strip()
                    body_part = parts[1].strip() if len(parts) > 1 else ''
                    subject = subject_part
                    body = body_part
                else:
                    # 最后备用：使用原始subject，整个响应作为body
                    subject = f"Re: {original_subject}"
                    body = response_text.strip()
            
            # 确保不为空
            if not subject:
                subject = f"Re: {original_subject}"
            if not body:
                body = response_text.strip()
            
            return subject, body
            
        except Exception as e:
            self.logger.warning(f"解析响应失败: {e}, 使用备用解析")
            return f"Re: {original_subject}", response_text.strip()
    
    def generate_single_variant(self, seed_sample, variant, layer_name):
        """生成单个变体"""
        try:
            subject = seed_sample['subject']
            body = seed_sample['body']
            
            # 获取对应的prompt
            prompt_config = self.prompts[variant]
            system_prompt = prompt_config['system']
            user_prompt = prompt_config['user'].format(subject=subject, body=body)
            
            # 调用LLM
            response = self.call_llm(system_prompt, user_prompt)
            
            # 解析响应
            new_subject, new_body = self.parse_email_response(response, subject, body)
            
            # 生成新的data_id
            base_id = seed_sample['data_id'].split('_')[-1]  # 取原ID的最后部分
            new_data_id = f"CEAS08_SYN4_{layer_name.upper()}_{base_id}_{variant.upper()}"
            
            synthetic_sample = {
                'sender': 'synthetic@batch4.generated',
                'receiver': 'target@example.com', 
                'date': '2025-07-24',
                'subject': new_subject,
                'body': new_body,
                'label': 1,  # 恶意邮件
                'urls': '',
                'data_id': new_data_id,
                'metadata': {
                    'layer': layer_name,
                    'seed_id': seed_sample['data_id'],
                    'seed_distance': seed_sample['distance'],
                    'variant': variant,
                    'generation_success': True
                }
            }
            
            return {'success': True, 'sample': synthetic_sample}
            
        except Exception as e:
            self.logger.error(f"生成变体失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_layer_variants(self, layer_name, seeds):
        """为某层生成所有变体"""
        try:
            self.logger.info(f"开始为{layer_name}层生成合成数据...")
            
            # 准备任务列表
            tasks = []
            for seed_idx, seed_sample in enumerate(seeds):
                for variant in ['rewrite', 'rewrite_strong', 'rewrite_weak']:
                    tasks.append((seed_sample, variant, layer_name))
            
            self.logger.info(f"{layer_name}层: {len(tasks)} 个生成任务")
            
            # 并发执行生成任务
            results = []
            successful = 0
            failed = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.generate_single_variant, seed, variant, layer_name): (seed, variant)
                    for seed, variant, layer_name in tasks
                }
                
                for future in concurrent.futures.as_completed(future_to_task):
                    result = future.result()
                    if result['success']:
                        results.append(result['sample'])
                        successful += 1
                    else:
                        failed += 1
                        
                    # 进度报告
                    if (successful + failed) % 50 == 0:
                        self.logger.info(f"{layer_name}层进度: {successful + failed}/{len(tasks)} 完成")
            
            self.logger.info(f"{layer_name}层完成: 成功 {successful}, 失败 {failed}")
            return results
            
        except Exception as e:
            self.logger.error(f"{layer_name}层生成失败: {str(e)}")
            return []
    
    def save_synthetic_data(self) -> None:
        """保存合成数据到CSV文件"""
        try:
            self.logger.info("保存合成数据到CSV文件...")
            
            total_samples = 0
            
            for layer_name, samples in self.synthetic_results.items():
                if not samples:
                    self.logger.warning(f"{layer_name}层没有成功生成的样本")
                    continue
                
                # 转换为DataFrame
                df_data = []
                for sample in samples:
                    # 移除metadata用于CSV存储
                    csv_sample = {k: v for k, v in sample.items() if k != 'metadata'}
                    df_data.append(csv_sample)
                
                df = pd.DataFrame(df_data)
                
                # 保存CSV文件
                csv_file = self.synthetic_dir / f"{layer_name}_layer_synthetic.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8')
                
                self.logger.info(f"{layer_name}层合成数据保存到: {csv_file} ({len(df)} 个样本)")
                total_samples += len(df)
            
            # 保存完整的metadata信息
            metadata_file = self.synthetic_dir / "generation_metadata.json"
            generation_metadata = {
                'generation_date': '2025-07-24',
                'total_samples': total_samples,
                'seeds_per_layer': self.seeds_per_layer,
                'variants': ['rewrite', 'rewrite_strong', 'rewrite_weak'],
                'model_name': self.model_name,
                'layer_results': {}
            }
            
            for layer_name, samples in self.synthetic_results.items():
                variants_count = {}
                for sample in samples:
                    variant = sample['metadata']['variant']
                    variants_count[variant] = variants_count.get(variant, 0) + 1
                
                generation_metadata['layer_results'][layer_name] = {
                    'sample_count': len(samples),
                    'variants_distribution': variants_count
                }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(generation_metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"生成元数据保存到: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"保存合成数据失败: {str(e)}")
            raise
    
    def run_phase2_complete(self) -> bool:
        """执行完整的Phase 2流程"""
        try:
            self.logger.info("="*50)
            self.logger.info("开始执行阶段4 Phase 2: 分层种子采样和合成数据生成")
            self.logger.info("="*50)
            
            # Step 1: 加载分层数据
            self.logger.info("Step 1: 加载Phase 1分层结果")
            if not self.load_stratified_data():
                return False
            
            # Step 2: 种子采样
            self.logger.info("Step 2: 从各层采样种子")
            self.sample_seeds_from_layers()
            
            # Step 3: 分层生成合成数据
            self.logger.info("Step 3: 分层生成合成数据")
            self.synthetic_results = {}
            
            for layer_name, seeds in self.seeds_by_layer.items():
                layer_results = self.generate_layer_variants(layer_name, seeds)
                self.synthetic_results[layer_name] = layer_results
            
            # Step 4: 保存结果
            self.logger.info("Step 4: 保存合成数据")
            self.save_synthetic_data()
            
            # 最终报告
            self.logger.info("="*50)
            self.logger.info("阶段4 Phase 2 执行完成!")
            total_samples = sum(len(samples) for samples in self.synthetic_results.values())
            self.logger.info(f"总计生成 {total_samples} 个合成样本")
            for layer_name, samples in self.synthetic_results.items():
                self.logger.info(f"- {layer_name}层: {len(samples)} 个样本")
            self.logger.info("="*50)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 2执行失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 设置日志
    logger = setup_logger("batch4_phase2", log_level=logging.INFO)
    
    try:
        # 创建生成器
        generator = Batch4SyntheticGenerator(logger)
        
        # 执行Phase 2
        success = generator.run_phase2_complete()
        
        if success:
            logger.info("阶段4 Phase 2 successfully completed!")
            return 0
        else:
            logger.error("阶段4 Phase 2 failed!")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())