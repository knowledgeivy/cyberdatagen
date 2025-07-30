#!/usr/bin/env python3
"""
批次5实验 - 一键完整执行脚本

按顺序执行批次5所有阶段：
1. Phase 1: 统一Embedding空间构建
2. Phase 2: 增强可视化分析  
3. Phase 3: 纯净数据集重构
4. Phase 4: 模型训练与独立评估

解决批次4数据混合污染问题，建立纯净数据源的科学对比基础

作者: Claude
创建时间: 2025-07-30
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging

# 项目路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cyberdata.utils.logger_config import setup_logger

class Batch5CompleteRunner:
    """批次5完整执行器"""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_logger("batch5_complete")
        self.project_root = PROJECT_ROOT
        self.scripts_dir = self.project_root / "src" / "cyberdata" / "scripts"
        
        # 执行阶段配置
        self.phases = [
            {
                'name': 'Phase 1: 统一Embedding空间构建',
                'script': 'batch5_phase1_unified_embedding.py',
                'description': '为所有数据建立统一的embedding表示空间'
            },
            {
                'name': 'Phase 2: 增强可视化分析',
                'script': 'batch5_phase2_enhanced_visualization.py', 
                'description': '全面可视化不同数据源在embedding空间的分布'
            },
            {
                'name': 'Phase 3: 纯净数据集重构',
                'script': 'batch5_phase3_pure_dataset_construction.py',
                'description': '构建严格分离的纯净训练数据集'
            },
            {
                'name': 'Phase 4: 模型训练与独立评估',
                'script': 'batch5_phase4_independent_evaluation.py',
                'description': '独立评估不同纯净数据源的检测性能'
            }
        ]
        
        self.logger.info("批次5完整执行器初始化完成")
        self.logger.info(f"将按顺序执行 {len(self.phases)} 个阶段")
    
    def run_single_phase(self, phase_config: dict) -> bool:
        """执行单个阶段"""
        try:
            phase_name = phase_config['name']
            script_name = phase_config['script']
            script_path = self.scripts_dir / script_name
            
            self.logger.info("="*60)
            self.logger.info(f"开始执行: {phase_name}")
            self.logger.info(f"脚本: {script_name}")
            self.logger.info(f"描述: {phase_config['description']}")
            self.logger.info("="*60)
            
            if not script_path.exists():
                raise FileNotFoundError(f"脚本文件不存在: {script_path}")
            
            # 执行脚本
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, str(script_path)
            ], cwd=str(self.project_root), capture_output=True, text=True)
            
            elapsed_time = time.time() - start_time
            
            # 检查执行结果
            if result.returncode == 0:
                self.logger.info(f"✅ {phase_name} 执行成功!")
                self.logger.info(f"⏱️ 耗时: {elapsed_time/60:.1f} 分钟")
                
                # 输出脚本的关键信息
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    # 只显示最后几行重要信息
                    for line in stdout_lines[-10:]:
                        if any(keyword in line for keyword in ['成功', '完成', '保存', '✅', '总计', '样本']):
                            self.logger.info(f"📋 {line}")
                
                return True
            else:
                self.logger.error(f"❌ {phase_name} 执行失败!")
                self.logger.error(f"返回码: {result.returncode}")
                
                if result.stderr:
                    self.logger.error("错误信息:")
                    for line in result.stderr.strip().split('\n')[-5:]:  # 显示最后5行错误
                        self.logger.error(f"  {line}")
                
                if result.stdout:
                    self.logger.info("输出信息:")
                    for line in result.stdout.strip().split('\n')[-5:]:  # 显示最后5行输出
                        self.logger.info(f"  {line}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"执行阶段 {phase_config['name']} 失败: {str(e)}")
            return False
    
    def run_all_phases(self) -> bool:
        """执行所有阶段"""
        try:
            self.logger.info("🚀 开始执行批次5完整实验流程")
            self.logger.info(f"📅 执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            overall_start_time = time.time()
            successful_phases = 0
            phase_results = []
            
            for i, phase_config in enumerate(self.phases, 1):
                self.logger.info(f"\n📍 阶段 {i}/{len(self.phases)}")
                
                phase_start = time.time()
                success = self.run_single_phase(phase_config)
                phase_time = time.time() - phase_start
                
                phase_results.append({
                    'phase_name': phase_config['name'],
                    'success': success,
                    'duration': phase_time
                })
                
                if success:
                    successful_phases += 1
                    self.logger.info(f"✅ 阶段 {i} 完成 ({phase_time/60:.1f} 分钟)")
                else:
                    self.logger.error(f"❌ 阶段 {i} 失败")
                    
                    # 询问是否继续
                    user_continue = input(f"\n阶段 {i} 失败，是否继续执行下一阶段？(y/N): ").strip().lower()
                    if user_continue not in ['y', 'yes']:
                        self.logger.info("用户选择终止执行")
                        break
            
            overall_time = time.time() - overall_start_time
            
            # 输出总结
            self.logger.info("\n" + "="*60)
            self.logger.info("🎯 批次5实验执行总结")
            self.logger.info("="*60)
            self.logger.info(f"📊 总体结果: {successful_phases}/{len(self.phases)} 个阶段成功")
            self.logger.info(f"⏱️ 总耗时: {overall_time/60:.1f} 分钟")
            self.logger.info(f"📅 完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            self.logger.info("\n各阶段执行结果:")
            for i, result in enumerate(phase_results, 1):
                status = "✅ 成功" if result['success'] else "❌ 失败"
                duration = f"{result['duration']/60:.1f}分钟"
                self.logger.info(f"  {i}. {result['phase_name']}: {status} ({duration})")
            
            if successful_phases == len(self.phases):
                self.logger.info("\n🎉 批次5实验完整执行成功!")
                self.logger.info("📂 结果位置:")
                self.logger.info("  - 统一embedding: data/batch5/unified_embeddings/")
                self.logger.info("  - 可视化分析: data/batch5/visualizations/")
                self.logger.info("  - 纯净数据集: data/batch5/pure_datasets/")
                self.logger.info("  - 性能评估: data/batch5/results/")
                self.logger.info("  - 训练模型: data/batch5/models/")
            else:
                self.logger.warning(f"\n⚠️ 部分阶段执行失败 ({successful_phases}/{len(self.phases)})")
                self.logger.info("请检查失败阶段的日志并重新执行相应脚本")
            
            self.logger.info("="*60)
            
            return successful_phases == len(self.phases)
            
        except Exception as e:
            self.logger.error(f"批次5完整执行失败: {str(e)}")
            return False
    
    def check_prerequisites(self) -> bool:
        """检查执行前提条件"""
        try:
            self.logger.info("检查执行前提条件...")
            
            # 检查批次4数据是否存在
            batch4_dir = self.project_root / "data" / "batch4_fresh"
            required_files = [
                "train_malicious.csv",
                "train_benign.csv.gz", 
                "raw_test_set.csv",
                "synthetic/core_synthetic.csv",
                "synthetic/inner_synthetic.csv",
                "synthetic/outer_synthetic.csv",
                "synthetic/edge_synthetic.csv"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (batch4_dir / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error("批次4数据文件缺失:")
                for file_path in missing_files:
                    self.logger.error(f"  - {file_path}")
                return False
            
            # 检查脚本文件是否存在
            missing_scripts = []
            for phase_config in self.phases:
                script_path = self.scripts_dir / phase_config['script']
                if not script_path.exists():
                    missing_scripts.append(phase_config['script'])
            
            if missing_scripts:
                self.logger.error("批次5脚本文件缺失:")
                for script in missing_scripts:
                    self.logger.error(f"  - {script}")
                return False
            
            self.logger.info("✅ 前提条件检查通过")
            return True
            
        except Exception as e:
            self.logger.error(f"前提条件检查失败: {str(e)}")
            return False

def main():
    """主函数"""
    # 设置日志
    logger = setup_logger("batch5_complete", log_level=logging.INFO)
    
    try:
        print("🔬 批次5实验 - 纯净数据有效性验证")
        print("解决批次4数据混合污染问题的科学实验重设计")
        print("-" * 60)
        
        # 创建执行器
        runner = Batch5CompleteRunner(logger)
        
        # 检查前提条件
        if not runner.check_prerequisites():
            logger.error("前提条件不满足，无法执行批次5实验")
            return 1
        
        # 询问用户确认
        print("\n即将执行以下阶段:")
        for i, phase in enumerate(runner.phases, 1):
            print(f"  {i}. {phase['name']}")
            print(f"     {phase['description']}")
        
        user_confirm = input(f"\n是否开始执行批次5完整实验？(y/N): ").strip().lower()
        if user_confirm not in ['y', 'yes']:
            logger.info("用户取消执行")
            return 0
        
        # 执行所有阶段
        success = runner.run_all_phases()
        
        if success:
            logger.info("批次5实验完整执行成功!")
            return 0
        else:
            logger.error("批次5实验执行未完全成功")
            return 1
            
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        return 1
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())