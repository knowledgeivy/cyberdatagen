# src/cyberdata/scripts/run_batch3_complete.py

"""
阶段3完整执行脚本
运行距离临界值分析的完整流程：数据分析 -> 可视化 -> 报告生成
"""

import sys
import time
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
if 'src/cyberdata/scripts' in str(current_file):
    PROJECT_ROOT = current_file.parents[3]
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from cyberdata.scripts.batch3_distance_analysis import Batch3DistanceAnalyzer
from cyberdata.scripts.batch3_visualize_results import Batch3Visualizer

def run_complete_batch3_analysis():
    """运行完整的batch3分析流程"""
    print("=" * 60)
    print("BATCH 3: 距离临界值分析 - 完整流程")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Stage 1: 距离分析实验
        print("\\n🔬 阶段1: 执行距离分析实验...")
        analyzer = Batch3DistanceAnalyzer()
        analysis_results = analyzer.run_full_analysis()
        
        if analysis_results is None:
            print("❌ 距离分析失败，终止流程")
            return False
        
        print(f"✅ 距离分析完成，测试了 {analysis_results['successful_experiments']} 个距离区间")
        
        # Stage 2: 结果可视化
        print("\\n📊 阶段2: 生成可视化和分析报告...")
        visualizer = Batch3Visualizer()
        visualizer.run_full_visualization()
        
        print("✅ 可视化分析完成")
        
        # Stage 3: 总结
        elapsed_time = time.time() - start_time
        print("\\n" + "=" * 60)
        print("🎉 BATCH 3 分析流程完成!")
        print("=" * 60)
        print(f"⏱️  总耗时: {elapsed_time/60:.1f} 分钟")
        print(f"📁 结果目录: {PROJECT_ROOT}/data/batch3/")
        print(f"📊 可视化: {PROJECT_ROOT}/data/batch3/analysis/")
        
        # 显示关键发现
        if analysis_results['critical_points']:
            print(f"🔍 发现 {len(analysis_results['critical_points'])} 个临界点:")
            for i, cp in enumerate(analysis_results['critical_points'], 1):
                dist_min, dist_max = cp['distance_range']
                print(f"   {i}. 距离 [{dist_min:.2f}, {dist_max:.2f}) - {cp['model']} 下降 {cp['performance_drop']*100:.1f}%")
        else:
            print("🔍 未发现显著临界点 (性能下降<5%)")
        
        print("\\n📋 建议下一步:")
        print("   1. 查看可视化结果: batch3_performance_curves.png")
        print("   2. 阅读分析报告: batch3_summary_report.md")
        print("   3. 检查交互式图表: batch3_interactive_analysis.html")
        
        return True
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = run_complete_batch3_analysis()
    
    if success:
        print("\\n✨ 阶段3分析成功完成！")
        return 0
    else:
        print("\\n💥 阶段3分析执行失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)