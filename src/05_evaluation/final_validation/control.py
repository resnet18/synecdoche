"""
最终测试统一控制脚本
功能：8G显存限制、计时、监控、汇总（工位电脑专用）
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

# ========== 核心配置 ==========
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_ROOT = BASE_DIR / "results" / "final_validation"
LIMIT_GB = 8  # 模拟目标显存（State of the Availability）

# 14基线全配置（已确认8G够用）
PIPELINE = [
    # 统计层（CPU，极速）
    {"name": "B0", "script": "b0.py", "type": "stats"},
    {"name": "B1", "script": "b1.py", "type": "stats"},
    {"name": "B2", "script": "b2.py", "type": "stats"},
    {"name": "B3", "script": "b3.py", "type": "stats"},
    {"name": "B4", "script": "b4.py", "type": "stats"},
    {"name": "B4P", "script": "b4_pretrained.py", "type": "stats", "skip_if_missing": "models/cc.zh.300.vec.gz"},
    {"name": "O1", "script": "o1.py", "type": "stats"},  # 欧氏距离版
    
    # 静态嵌入（GPU轻量，8G宽松）
    {"name": "B5", "script": "b5.py", "type": "embedding", "hf_mirror": True},
    {"name": "B6f", "script": "b6_frozen.py", "type": "embedding"},
    
    # 对比学习（GPU中等，8G可跑）
    {"name": "O3", "script": "o3.py", "type": "contrastive"},      # ELECTRA-small 14M
    {"name": "O2", "script": "o2.py", "type": "contrastive"},      # MiniLM SimCSE
    {"name": "O2p", "script": "o2_probe.py", "type": "contrastive"},  # 仅投影头
    {"name": "B6", "script": "b6.py", "type": "contrastive"},      # RoBERTa-large 355M（8G边界但用户确认可跑）
    
    # 本地LLM（GPU重载，8G临界但用户确认可跑）
    {"name": "B7", "script": "b7.py", "type": "llm"},  # Qwen3-4B INT4
    {"name": "B8", "script": "b8.py", "type": "llm"},  # Qwen3-8B INT4（OOM自动降级15→5→0句）
]

# ========== 显存限制核心 ==========
def setup_8g_limit():
    """计算并设置8G显存限制（通过环境变量传递）"""
    if not torch.cuda.is_available():
        print("未检测到GPU，以CPU模式运行（B0-O1正常，B5+会极慢或报错）")
        return False
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    fraction = LIMIT_GB / total
    
    if fraction >= 1.0:
        print(f"ℹ当前显存 {total:.1f}G ≤ {LIMIT_GB}G，无需限制")
        return False
    
    os.environ['CUDA_MEMORY_FRACTION'] = str(fraction)
    print(f"State of the Availability: {total:.1f}G → 强制限制 {LIMIT_GB}G (fraction={fraction:.2f})")
    print(f"子进程将通过环境变量自动继承限制")
    return True

def release_limit():
    """结束后清理（虽然进程退出自动清理，但显式删除更保险）"""
    if 'CUDA_MEMORY_FRACTION' in os.environ:
        del os.environ['CUDA_MEMORY_FRACTION']
        print("显存限制已清除")

# ========== 执行器 ==========
def run_stage(cfg, env):
    """执行单个基线"""
    name, script = cfg["name"], cfg["script"]
    script_path = Path(__file__).parent / script
    
    # 检查跳过条件
    if cfg.get("skip_if_missing"):
        missing_file = BASE_DIR / cfg["skip_if_missing"]
        if not missing_file.exists():
            print(f"\n跳过 {name}: 缺少 {cfg['skip_if_missing']}，请检查")
            return {"name": name, "status": "SKIPPED", "reason": "file_missing"}
    
    print(f"\n{'='*60}")
    print(f"{name} ({cfg['type']})")
    
    # 环境准备
    run_env = env.copy()
    if cfg.get("hf_mirror"):
        run_env['HF_ENDPOINT'] = 'https://hf-mirror.com'
        run_env['HF_HUB_OFFLINE'] = '0'
    
    # 计时开始
    t0 = time.time()
    gpu_before = torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=run_env,
            timeout=1800  # 单基线30分钟超时（LLM保险）
        )
        
        elapsed = time.time() - t0
        gpu_peak = max(gpu_before, torch.cuda.memory_allocated()/1024**2) if torch.cuda.is_available() else 0
        
        if result.returncode != 0:
            print(f"失败 ({elapsed:.1f}s)")
            print(f"错误: {result.stderr[:200]}")
            return {"name": name, "status": "FAILED", "time": elapsed, "error": result.stderr[:100]}
        
        # 读取预测结果
        pred_csv = RESULTS_ROOT / name.lower() / "predictions.csv"
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            score_col = [c for c in df.columns if 'score' in c][0]
            s = df[score_col]
            
            print(f"成功 ({elapsed:.1f}s, 峰值显存:{gpu_peak:.0f}MB)")
            print(f"   min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, std={s.std():.2f}")
            
            return {
                "name": name, "status": "OK", "time": elapsed, "gpu_mb": gpu_peak,
                "min": float(s.min()), "max": float(s.max()), 
                "mean": float(s.mean()), "std": float(s.std()),
                "samples": len(s)
            }
        else:
            print(f"无输出文件")
            return {"name": name, "status": "NO_OUTPUT", "time": elapsed}
            
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "TIMEOUT", "time": 1800}
    except Exception as e:
        return {"name": name, "status": "ERROR", "error": str(e)}

# ========== 主控 ==========
def main():
    print(f"{'='*70}")
    print("Synecdoche Control - Final Validation")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 模拟 {LIMIT_GB}G 显存环境 (State of the Availability)")
    print(f"{'='*70}")
    
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 设置8G限制（工位16G→模拟8G）
    limited = setup_8g_limit()
    base_env = os.environ.copy()
    
    # 执行全管线
    stats = []
    t_start = time.time()
    
    try:
        for cfg in PIPELINE:
            r = run_stage(cfg, base_env)
            stats.append(r)
            
            # LLM后强制清缓存（保险）
            if cfg["type"] == "llm" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        release_limit()
    
    # 生成汇总报告（类score.txt格式）
    total_time = time.time() - t_start
    
    lines = [
        "Synecdoche Final Validation Summary",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time/60:.1f}分钟 | 限制显存: {LIMIT_GB}G",
        "-" * 60
    ]
    
    for s in stats:
        if s["status"] == "OK":
            lines.append(f"{s['name']:6s} min={s['min']:6.2f}, max={s['max']:6.2f}, mean={s['mean']:6.2f}, std={s['std']:6.2f}, time={s['time']:5.1f}s")
        else:
            lines.append(f"{s['name']:6s} [{s['status']}]")
    
    lines.append("-" * 60)
    success = sum(1 for s in stats if s["status"] == "OK")
    lines.append(f"成功: {success}/{len(stats)}")
    
    # 保存
    summary_text = "\n".join(lines)
    print(f"\n{'='*70}")
    print(summary_text)
    
    with open(RESULTS_ROOT / "score.txt", 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    with open(RESULTS_ROOT / "control_stats.json", 'w') as f:
        json.dump({"stats": stats, "total_time": total_time}, f, indent=2)
    
    print(f"\n报告已保存: {RESULTS_ROOT / 'score.txt'}")

if __name__ == "__main__":
    main()