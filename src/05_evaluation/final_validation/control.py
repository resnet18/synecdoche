#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终测试统一控制脚本（UTF-8编码版）
功能：8G显存限制（stats/embedding）、LLM无限制、计时、监控、汇总
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

# ========== 路径配置 ==========
SCRIPT_DIR = Path(__file__).resolve().parent

def find_project_root(start_path):
    """向上查找包含 results 和 src 的目录作为项目根目录"""
    for parent in [start_path] + list(start_path.parents):
        if (parent / "results").exists() and (parent / "src").exists():
            return parent
    return start_path.parents[3]

PROJECT_ROOT = find_project_root(SCRIPT_DIR)
RESULTS_ROOT = PROJECT_ROOT / "results" / "final_validation"

print(f"脚本目录: {SCRIPT_DIR}")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"结果目录: {RESULTS_ROOT}")

# ========== 显存限制 ==========
LIMIT_GB = 8

# ========== 基线配置 ==========
PIPELINE = [
    {"name": "B0", "script": "b0.py", "type": "stats"},
    {"name": "B1", "script": "b1.py", "type": "stats"},
    {"name": "B2", "script": "b2.py", "type": "stats"},
    {"name": "B3", "script": "b3.py", "type": "stats"},
    {"name": "B4", "script": "b4.py", "type": "stats"},
    {"name": "B4P", "script": "b4_pretrained.py", "type": "stats",
     "skip_if_missing": "models/cc.zh.300.vec.gz"},
    {"name": "O1", "script": "o1.py", "type": "stats"},
    {"name": "B5", "script": "b5.py", "type": "embedding"},
    {"name": "B6f", "script": "b6_frozen.py", "type": "embedding"},
    {"name": "O3", "script": "o3.py", "type": "embedding"},
    {"name": "O2", "script": "o2.py", "type": "contrastive"},
    {"name": "O2p", "script": "o2_probe.py", "type": "contrastive"},
    {"name": "B6", "script": "b6.py", "type": "contrastive"},
    {"name": "B7_int4", "script": "b7_int4.py", "type": "llm", "output_dir": "b7_int4"},
    {"name": "B7_fp16", "script": "b7_fp16.py", "type": "llm", "output_dir": "b7_fp16"},
    {"name": "B8", "script": "b8.py", "type": "llm", "output_dir": "b8"},
]

# ========== 超时配置 ==========
TIMEOUT_MAP = {"stats": 300, "embedding": 600, "contrastive": 1200, "llm": 3600}
DEFAULT_TIMEOUT = 1800

# ========== 辅助函数 ==========
def setup_8g_limit():
    if not torch.cuda.is_available():
        print("未检测到GPU，将以CPU模式运行")
        return False
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    fraction = LIMIT_GB / total_gb
    if fraction >= 1.0:
        print(f"当前显存 {total_gb:.1f}G ≤ {LIMIT_GB}G，无需限制")
        return False
    os.environ['CUDA_MEMORY_FRACTION'] = str(fraction)
    print(f"显存限制: {total_gb:.1f}G → 最多使用 {LIMIT_GB}G (比例 {fraction:.2f})")
    print("   (注意：LLM任务不会继承此限制，由子进程自行管理)")
    return True

def get_peak_memory_from_child():
    peak_file = SCRIPT_DIR / "_gpu_peak_temp.txt"
    if not peak_file.exists():
        return 0
    try:
        peak_mb = float(peak_file.read_text(encoding='utf-8').strip())
        peak_file.unlink()
        return peak_mb
    except:
        return 0

def run_stage(cfg, base_env):
    name = cfg["name"]
    script = cfg["script"]
    script_path = SCRIPT_DIR / script
    task_type = cfg.get("type", "unknown")

    if not script_path.exists():
        return {"name": name, "status": "MISSING", "error": f"脚本不存在: {script_path}"}

    if "skip_if_missing" in cfg:
        missing_file = PROJECT_ROOT / cfg["skip_if_missing"]
        if not missing_file.exists():
            print(f"\n跳过 {name}: 缺少文件 {cfg['skip_if_missing']}")
            return {"name": name, "status": "SKIPPED", "reason": "file_missing"}

    print(f"\n{'='*60}")
    print(f"开始运行 {name} ({task_type})")

    run_env = base_env.copy()
    run_env['PYTHONIOENCODING'] = 'utf-8'
    temp_peak_file = SCRIPT_DIR / "_gpu_peak_temp.txt"
    run_env['PEAK_FILE_PATH'] = str(temp_peak_file)

    # LLM 任务：移除父进程限制（由子进程自行硬编码）
    if task_type == "llm" and 'CUDA_MEMORY_FRACTION' in run_env:
        del run_env['CUDA_MEMORY_FRACTION']
        print("   (LLM任务：子进程自行管理显存)")

    if temp_peak_file.exists():
        temp_peak_file.unlink()

    timeout = cfg.get("timeout", TIMEOUT_MAP.get(task_type, DEFAULT_TIMEOUT))
    t0 = time.time()
    log_file = SCRIPT_DIR / f"_subprocess_{name}.log"
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f_log:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=run_env,
                timeout=timeout
            )
            f_log.write(result.stdout)
    except Exception as e:
        return {"name": name, "status": "FAILED", "time": time.time()-t0, "error": str(e)}

    elapsed = time.time() - t0
    gpu_peak_mb = get_peak_memory_from_child()

    child_output = ""
    if log_file.exists():
        child_output = log_file.read_text(encoding='utf-8', errors='replace')

    # ===== 关键修复：严格区分成功/失败路径 =====
    if result.returncode != 0:
        print(f"失败 ({elapsed:.1f}s)")
        
        # 检测 OOM（只在失败时）
        is_oom = ("out of memory" in child_output.lower() or 
                  "cuda out of memory" in child_output.lower())
        if is_oom:
            return {"name": name, "status": "OOM", "time": elapsed, "error": "out of memory"}
        
        # 普通失败
        return {"name": name, "status": "FAILED", "time": elapsed, 
                "error": child_output[-500:]}

    # 成功路径（查找预测文件）
    output_dir = cfg.get("output_dir", Path(script).stem)
    pred_csv = RESULTS_ROOT / output_dir / "predictions.csv"
    
    if not pred_csv.exists():
        # 尝试备选路径
        alt_path = RESULTS_ROOT / name.lower() / "predictions.csv"
        if alt_path.exists():
            pred_csv = alt_path
    
    if pred_csv.exists():
        try:
            df = pd.read_csv(pred_csv)
            score_col = next((c for c in df.columns if 'score' in c.lower()), df.columns[0])
            s = df[score_col]
            print(f"成功 ({elapsed:.1f}s, 峰值显存: {gpu_peak_mb:.0f} MB)")
            print(f"   min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}, std={s.std():.2f}")
            if log_file.exists():
                log_file.unlink()
            return {"name": name, "status": "OK", "time": elapsed, "gpu_mb": gpu_peak_mb,
                    "min": float(s.min()), "max": float(s.max()), 
                    "mean": float(s.mean()), "std": float(s.std()), "samples": len(s)}
        except Exception as e:
            return {"name": name, "status": "NO_OUTPUT", "time": elapsed, "error": str(e)}
    else:
        return {"name": name, "status": "NO_OUTPUT", "time": elapsed, 
                "error": f"找不到预测文件: {pred_csv}"}

def main():
    print("=" * 70)
    print("Final Validation (统一控制脚本 - UTF-8编码版)")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标显存限制: {LIMIT_GB} GB (仅对非LLM任务)")
    print("=" * 70)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    setup_8g_limit()
    base_env = os.environ.copy()

    results = []
    total_start = time.time()

    for cfg in PIPELINE:
        r = run_stage(cfg, base_env)
        results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)

    summary_lines = [
        "Final Validation Summary",
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time/60:.1f} 分钟",
        f"限制显存: {LIMIT_GB} GB (LLM除外)",
        "-" * 60
    ]

    ok_count = 0
    for r in results:
        if r["status"] == "OK":
            ok_count += 1
            line = (f"{r['name']:6s} min={r['min']:6.2f}, max={r['max']:6.2f}, "
                    f"mean={r['mean']:6.2f}, std={r['std']:6.2f}, "
                    f"time={r['time']:5.1f}s, gpu={r['gpu_mb']:.0f}MB")
            summary_lines.append(line)
            print(line)
        
        elif r["status"] == "OOM":
            line = f"{r['name']:6s} [OOM] gpu=out of memory, time={r['time']:.1f}s"
            summary_lines.append(line)
            print(line)
        
        else:
            status_str = f"{r['name']:6s} [{r['status']}]"
            if "error" in r:
                status_str += f" - {r['error'][:50]}"
            summary_lines.append(status_str)
            print(status_str)

    summary_lines.append("-" * 60)
    summary_lines.append(f"成功: {ok_count}/{len(results)}")
    summary_lines.append("=" * 70)

    report_txt = "\n".join(summary_lines)
    report_path = RESULTS_ROOT / "score.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_txt)
    print(f"\n文本报告已保存: {report_path}")

    json_path = RESULTS_ROOT / "control_stats.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"stats": results, "total_time": total_time}, f, indent=2, ensure_ascii=False)
    print(f"JSON 详情已保存: {json_path}")

    for log in SCRIPT_DIR.glob("_subprocess_*.log"):
        log.unlink()

    print("\n跑完了，去睡觉！")

if __name__ == "__main__":
    main()