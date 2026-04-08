# src/utils/model_loader.py
import os
import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

def is_huggingface_reachable():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2)
        return True
    except OSError:
        return False

def download_via_modelscope(repo_id, cache_dir):
    """使用 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        print(f"从 ModelScope 下载: {repo_id}")
        print(f"   目标目录: {cache_dir}")
        # 新版 modelscope 不需要 resume_download 参数，自动支持断点续传
        local_path = snapshot_download(repo_id, local_dir=cache_dir)
        print(f"✓ 下载完成: {local_path}")
        return local_path
    except ImportError:
        raise ImportError("请安装 modelscope: pip install modelscope")
    except Exception as e:
        raise RuntimeError(f"ModelScope 下载失败: {e}\n请手动下载后放入 {cache_dir}")

def load_qwen_model(model_name_or_path, num_bits=4, auto_download=True):
    """
    加载 Qwen 模型，支持自动精度选择：
    - 如果 num_bits 指定为 4 或 None，会根据模型大小自动选择：
        * 4B 模型（含 "4B"） -> FP16 (num_bits=None)
        * 8B 模型（含 "8B"） -> INT4 (num_bits=4)
    - 显式传入 num_bits=4 或 num_bits=None 也会触发自动规则，
      但 None 且无法判断时默认 FP16。
    """
    # === 自动精度选择逻辑 ===
    if num_bits is None:
        # 默认自动判断：如果模型名称包含 "8B" 则使用 INT4，否则使用 FP16
        if "8B" in model_name_or_path:
            effective_bits = 4
            print("检测到 8B 模型，自动使用 INT4 量化以节省显存")
        else:
            effective_bits = None  # FP16
            print("检测到 4B 或更小模型，使用 FP16 精度")
    else:
        effective_bits = num_bits

    # 如果需要 INT4 量化，检查 bitsandbytes 是否安装
    if effective_bits == 4:
        try:
            import bitsandbytes
            print(f"bitsandbytes 版本: {bitsandbytes.__version__}")
        except ImportError:
            raise ImportError(
                "需要 INT4 量化但 bitsandbytes 未安装。请运行: pip install bitsandbytes"
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None

    """
    加载 Qwen 模型
    """
    local_path = None
    is_local = False
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # 1. 本地路径检查（包含 / 或 \ 且存在）
    if ('/' in model_name_or_path or '\\' in model_name_or_path) and Path(model_name_or_path).exists():
        local_path = model_name_or_path
        is_local = True
        print(f"本地路径: {local_path}")
    
    # 2. 统一下载到项目目录 models/
    else:
        model_name_safe = model_name_or_path.replace('/', '--')
        cache_dir = base_dir / "models" / model_name_safe
        
        if cache_dir.exists() and any(cache_dir.glob("*.safetensors")):
            # 已下载过
            local_path = str(cache_dir)
            is_local = True
            print(f"使用已下载模型: {local_path}")
        elif auto_download:
            # 需要下载
            print(f"准备下载到: {cache_dir}")
            local_path = download_via_modelscope(model_name_or_path, str(cache_dir))
            is_local = True
        else:
            raise RuntimeError(f"模型未找到且 auto_download=False: {model_name_or_path}")

    # 加载（强制本地，不联网）
    print(f"正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        trust_remote_code=True,
        local_files_only=True  # 关键：强制本地
    )
    
    print(f"正在加载模型（INT4={num_bits==4}）...")
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True  # 关键：强制本地
    )
    
    print("✓ 模型加载完成")
    return model, tokenizer