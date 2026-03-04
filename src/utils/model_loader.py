# src/utils/model_loader.py
import os
import socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def is_huggingface_reachable():
    """检查能否连上 Hugging Face（2秒超时）"""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2)
        return True
    except OSError:
        return False

def load_qwen_model(model_name, num_bits=4):
    """
    智能加载Qwen模型：能连网就正常加载，连不上就强制本地
    解决Qwen tokenizer强制联网查询Mistral元数据的问题
    """
    offline_mode = not is_huggingface_reachable()
    
    if offline_mode:
        print("网络不可达，强制使用本地缓存模式")
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(num_bits == 4),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    ) if num_bits == 4 else None
    
    # 关键：根据网络状况自动选择 local_files_only
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=offline_mode
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=offline_mode
    )
    
    return model, tokenizer