import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("计算能力:", torch.cuda.get_device_capability(0))
    # 简单测试一下GPU计算
    x = torch.rand(5, 3).cuda()
    print(f"测试张量所在设备: {x.device}") # 应输出 cuda:0
else:
    print("CUDA不可用，请检查安装。")