import torch

print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前 CUDA 设备数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
else:
    print("未检测到 CUDA 设备或驱动未正确安装。")
import torch
print(torch.__version__)
