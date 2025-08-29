import torch

# 检查 CUDA 是否可用
print("CUDA available:", torch.cuda.is_available())

# 如果可用，打印 GPU 名称和数量
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 简单测试：在 GPU 上建立张量并计算
    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    print("Test result on GPU:", torch.mm(x, y))
    
print(torch.inf*0.0)