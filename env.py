import torch
print(torch.cuda.is_available())  # 应该返回True
print(torch.cuda.current_device())  # 打印当前设备索引
print(torch.cuda.get_device_name(0))  # 打印GPU的名称
