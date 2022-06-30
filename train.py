import torch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
