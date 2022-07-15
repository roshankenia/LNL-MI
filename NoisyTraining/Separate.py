import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from KModelTrainer import KModelTrain
import time

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# store starting time
begin = time.time()

# obtain data tensors
x_tensor = torch.load('x_tensor.pt')
# size [n_samples, 1]
y_tensor = torch.load('y_tensor.pt')

# obtain noisy indexes so we can plot them
noise_tensor = torch.load('moise_tensor.pt')

noise_count = 0

for noise in noise_tensor:
    if noise == 1:
        noise_count += 1

print('There are', len(noise_tensor),
      'samples in total, and', noise_count, 'are noisy')
