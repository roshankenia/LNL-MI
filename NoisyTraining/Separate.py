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
from DataSplitter import KDataSplitter
from ModelTrain import FDModel
from Cifar10TestClean import Cifar10BinaryCleanTest
import time

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# we know want to split the hard samples from the noisy samples


# split data
data_splitter = KDataSplitter(x_tensor, y_tensor, k=2)
x_tensors, y_tensors = data_splitter.split()
# train one model on half the data
first_model = FDModel(x_tensor, y_tensor, num_epochs=50)
first_model.train()


test_dataset = Cifar10BinaryCleanTest()
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128,
                                          shuffle=False)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    x = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = first_model.predict(images)

        y_test_pred = torch.sigmoid(outputs)
        y_pred_tag = torch.round(y_test_pred)

        if x == 0:
            print('labels:', labels[0:20])
            # print('outputs:', outputs[0:20])
            print('predicted:', y_pred_tag[0:20])
            x = 1

        n_correct += (y_pred_tag == labels).sum().float()
        n_samples += len(labels)
    acc = n_correct/n_samples
    acc = torch.round(acc * 100)
    print('Number correct:', n_correct, 'out of:', n_samples)
    print(f'Accuracy of the network: {acc} %')


# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")
