import os
import torch
import pandas as pd
import sys
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import random

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True)

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # simple data augmentation
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(
    #     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

images = []
labels = []

# obtain samples that we need, in our case we are using deer and horses
i = 0
while i < train_dataset.__len__():
    image, label = train_dataset[i]
    # deer
    if label == 4:
        images.append(transform(image).to(torch.float32))
        labels.append(1)
    # horse
    elif label == 7:
        images.append(transform(image).to(torch.float32))
        labels.append(1)
    # automobile
    elif label == 1:
        images.append(transform(image).to(torch.float32))
        labels.append(0)
    # truck
    elif label == 9:
        images.append(transform(image).to(torch.float32))
        labels.append(0)
    if i % 1000 == 0:
        print('at', i)
    i += 1

# now flip labels with random probability
threshold = 0.25  # represents XX% probability of incorrect label
flippedIndexes = []
for j in range(len(labels)):
    # generate random float
    flip_prob = random.random()
    # check if need to flip
    if flip_prob <= threshold:
        # flip label 0 -> 1 or 1 -> 0
        labels[j] = labels[j] * -1 + 1
        flippedIndexes.append(j)

images = torch.stack(images)
data_tensor = torch.tensor(images, dtype=torch.float32)
print(data_tensor.shape)
# print(data_tensor)
# save data file
torch.save(data_tensor, 'cifar10_noisy_data_tensor_nonorm.pt')

# get ground truth values
ground_truth_tensor = torch.tensor(labels, dtype=torch.float32)
ground_truth_tensor = torch.unsqueeze(ground_truth_tensor, 1)
print(ground_truth_tensor.shape)
# save ground truth file
torch.save(ground_truth_tensor, 'cifar10_noisy_ground_truth_tensor_nonorm.pt')

# save noisy indexes
noise_index_tensor = torch.tensor(flippedIndexes, dtype=torch.float32)
torch.save(noise_index_tensor, 'cifar10_noisy_index_tensor.pt')
