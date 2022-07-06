import os
import torch
import pandas as pd
import sys
from PIL import Image
import torchvision.transforms as transforms
import torchvision

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True)

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),  # simple data augmentation
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(
    #     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

images = []
labels = []

i = 0
while i < train_dataset.__len__():
    image, label = train_dataset[i]
    if label == 4:
        images.append(transform(image).to(torch.float32))
        labels.append(0)
    elif label == 7:
        images.append(transform(image).to(torch.float32))
        labels.append(1)
    if i % 1000 == 0:
        print('at', i)
    i += 1

images = torch.stack(images)
data_tensor = torch.tensor(images, dtype=torch.float32)
print(data_tensor.shape)
# print(data_tensor)
# save data file
torch.save(data_tensor, 'cifar10_clean_data_tensor_test.pt')

# get ground truth values
ground_truth_tensor = torch.tensor(labels, dtype=torch.float32)
ground_truth_tensor = torch.unsqueeze(ground_truth_tensor, 1)
print(ground_truth_tensor.shape)
# save ground truth file
torch.save(ground_truth_tensor, 'cifar10_clean_ground_truth_tensor_test.pt')