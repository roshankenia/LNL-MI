import os
import torch
import pandas as pd
import sys
from PIL import Image
import torchvision.transforms as transforms

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')


# get all filenames
filenames = sorted(os.listdir('../ISBI2016_ISIC_Part3_Training_Data'))

# convert each image to a tensor
images = []
for filename in filenames:
    # Read the image
    image = Image.open('../ISBI2016_ISIC_Part3_Training_Data/'+filename)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),  # simple data augmentation
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(
        #     brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor()
    ])

    # Convert the PIL image to Torch tensor
    img_tensor = transform(image).to(torch.float32)

    images.append(img_tensor)

images = torch.stack(images)
data_tensor = torch.tensor(images, dtype=torch.float32)
print(data_tensor.shape)
# print(data_tensor)
# save data file
torch.save(data_tensor, 'data_tensor.pt')

# get ground truth values
train = pd.read_csv(
    '../ISBI2016_ISIC_Part3_Training_GroundTruth.csv', header=None)
ground_truth_tensor = torch.tensor(
    pd.factorize(train[1])[0], dtype=torch.float32)
ground_truth_tensor = torch.unsqueeze(ground_truth_tensor, 1)
print(ground_truth_tensor.shape)
# save ground truth file
torch.save(ground_truth_tensor, 'ground_truth_tensor.pt')
