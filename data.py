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


# get all filenames
filenames = sorted(os.listdir('../ISBI2016_ISIC_Part3_Training_Data'))

# convert each image to a tensor
images = []
for filename in filenames:
    # Read the image
    image = Image.open('../ISBI2016_ISIC_Part3_Training_Data/'+filename)

    transform = transforms.Compose([
        transforms.PILToTensor()])

    # Convert the PIL image to Torch tensor
    img_tensor = transform(image).to(torch.float32)/255

    images.append(img_tensor)

print(images)

# get ground truth values
train = pd.read_csv('../ISBI2016_ISIC_Part3_Training_GroundTruth.csv')
train_tensor = torch.tensor(pd.factorize(train['benign'])[0])
print(train_tensor)
