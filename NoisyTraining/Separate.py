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
x_tensor = torch.load('x_tensor.pt').to(device)
# size [n_samples, 1]
y_tensor = torch.load('y_tensor.pt').to(device)

# obtain noisy indexes so we can plot them
noise_tensor = torch.load('moise_tensor.pt')

noise_count = 0

for noise in noise_tensor:
    if noise == 1:
        noise_count += 1

print('There are', len(noise_tensor),
      'samples in total, and', noise_count, 'are noisy')

# we know want to split the hard samples from the noisy samples
iterationData = []
loss = nn.BCELoss()

for iter in range(2):
    print('Iteration:', iter)
    # split data
    data_splitter = KDataSplitter(x_tensor, y_tensor, k=2)
    x_tensors, y_tensors = data_splitter.split()
    # train one model on half the data
    first_model = FDModel(x_tensors[0], y_tensors[0], num_epochs=50)
    first_model.train()
    print('Calculating for first model')
    # predict on all of data and note entropy and peak value
    entropy = []
    peakValue = []
    # prediction
    prediction = torch.sigmoid(first_model.predict(x_tensor))
    for i in range(len(x_tensor)):
        y_sample = y_tensor[i]

        # obtain predictions from each model
        y_pred = prediction[i]

        # print(y_pred)
        # print(y_sample)
        # compute binary cross entropy loss using this average
        bce = loss(y_pred, y_sample)
        # print('bce:', bce)
        # compute peak value
        peak = None
        if 1-y_pred.item() == 0:
            peak = 100
        else:
            peak = y_pred.item()/(1-y_pred.item())
        # print('furth:', furthestUncertainty, res)
        entropy.append(bce.item())
        peakValue.append(peak)
    iterationData.append([prediction, entropy, peakValue])

    # train another model on other half
    second_model = FDModel(x_tensors[1], y_tensors[1], num_epochs=50)
    second_model.train()
    print('Calculating for second model')
    # predict on all of data and note entropy and peak value
    entropy = []
    peakValue = []
    # prediction
    prediction = torch.sigmoid(second_model.predict(x_tensor))
    for i in range(len(x_tensor)):
        y_sample = y_tensor[i]

        # obtain predictions from each model
        y_pred = prediction[i]

        # print(y_avg)
        # print(y_sample)
        # compute binary cross entropy loss using this average
        bce = loss(y_pred, y_sample)
        # print('bce:', bce)
        # compute peak value
        peak = None
        if 1-y_pred.item() == 0:
            peak = 100
        else:
            peak = y_pred.item()/(1-y_pred.item())
        # print('furth:', furthestUncertainty, res)
        entropy.append(bce.item())
        peakValue.append(peak)
    iterationData.append([prediction, entropy, peakValue])

# calculate variance in prediction, entropy, and peak value
sampleData = []
predictionVars = []
entropyVars = []
peakVars = []
print('Calculating std for all samples')
for j in range(len(x_tensor)):
    entropyVals = []
    peakVals = []
    predictionVals = []
    # obtain data
    for iter in range(4):
        predictionVals.append(iterationData[iter][0][j].item())
        entropyVals.append(iterationData[iter][1][j])
        peakVals.append(iterationData[iter][2][j])
    # calculate stds
    predictionVar = np.std(predictionVals)
    entropyVar = np.std(entropyVals)
    peakVar = np.std(peakVals)

    sampleData.append([predictionVar, entropyVar, peakVar])
    predictionVars.append(predictionVar)
    entropyVars.append(entropyVar)
    peakVars.append(peakVar)

print(sampleData)


# make plot of entropy and peak val
print('Making entropy and peak val plot')
result_df = pd.DataFrame(
    {'Standard Deviation in Entropy': entropyVars, 'Standard Deviation in Peak Value': peakVars, 'label': noise_tensor})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='Standard Deviation in Entropy', y='Standard Deviation in Peak Value',
                hue='label', data=result_df, ax=ax, s=10)
plt.title('STD in Entropy vs Peak Value')
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
picName = 'stdEntvsPeak.png'
plt.savefig(picName)
plt.close()

# make plot of entropy and prediction
print('Making entropy and prediction plot')
result_df = pd.DataFrame(
    {'Standard Deviation in Entropy': entropyVars, 'Standard Deviation in Prediction': predictionVars, 'label': noise_tensor})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='Standard Deviation in Entropy', y='Standard Deviation in Prediction',
                hue='label', data=result_df, ax=ax, s=10)
plt.title('STD in Entropy vs Prediction')
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
picName = 'stdEntvsPred.png'
plt.savefig(picName)
plt.close()

# make plot of peak val and prediction
print('Making peak val and prediction plot')
result_df = pd.DataFrame(
    {'Standard Deviation in Peak Value': peakVars, 'Standard Deviation in Prediction': predictionVars, 'label': noise_tensor})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='Standard Deviation in Peak Value', y='Standard Deviation in Prediction',
                hue='label', data=result_df, ax=ax, s=10)
plt.title('STD in Peak Value vs Prediction')
ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
picName = 'stdPeakvsPred.png'
plt.savefig(picName)
plt.close()


# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")

# test_dataset = Cifar10BinaryCleanTest()
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128,
#                                           shuffle=False)

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     x = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = first_model.predict(images)

#         y_test_pred = torch.sigmoid(outputs)
#         y_pred_tag = torch.round(y_test_pred)

#         if x == 0:
#             print('labels:', labels[0:20])
#             # print('outputs:', outputs[0:20])
#             print('predicted:', y_pred_tag[0:20])
#             x = 1

#         n_correct += (y_pred_tag == labels).sum().float()
#         n_samples += len(labels)
#     acc = n_correct/n_samples
#     acc = torch.round(acc * 100)
#     print('Number correct:', n_correct, 'out of:', n_samples)
#     print(f'Accuracy of the network: {acc} %')
