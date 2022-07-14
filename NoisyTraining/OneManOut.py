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
x_tensor = torch.load('cifar10_noisy_data_tensor_nonorm.pt')
# size [n_samples, 1]
y_tensor = torch.load('cifar10_noisy_ground_truth_tensor_nonorm.pt')

# obtain noisy indexes so we can plot them
noise_indexes = torch.load('cifar10_noisy_index_tensor.pt')
print(noise_indexes)
noisy_data = np.zeros(len(y_tensor))
noiseCount = 0
for index in noise_indexes:
    noisy_data[int(index.item())] = 1
    noiseCount += 1

totalRelabelArray = []
correctRelabelArray = []
incorrectRelabelArray = []

cleanCount = len(noisy_data)-noiseCount

totalCleanArray = [cleanCount]
totalNoiseArray = [noiseCount]

iteration = [0]
ite = 0


c = 0
while True:
    c += 1
    if c == 3:
        break
    # make our K Model Trainer where k represents number of models
    model_trainer = KModelTrain(x_tensor, y_tensor, k=2, num_epochs=2)

    # compute metrics for all samples
    print('Calculating Uncertainties')
    bces, furthest, pred, ensemblePred = model_trainer.calculateUncertainty(
        x_tensor, y_tensor)

    # make plot of bce and furthest uncertainty
    print('Making plot')
    result_df = pd.DataFrame(
        {'BCE': bces, 'Furthest Uncertainty': furthest, 'label': noisy_data})
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x='BCE', y='Furthest Uncertainty',
                    hue='label', data=result_df, ax=ax, s=10)
    plt.title('BCE vs Furthest Uncertainty')
    ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    picName = 'bce-vs-furth-'+str(ite)+'.png'
    plt.savefig(picName)
    plt.close()

    print('Flipping uncertain samples')
    # now flip samples that are uncertain
    new_x = x_tensor.clone().detach()
    new_y = y_tensor.clone().detach()
    totalRelabel = 0
    correctRelabel = 0
    incorrectRelabel = 0
    for i in range(len(x_tensor)):
        # if the BCE and uncertainty is above the thresholds we relabel
        # if pred[i] != y_tensor[i].item() and bces[i] > 1.25 and furthest[i] > 0.8:
        numOver9 = 0
        numUnder1 = 0
        for singlePred in ensemblePred[i]:
            if singlePred > 0.9:
                numOver9 += 1
            elif singlePred < 0.1:
                numUnder1 += 1
        if bces[i] > 1.25:
            if numOver9 == 7 or numUnder1 == 7:
                new_y[i] = -1 * new_y[i] + 1
                totalRelabel += 1
                # chek if correct relabel
                if noisy_data[i] == 1:
                    correctRelabel += 1
                    noisy_data[i] = 0
                else:
                    incorrectRelabel += 1
                    noisy_data[i] = 1
    print(
        f'Total Relabeled: {totalRelabel}, Correctly Relabeled: {correctRelabel}, Incorrectly Relabeled: {incorrectRelabel}')
    totalRelabelArray.append(totalRelabel)
    correctRelabelArray.append(correctRelabel)
    incorrectRelabelArray.append(incorrectRelabel)

    # count up number of noisy and clean remaining
    noiseCount = 0
    cleanCount = 0
    for noise in noisy_data:
        if noise == 0:
            cleanCount += 1
        else:
            noiseCount += 1

    totalNoiseArray.append(noiseCount)
    totalCleanArray.append(cleanCount)

    # set tensors to new data
    x_tensor = new_x
    y_tensor = new_y

    ite += 1
    iteration.append(ite)

    if correctRelabel < 50:
        break

print('Plotting total relabels')
result_df = pd.DataFrame(
    {'Iteration': iteration[1:], 'Total Relabeled': totalRelabelArray})
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x='Iteration', y='Total Relabeled', data=result_df, ax=ax)
plt.title('Total Relabels per Iteration')
picName = 'totalRelabels.png'
plt.savefig(picName)
plt.close()

print('Plotting correct relabels')
result_df = pd.DataFrame(
    {'Iteration': iteration[1:], 'Total Correct': correctRelabelArray})
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x='Iteration', y='Total Correct', data=result_df, ax=ax)
plt.title('Total Correct per Iteration')
picName = 'totalCorrect.png'
plt.savefig(picName)
plt.close()


print('Plotting incorrect relabels')
result_df = pd.DataFrame(
    {'Iteration': iteration[1:], 'Total Incorrect': incorrectRelabelArray})
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x='Iteration', y='Total Incorrect', data=result_df, ax=ax)
plt.title('Total Incorrect per Iteration')
picName = 'totalIncorrect.png'
plt.savefig(picName)
plt.close()


print('Plotting clean')
result_df = pd.DataFrame(
    {'Iteration': iteration, 'Total Clean': totalCleanArray})
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x='Iteration', y='Total Clean', data=result_df, ax=ax)
plt.title('Total Clean per Iteration')
picName = 'totalClean.png'
plt.savefig(picName)
plt.close()

print('Plotting noisy')
result_df = pd.DataFrame(
    {'Iteration': iteration, 'Total Noisy': totalNoiseArray})
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(x='Iteration', y='Total Noisy', data=result_df, ax=ax)
plt.title('Total Noisy per Iteration')
picName = 'totalNoisy.png'
plt.savefig(picName)
plt.close()

# save data
torch.save(x_tensor, 'x_tensor.pt')
torch.save(y_tensor, 'y_tensor.pt')
# save noisy indexes
noisy_indexes = torch.tensor(noisy_data, dtype=torch.float32)
torch.save(noisy_indexes, 'moise_tensor.pt')

# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")
