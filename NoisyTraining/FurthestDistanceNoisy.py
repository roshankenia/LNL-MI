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
for index in noise_indexes:
    noisy_data[int(index.item())] = 1


for i in range(5):
    # make our K Model Trainer where k represents number of models
    model_trainer = KModelTrain(x_tensor, y_tensor, k=8, num_epochs=50)

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
    picName = 'bce-vs-furth-'+str(i)+'.png'
    plt.savefig(picName)
    plt.close()

    print('Flipping uncertain samples')
    # now flip samples that are uncertain
    new_x = x_tensor.clone().detach()
    new_y = y_tensor.clone().detach()
    totalRelabel = 0
    correctRelabel = 0
    incorrectRelabel = 0
    n = 0
    for i in range(len(x_tensor)):
        # if the BCE and uncertainty is above the thresholds we relabel
        # print(bces[i], furthest[i])
        # if pred[i] != y_tensor[i].item() and bces[i] > 1.25 and furthest[i] > 0.8:
        # by raising bce threshold to 2 we only select the samples we are confident are noisy
        numOver9 = 0
        numUnder1 = 0
        for singlePred in ensemblePred[i]:
            if singlePred > 0.9:
                numOver9 += 1
            elif singlePred < 0.1:
                numUnder1 += 1
        noiseRel = 0
        cleanRel = 1
        if numOver9 == 7 or numUnder1 == 7:
            if noisy_data[i] == 1:
                noiseRel += 1
                # print('Noisy', torch.round(
                #     ensemblePred[i]), bces[i], furthest[i])
                # print('Number over .9:', numOver9,
                #       'Num under .1:', numUnder1)
            else:
                cleanRel += 1
                print('Clean', ensemblePred[i], '\n', torch.round(
                    ensemblePred[i]), bces[i], furthest[i])
                # print('Number over .9:', numOver9,
                #       'Num under .1:', numUnder1)
        print('NoiseRel:', noiseRel)
        print('CleanRel:', cleanRel)
        if bces[i] > 1.25:
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

    # set tensors to new data
    x_tensor = new_x
    y_tensor = new_y


# store end time
end = time.time()
timeTaken = time.strftime("%H:%M:%S", time.gmtime(end-begin))
# total time taken
print(f"Total runtime of the program is {timeTaken}")

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Hyper-parameters
# num_epochs = 50
# batch_size = 512
# learning_rate = 0.01

# # dataset has PILImage images of range [0, 1].
# # We transform them to Tensors of normalized range [-1, 1]

# train_dataset = Cifar10BinaryNoisy()
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
#                                            shuffle=True)

# test_dataset = Cifar10BinaryNoisyTest()
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
#                                           shuffle=False)

# model = torchvision.models.resnet34(pretrained=False, num_classes=1).to(device)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # scheduler.step()

#         if (i+1) % 10 == 0:
#             print(
#                 f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
#             # print(binary_acc(outputs, labels))
#             # print(outputs)
#             # print(labels)

# print('Finished Training')
# # PATH = './resnet-mi.pth'
# # torch.save(model.state_dict(), PATH)

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     x = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)

#         y_test_pred = torch.sigmoid(outputs)
#         y_pred_tag = torch.round(y_test_pred)

#         if x == 0:
#             # print('labels:', labels[0:20])
#             # print('outputs:', outputs[0:20])
#             # print('predicted:', y_pred_tag[0:20])
#             x = 1

#         n_correct += (y_pred_tag == labels).sum().float()
#         n_samples += len(labels)
#     acc = n_correct/n_samples
#     acc = torch.round(acc * 100)
#     print('Number correct:', n_correct.item(), 'out of:', n_samples)
#     noise_index_tensor = torch.load('cifar10_noisy_index_tensor.pt')
#     print('Number of noisy samples:', len(noise_index_tensor))
#     print(f'Accuracy of the network: {acc} %')
