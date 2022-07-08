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

# ensure we are running on the correct gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    print('exiting')
    sys.exit()
else:
    print('GPU is being properly used')

# obtain data tensors

x_tensor = torch.load('cifar10_noisy_data_tensor_nonorm.pt')
# size [n_samples, 1]
y_tensor = torch.load('cifar10_noisy_ground_truth_tensor_nonorm.pt')

# make our K Model Trainer where k represents number of models
model_trainer = KModelTrain(x_tensor, y_tensor, k=2)

# compute metrics for all samples
bces, furthest = model_trainer.calculateUncertainty(x_tensor, y_tensor)

# obtain noisy indexes so we can plot them
noise_indexes = torch.load('cifar10_noisy_index_tensor.pt')

noisy_data = np.zeros(len(y_tensor))
for index in noise_indexes:
    noisy_data[index] = 1

# make plot of bce and furthest uncertainty
result_df = pd.DataFrame(
    {'BCE': bces, 'Furthest Uncertainty': furthest, 'label': noisy_data})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='BCE', y='Furthest Uncertainty',
                hue='label', data=result_df, ax=ax, s=10)
plt.title('BCE vs Furthest Uncertainty')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.savefig('bce-vs-furth.png')
plt.close()


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
