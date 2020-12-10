# coding: utf-8

# In[24]:


# !pip3 install torch
# !pip3 install pandas
# get_ipython().system('pip3 install pytorch-ignite')


# In[25]:


import ignite


# In[26]:


import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision

from model import CNNTest
from load_data import get_data_loader
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from torchviz import make_dot


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

BAT_SIZE = 64

# net = CNNTest(BAT_SIZE)
model = CNNTest(BAT_SIZE).to(device=device)
# net = Example()
#make_dot(model)

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100], device=device))

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loader, seq_data, labels = get_data_loader("data/fullset_train.csv", BAT_SIZE, device)
val_loader, x, l = get_data_loader("data/fullset_test.csv", BAT_SIZE, device)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
# nonviral_points = []
# viral_points = []
emb = model.embedding(seq_data)
emb = emb.cpu().detach().numpy()
print(emb.shape)
# for i in range(0, emb.shape[0]):
#     pca = PCA(n_components=2)
#     pca.fit(emb[i][:][:])
#     print(i, emb.shape[0])
#     if labels[i] == 1:
#         viral_points.append(pca.singular_values_)
#     else:
#         nonviral_points.append(pca.singular_values_)
    

# plt.scatter(*zip(*nonviral_points), c='lightblue', label='non-viral')
# plt.scatter(*zip(*viral_points), c='coral', label='viral')
# #plt.legend()
# plt.show()


def thresholded_output_transform(output):
    # print(output)
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    # y_pred = F.log_softmax(y_pred)
    y_pred = torch.round(y_pred)
    # print(y_pred)
    return y_pred, y


val_metrics = {
    "nll": Loss(criterion),
    "recall": Recall(output_transform=thresholded_output_transform),
    "precision": Precision(output_transform=thresholded_output_transform),
    "accuracy": Accuracy(output_transform=thresholded_output_transform),
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        "Training Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} Acc: {:.2f}".format(
            trainer.state.epoch,
            metrics["nll"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
        )
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} Acc: {:.2f}".format(
            trainer.state.epoch,
            metrics["nll"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
        )
    )


trainer.run(train_loader, max_epochs=1)

