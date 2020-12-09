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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision

from model import CNNTest
from load_data import get_data_loader

BAT_SIZE = 64

# net = CNNTest(BAT_SIZE)
model = CNNTest(BAT_SIZE)
# net = Example()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([10]))

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = get_data_loader("data/fullset_train.csv", BAT_SIZE)
val_loader = get_data_loader("data/fullset_test.csv", BAT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

trainer = create_supervised_trainer(model, optimizer, criterion)


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
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


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


trainer.run(train_loader, max_epochs=2)

