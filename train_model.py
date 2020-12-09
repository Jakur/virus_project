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


# In[27]:


BAT_SIZE = 64


# In[28]:


class SeqData(Dataset):
    def __init__(self, sequences, labels):
        self.data = torch.from_numpy(sequences)
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.labels = self.labels.view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index]
        data_val = self.data[index]
        return data_val, label


# In[29]:


def get_data_loader(fname):
    df = pd.read_csv(fname, names=["name", "seq", "class"])
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}

    def mapping_fn(string):
        x = [mapping[x] for x in string]
        return x

    column = df["seq"].apply(lambda x: mapping_fn(x))
    data = np.zeros((len(df), 300), dtype=np.int64)
    for i, d in enumerate(data):
        data[i, :] = d
    print(df["class"].sum())
    dataset = SeqData(data, df["class"].values)
    data_loader = DataLoader(dataset, batch_size=BAT_SIZE, shuffle=True, drop_last=True)
    return data_loader


train_loader = get_data_loader("data/fullset_test.csv")


# In[32]:

# net = CNNTest(BAT_SIZE)
model = CNNTest(BAT_SIZE)
# net = Example()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = get_data_loader("data/fullset_train.csv")
val_loader = get_data_loader("data/fullset_test.csv")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

trainer = create_supervised_trainer(model, optimizer, criterion)


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = F.log_softmax(y_pred)
    y_pred = torch.round(y_pred)
    return y_pred, y


val_metrics = {
    "nll": Loss(criterion),
    "recall": Recall(output_transform=thresholded_output_transform),
    "precision": Precision(output_transform=thresholded_output_transform),
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
        "Training Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f}".format(
            trainer.state.epoch, metrics["nll"], metrics["precision"], metrics["recall"]
        )
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f}".format(
            trainer.state.epoch, metrics["nll"], metrics["precision"], metrics["recall"]
        )
    )


trainer.run(train_loader, max_epochs=1)

