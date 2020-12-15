import pandas as pd
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ignite
from torch.utils.data import DataLoader, TensorDataset, Dataset
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Recall, Precision, Fbeta

from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from model import ResNet
from load_data import get_data_loader_trimers

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

BAT_SIZE = 128

model = ResNet(BAT_SIZE).to(device=device)


def plot_embedding(model, mapping):
    parms = None
    for x in model.embed.parameters():
        parms = x.cpu().detach()
        break
    print(parms.shape)
    parms = parms.numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(parms)
    x = reduced[:, 0]
    y = reduced[:, 1]
    m = [(v, k) for k, v in mapping.items()]
    m.sort()
    print(m)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, (_, text) in enumerate(m):
        if text in ["ATG", "TGA", "TAA", "TAG"]:
            ax.annotate(text, (x[i], y[i]))
    ax.set_xlabel("Reduced Dimension 0")
    ax.set_ylabel("Reduced Dimension 1")
    ax.set_title("PCA-Reduced Trimer Embedding")
    plt.show()


pos_weight = 5
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

optimizer = optim.AdamW(model.parameters(), weight_decay=0.01)

train_loader, mapping = get_data_loader_trimers(
    "data/fullset_train.csv", BAT_SIZE, device
)
val_loader, mapping = get_data_loader_trimers(
    "data/fullset_validation.csv", BAT_SIZE, device, mapping=mapping
)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    y_pred = torch.round(y_pred)
    return y_pred, y


val_metrics = {
    "nll": Loss(criterion),
    "recall": Recall(output_transform=thresholded_output_transform),
    "precision": Precision(output_transform=thresholded_output_transform),
    "accuracy": Accuracy(output_transform=thresholded_output_transform),
}
val_metrics["f-score"] = Fbeta(
    0.5, False, val_metrics["precision"], val_metrics["recall"]
)
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


def score_function(engine):
    return engine.state.metrics["recall"]


to_save = {"model": model}
handler = Checkpoint(
    to_save,
    DiskSaver("models"),
    filename_prefix=f"T{pos_weight}best",
    score_function=score_function,
    score_name="loss",
    global_step_transform=global_step_from_engine(trainer),
)

evaluator.add_event_handler(Events.COMPLETED, handler)


@trainer.on(Events.ITERATION_COMPLETED(every=500))
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        "Training Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} F-Score: {:.2f}".format(
            trainer.state.epoch,
            metrics["nll"],
            metrics["precision"],
            metrics["recall"],
            metrics["f-score"],
        )
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {} Avg loss: {:.2f} Precision: {:.2f} Recall: {:.2f} F-Score: {:.2f}".format(
            trainer.state.epoch,
            metrics["nll"],
            metrics["precision"],
            metrics["recall"],
            metrics["f-score"],
        )
    )


trainer.run(train_loader, max_epochs=5)

plot_embedding(model, mapping)
