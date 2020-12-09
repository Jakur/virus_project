from collections import Counter
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset


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


def get_data_loader(fname, bs, ratio=None):
    df = pd.read_csv(fname, names=["name", "seq", "class"])
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}

    def mapping_fn(string):
        x = [mapping[x] for x in string]
        return x

    if ratio == None:
        column = df["seq"].apply(lambda x: mapping_fn(x))
        data = np.zeros((len(df), 300), dtype=np.int64)
        for i, d in enumerate(column):
            data[i, :] = d
        dataset = SeqData(data, df["class"].values)
    else:
        positives = (np.sum(df["class"].values) // bs) * bs
        data = np.zeros((positives * 2, 300), dtype=np.int64)
        pos_rows = df.loc[df["class"] == 1]
        neg_rows = df.loc[df["class"] == 0]
        x = pos_rows["seq"].apply(lambda x: mapping_fn(x)).values
        for i, d in enumerate(x):
            data[i, :] = d
        y = neg_rows["seq"].apply(lambda x: mapping_fn(x)).values
        for i, d in enumerate(y):
            if i <= positives * 2:
                break
            data[i + positives, :] = d
        labels = np.concatenate((np.ones(positives), np.zeros(positives)), axis=None)
        dataset = SeqData(data, labels)
    # Check correctness on first row
    # counter = Counter(df["seq"][0])
    # for a, b in counter.items():
    #     print(f"{a} {b}")
    # s = sum(mapping[k] * v for k, v in counter.items())
    # print(s)
    # assert s == np.sum(data[0, :])
    # print(df["class"].sum())

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    return data_loader


if __name__ == "__main__":
    x = get_data_loader("data/fullset_test.csv", 64, ratio=1)
    print(len(x.dataset))

