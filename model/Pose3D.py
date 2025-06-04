import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Human36MDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).view(len(inputs), -1)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(len(targets), -1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(),
            nn.Dropout(p_dropout)
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(linear_size, linear_size),
                nn.BatchNorm1d(linear_size),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(linear_size, linear_size),
                nn.BatchNorm1d(linear_size),
                nn.ReLU(),
                nn.Dropout(p_dropout)
            ) for _ in range(num_stage)
        ])

        self.output_layer = nn.Linear(linear_size, output_size)

    def forward(self, x):
        y = self.input_layer(x)
        for block in self.blocks:
            y = y + block(y) # 잔차 블록
        return self.output_layer(y)


def mpjpe(predicted, target):
    predicted = predicted.view(predicted.shape[0], -1, 3)
    target = target.view(target.shape[0], -1, 3)
    return torch.mean(torch.norm(predicted - target, dim=2))