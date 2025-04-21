import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_DIR, CHECKPOINT

class Options:
    def __init__(self):
        self.linear_size = 1024
        self.num_stage = 2
        self.epochs = 200
        self.lr = 1e-3
        self.dropout = 0.5
        self.data_2d_path = os.path.join(DATA_DIR, 'data_2d_h36m_gt.npz')
        self.data_3d_path = os.path.join(DATA_DIR, 'data_3d_h36m.npz')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 64
        self.save_path = CHECKPOINT
        self.val_interval=20


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


class Human36MDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32).view(len(inputs), -1)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(len(targets), -1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

def load_by_subject_split(data_2d_path, data_3d_path,
                          train_subjects=['S1', 'S5', 'S6', 'S7', 'S8'],
                          val_subjects=['S9', 'S11']):
    data_2d = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()
    data_3d = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()

    def collect(subjects):
        x, y = [], []
        for subj in subjects:
            for act in data_2d[subj]:
                poses_2d_cams = data_2d[subj][act]    # list of camera views
                poses_3d = data_3d[subj][act]         # single view (shared across cameras)

                for cam_idx in range(len(poses_2d_cams)):
                    poses_2d = poses_2d_cams[cam_idx]
                    if poses_2d.shape[0] != poses_3d.shape[0]:
                        continue
                    x.append(poses_2d)
                    y.append(poses_3d)
        return np.vstack(x), np.vstack(y)

    x_train, y_train = collect(train_subjects)
    x_val, y_val = collect(val_subjects)

    return Human36MDataset(x_train, y_train), Human36MDataset(x_val, y_val)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mpjpe(pred, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def train(model, train_loader, val_loader, optimizer, device, 
          epochs=200, save_path='best_model.pth', val_interval=20):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = mpjpe(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] MPJPE: {avg_loss:.4f}")

        if (epoch+1)%val_interval == 0 or epoch == epochs -1:
            avg_val = evaluate(model, val_loader, device)
            val_losses.append(avg_val)
            val_losses.append(epoch+1)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at epoch {epoch+1} with Val MPJPE: {best_val_loss:.4f}")
            print(f"[Epoch {epoch+1}] Train MPJPE: {avg_loss:.4f} | Val MPJPE: {avg_val:.4f}")
        else:
            print(f"[Epoch {epoch+1}] Train MPJPE: {avg_loss:.4f}")

def main():
    opt = Options()

    # 데이터셋 로딩
    train_dataset, val_dataset = load_by_subject_split(
        opt.data_2d_path, opt.data_3d_path
    )

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    # 모델 구성
    model = LinearModel(
        input_size=train_dataset[0][0].shape[0],
        output_size=train_dataset[0][1].shape[0],
        linear_size=opt.linear_size,
        num_stage=opt.num_stage,
        p_dropout=opt.dropout
    ).to(opt.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 학습 시작
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=opt.device,
        epochs=opt.epochs,
        save_path=opt.save_path,
        val_interval=opt.val_interval
    )

if __name__ == '__main__':
    main()

