import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, speeds, trajectories):
        self.speeds = speeds
        self.trajectories = trajectories

    def __len__(self):
        return len(self.speeds)

    def __getitem__(self, idx):
        speed = self.speeds[idx]
        trajectory = self.trajectories[idx]
        return torch.tensor(speed, dtype=torch.float32).unsqueeze(dim = 0), torch.tensor(trajectory, dtype=torch.float32).unsqueeze(dim = 0)
    

