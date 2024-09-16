from generate_trajectory_data import save_trajectories, process_data
from dataloader import TrajectoryDataset
from model import ADAPT
from loss import Loss
import torch
from torch.utils.data import Dataset, DataLoader


# generate dataset
output = save_trajectories(1,40,0.1)
# Processed data
speeds, trajectories = process_data(output)
# Create the dataset and dataloaders
dataset = TrajectoryDataset(speeds, trajectories )


# train test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# train and validation loader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# load model
model = ADAPT()

# Initialize the custom loss function
criterion = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            trajs, probs = model(inputs)
            loss = criterion(trajs, targets,probs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")





if __name__ == "__main__":
    num_epochs = 10
    train_model(model, train_loader, criterion, optimizer, num_epochs)