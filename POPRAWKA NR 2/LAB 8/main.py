import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = np.linspace(-10, 10, 20000).reshape(-1, 1)
y = np.sin(x)

x = torch.FloatTensor(x).to(device)
y = torch.FloatTensor(y).to(device)

num_samples = len(x)
indices = torch.randperm(num_samples)
train_size = int(0.8 * num_samples)
train_indices = indices[:train_size]
test_indices = indices[train_size:]
x_train, y_train = x[train_indices], y[train_indices]
x_test, y_test = x[test_indices], y[test_indices]

x_extra = np.linspace(-30, 30, 5000).reshape(-1, 1)
y_extra = np.sin(x_extra)
x_extra = torch.FloatTensor(x_extra).to(device)
y_extra = torch.FloatTensor(y_extra).to(device)

class SineApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.act(self.fc5(x))
        x = self.fc6(x)  # no ReLU on output (regression)
        return x

model = SineApproximator().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = float("inf")
best_model_state = None
patience = 50
epochs_no_improve = 0

num_epochs = 5000
pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(x_test)
        val_loss = criterion(pred_val, y_test)

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.6f} Val: {val_loss.item():.6f}")

    if epochs_no_improve >= patience:
        tqdm.write("Early stopping triggered.")
        break

# Load the best model state
if best_model_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

model.eval()
with torch.no_grad():
    pred_test = model(x_test)
    test_loss = criterion(pred_test, y_test)
    pred_extra = model(x_extra)
    extra_loss = criterion(pred_extra, y_extra)

tqdm.write(f"Test Loss (within training range): {test_loss.item():.6f}")
tqdm.write(f"Test Loss (outside training range): {extra_loss.item():.6f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), label="True", s=10)
plt.scatter(x_test.cpu().numpy(), pred_test.cpu().numpy(), label="Predicted", s=10)
plt.title("Test Data (Within Training Range)")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_extra.cpu().numpy(), y_extra.cpu().numpy(), label="True", s=10)
plt.scatter(x_extra.cpu().numpy(), pred_extra.cpu().numpy(), label="Predicted", s=10)
plt.title("Test Data (Outside Training Range)")
plt.legend()

plt.tight_layout()
plt.show()