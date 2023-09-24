# main
import torch
import torch.optim as optim
import torch.nn as nn

from paths import generate_paths
from normalize import save_normalized_data
from reshape import reshape_CNN_data, reshape_MLP_data, reshape_target_data
from model import CombinedNN

# Generate results folders, CSV paths, etc.
paths = generate_paths()

# Normalize the data and save the normalization details
xCNN, xMLP, target = save_normalized_data(paths)
print(f"Shape of xCNN after normalization: {xCNN.shape}")

# Reshape normalized data
reshaped_CNN_data = reshape_CNN_data(xCNN)
print(f"Shape of reshaped_CNN_data: {reshaped_CNN_data.shape}")
reshaped_MLP_data = reshape_MLP_data(xMLP, reshaped_CNN_data)
reshaped_target_data = reshape_target_data(target)

#set_model
model = CombinedNN(cnn_input_channels=1, cnn_output_channels=32, mlp_input_features=120, mlp_hidden_dim=64, output_dim=1)

# Convert data to tensor
xCNN_tensor = torch.tensor(reshaped_CNN_data, dtype=torch.float32)
print(f"Shape of xCNN_tensor: {xCNN_tensor.shape}")
xMLP_tensor = torch.tensor(reshaped_MLP_data, dtype=torch.float32)
target_tensor = torch.tensor(reshaped_target_data, dtype=torch.float32)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(xCNN_tensor, xMLP_tensor)
    loss = criterion(outputs, target_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# TODO: training
# TODO: testing
# TODO: visualization