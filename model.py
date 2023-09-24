import torch
import torch.nn as nn

class CombinedNN(nn.Module):
    def __init__(self, cnn_input_channels, cnn_output_channels, mlp_input_features, mlp_hidden_dim, output_dim):
        super(CombinedNN, self).__init__()

        # Define the CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(cnn_input_channels, cnn_output_channels, kernel_size=(3, 2), padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the size of the flattened CNN output
        cnn_output_size = cnn_output_channels * (mlp_input_features // 2)

        # Define the MLP layers
        self.mlp_layers = nn.Sequential(
            nn.Linear(mlp_input_features + cnn_output_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, x_cnn, x_mlp):
        # Pass input through CNN layers
        x_cnn = self.cnn_layers(x_cnn)

        # Concatenate CNN output with MLP input
        combined_input = torch.cat((x_cnn, x_mlp), dim=1)

        # Pass combined input through MLP layers
        output = self.mlp_layers(combined_input)

        return output
