from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class Audio2LogMelSpec(nn.Module):
    def __init__(self, sr: int, n_fft: int, hop_length: int):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.melspec = MelSpectrogram(
            sample_rate = sr,
            n_fft = n_fft,
            hop_length = hop_length
        )
        self.amp_to_db = AmplitudeToDB()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.melspec(x)
        x = self.amp_to_db(x)
        return x


class smallCNNModel(nn.Module):
    def __init__(self, 
                 num_regression_outputs: int, 
                 num_classification_outputs: int, 
                 num_classes: List[int]):
        super().__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Output heads
        self.regression_head = nn.Linear(256, num_regression_outputs)
        self.classification_heads = [] # should use module list
        for i in range(num_classification_outputs):
            self.classification_heads.append(nn.Linear(256, num_classes[i]))
        
    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        
        # Apply global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Pass through the output heads
        regression_outputs = torch.sigmoid(self.regression_head(x))
        classification_outputs = []
        for head in self.classification_heads:
            classification_outputs.append(head(x))
        
        return regression_outputs, classification_outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1, stride: int = 1):
        super().__init__()
        self.expansion = expansion
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*self.expansion),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

        self.dropout = nn.Dropout2d(0.1)
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.residual(x)
        x += residual
        x = self.dropout(x)
        return self.final_activation(x)


class CNNModel(nn.Module):
    def __init__(self, 
                 num_regression_outputs: int, 
                 num_classification_outputs: int, 
                 num_classes: List[int]):
        super().__init__()
        
        # Initial layers with larger kernel
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(0.1)
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 256, stride=2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final feature processing
        self.fc_dropout = nn.Dropout(0.3)
        
        # Output heads
        self.regression_head = nn.Linear(256, num_regression_outputs)
        self.classification_heads = nn.ModuleList([
            nn.Linear(256, num_classes[i]) 
            for i in range(num_classification_outputs)
        ])

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels)
        )

    def get_embedding(self, x: Tensor) -> Tensor:
        # Initial processing
        x = self.initial(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_dropout(x)
        return x

    def forward(self, x: Tensor) -> tuple:
        x = self.get_embedding(x)
        
        # Output heads
        regression = torch.sigmoid(self.regression_head(x))
        classifications = [head(x) for head in self.classification_heads]
        
        return regression, classifications


if __name__ == '__main__':
    
    from data import VitalSoundDataModule
    from paths import CONFIGS_DIR
    from utils import load_yaml_config
    import lightning as L
    import os
    
    # Load configuration from YAML file
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'minimal_training.yaml'))

    # Set seed for reproducibility
    L.seed_everything(config['seed'])

    # Usage
    data_module = VitalSoundDataModule(config)
    data_module.setup()

    # Accessing the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    spec = Audio2LogMelSpec(
        config['model']['sr'], 
        config['model']['n_fft'], 
        config['model']['hop_length']
    )

    model = CNNModel(
        config['model']['num_regression_outputs'], 
        config['model']['num_classification_outputs'], 
        config['model']['num_classes']
    )

    # Calculate the total number of parameters
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    for batch in train_loader:
        x, y = batch
        x = spec(x)
        print(x.shape)
        print(y.shape)
        embeddings = model.get_embedding(x)
        regression, classifications = model(x)
        print(embeddings.shape)
        print(regression.shape)
        print([cls.shape for cls in classifications])
        break
