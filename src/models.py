from typing import List
import torch
import torch.nn as nn
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

class CNNModel(nn.Module):
    def __init__(self, num_regression_outputs: int, 
        num_classification_outputs: int, num_classes: List[int]):
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
        self.classification_heads = []
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
        regression_output = torch.sigmoid(self.regression_head(x))
        classification_outputs = []
        for head in self.classification_heads:
            classification_outputs.append(head(x))
        
        return regression_output, classification_outputs

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

    spec = Audio2LogMelSpec(config['model']['sr'], 
        config['model']['n_fft'], config['model']['hop_length'])

    model = CNNModel(config['model']['continuous_params_num'], 
        config['model']['class1_num'], config['model']['class2_num'], 
        config['model']['class3_num'])

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    for batch in train_loader:
        x, y = batch
        x = spec(x)
        continuous_output, class_output1, class_output2, class_output3 = model(x)
        print(x.shape)
        print(y.shape)
        print(continuous_output.shape)
        print(class_output1.shape)
        print(class_output2.shape)
        print(class_output3.shape)
        break