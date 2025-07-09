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
                 num_classes: list[int]):
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
                 num_classes: list[int]):
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

        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output
            return hook
        self.initial[1].register_forward_hook(get_activation('initial_conv'))
        self.layer1[0].residual[1].register_forward_hook(get_activation('layer1_conv_0'))
        self.layer2[0].residual[1].register_forward_hook(get_activation('layer2_conv_0'))
        self.layer3[0].residual[1].register_forward_hook(get_activation('layer3_conv_0'))
        self.layer4[0].residual[1].register_forward_hook(get_activation('layer4_conv_0'))

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride),
            ResidualBlock(out_channels, out_channels)
        )

    def get_task_embedding(self, x: Tensor) -> Tensor:
        # Initial processing
        x = self.initial(x) # (B, 1, H, W) -> (B, 64, H/4, W/4)
        
        # Residual blocks
        x = self.layer1(x) # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        x = self.layer2(x) # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        x = self.layer3(x) # (B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
        x = self.layer4(x) # (B, 256, H/16, W/16) -> (B, 256, H/32, W/32)
        
        # Global pooling and flatten
        x = self.global_pool(x) # (B, 256, H/32, W/32) -> (B, 256, 1, 1)
        x = torch.flatten(x, start_dim=1) # (B, 256, 1, 1) -> (B, 256)
        x = self.fc_dropout(x) # (B, 256) -> (B, 256)
        return x

    def get_style_embedding_Gatys(self, x: Tensor) -> Tensor:
        """
        Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. 
        "Image style transfer using convolutional neural networks." 
        CVPR 2016.
        """
        x = self.get_task_embedding(x) # feed forward computation to trigger hooks
        style_x = []
        size_counter = 0
        for act in self.activations.values():
            B, C, H, W = act.shape
            size_counter += C ** 2
            act = act.view(B, C, -1)
            gram = torch.bmm(act, act.transpose(1, 2)) / (2* H * W)
            gram = gram.view(B, -1)
            style_x.append(gram)
        style_x = torch.cat(style_x, dim=1)
        assert style_x.shape[1] == size_counter
        return style_x

    def get_style_embedding_Huang(self, x: Tensor) -> Tensor:
        """
        Huang, Xun, and Serge Belongie. 
        "Arbitrary style transfer in real-time with adaptive instance normalization." 
        ICCV 2017.
        """
        x = self.get_task_embedding(x)  # Feed forward computation to trigger hooks
        style_x = []
        size_counter = 0
        for act in self.activations.values():
            B, C, H, W = act.shape
            size_counter += 2 * C  # Each channel contributes mean and std
            act = act.view(B, C, -1)  # Reshape to (B, C, H*W)
            mean = act.mean(dim=2)  # (B, C)
            std = act.std(dim=2, correction=0)  # (B, C) - Uses correction=0 for consistency
            style_x.append(torch.cat([mean, std], dim=1))  # (B, 2*C)
        style_x = torch.cat(style_x, dim=1)  # Concatenate across all layers
        assert style_x.shape[1] == size_counter
        return style_x

    def unsqueeze_input(self, x: Tensor) -> Tensor:
        for _ in range(4 - x.dim()):
            x = x.unsqueeze(0)
        return x

    def forward(self, x: Tensor) -> tuple:
        x = self.get_task_embedding(x)
        
        # Output heads
        regression = torch.sigmoid(self.regression_head(x))
        classifications = [head(x) for head in self.classification_heads]
        
        return regression, classifications

# ---------- example code ---------- #

if __name__ == '__main__':
    
    from data import VitalSoundDataModule
    from paths import CONFIGS_DIR
    from utils import load_yaml_config
    import lightning as L
    import os
    
    # Load configuration from YAML file
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'train_one_epoch.yaml'))
    config['data']['num_workers'] = 0 # testing with a single CPU core

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
        print('\nFor one batch of training data:')
        
        x, y = batch
        x = spec(x)
        print('input spectrogram shape:', x.shape)
        print('label shape:', y.shape)
        
        task_emb = model.get_task_embedding(x)
        Gatyes_emb = model.get_style_embedding_Gatys(x)
        Huang_emb = model.get_style_embedding_Huang(x)
        print('task embedding shape:', task_emb.shape)
        print('Gatyes embedding shape:', Gatyes_emb.shape)
        print('Huang embedding shape:', Huang_emb.shape)
        
        regression, classifications = model(x)
        print('regression prediction shape:', regression.shape)
        print('classfication prediction shapes:', [cls_.shape for cls_ in classifications])
        
        break
