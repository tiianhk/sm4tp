import torch
import torch.nn.functional as F
import lightning as L

from models import Audio2LogMelSpec, CNNModel
from torchmetrics import MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score

class VitalSoundMatching(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        # Initialize the mel spectrogram transformation
        self.mel_transform = Audio2LogMelSpec(config['model']['sr'], 
        config['model']['n_fft'], config['model']['hop_length'])
        
        # Initialize the CNN model
        self.model = CNNModel(config['model']['continuous_params_num'], 
        config['model']['class1_num'], config['model']['class2_num'], 
        config['model']['class3_num'])

        # eval
        # self.mae_list = [MeanAbsoluteError() for _ in range(7)]
        # F1 Metrics for classification outputs
        self.f1_class1 = MulticlassF1Score(num_classes=config['model']['class1_num'])
        self.f1_class2 = MulticlassF1Score(num_classes=config['model']['class2_num'])
        self.f1_class3 = MulticlassF1Score(num_classes=config['model']['class3_num'])

    def forward(self, x):
        # Apply mel spectrogram transformation
        x = self.mel_transform(x)
        
        # Forward pass through the CNN model
        continuous_output, class_output1, class_output2, class_output3 = self.model(x)
        
        return continuous_output, class_output1, class_output2, class_output3
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        continuous_out, class_out1, class_out2, class_out3 = self(x)
        
        # Compute MAE for continuous output (y[:,:7] corresponds to continuous values)
        continuous_loss = torch.tensor(.0, device=self.device)
        for i in range(7):
            continuous_loss += F.l1_loss(continuous_out[:, i], y[:, i])
        
        # Compute cross entropy loss for the classification tasks (using raw logits)
        class_loss1 = F.cross_entropy(class_out1, y[:, 7].long())  # Class 1 (class1_num)
        class_loss2 = F.cross_entropy(class_out2, y[:, 8].long())  # Class 2 (class2_num)
        class_loss3 = F.cross_entropy(class_out3, y[:, 9].long())  # Class 3 (class3_num)
        
        # Total loss
        total_loss = continuous_loss + class_loss1 + class_loss2 + class_loss3
        return total_loss

    def on_validation_epoch_start(self):
        self.maes = torch.zeros(7, device=self.device)
        self.step_num = torch.tensor(0., device=self.device)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        continuous_out, class_out1, class_out2, class_out3 = self(x)
        for i in range(7):
            self.maes[i] += F.l1_loss(continuous_out[:, i], y[:, i])
        self.f1_class1.update(class_out1, y[:, 7].long())
        self.f1_class2.update(class_out2, y[:, 8].long())
        self.f1_class3.update(class_out3, y[:, 9].long())
        self.step_num += 1.
    
    def on_validation_epoch_end(self):
        for i in range(10):
            pname = self.config['model']['parameter_names'][i]
            if i < 7:
                self.log(f'val-l1/{pname}', self.maes[i] / self.step_num)
            elif i == 7:
                self.log(f'val-f1/{pname}', self.f1_class1.compute())
            elif i == 8:
                self.log(f'val-f1/{pname}', self.f1_class2.compute())
            elif i == 9:
                self.log(f'val-f1/{pname}', self.f1_class3.compute())
        self.f1_class1.reset()
        self.f1_class2.reset()
        self.f1_class3.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == '__main__':
    
    from data import VitalSoundDataModule
    from lightning.pytorch.loggers import TensorBoardLogger
    from utils import load_yaml_config
    from paths import CONFIGS_DIR
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Load configuration from YAML file
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'minimal_training.yaml'))

    # Set seed for reproducibility
    L.seed_everything(config['seed'])

    # Usage
    data_module = VitalSoundDataModule(config)
    data_module.setup()

    model = VitalSoundMatching(config)

    trainer = L.Trainer(max_epochs=200, logger=TensorBoardLogger('lightning_logs', name='vitalsound_matching'))

    trainer.fit(model, data_module)

