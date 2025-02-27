import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L

from models import Audio2LogMelSpec, CNNModel
from torchmetrics.classification import MulticlassF1Score

class VitalSoundMatching(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.mel_spec = Audio2LogMelSpec(
            config['model']['sr'], 
            config['model']['n_fft'], 
            config['model']['hop_length']
        )

        self.n_rgs = config['model']['num_regression_outputs']
        self.n_cls = config['model']['num_classification_outputs']
        self.cls_nums = config['model']['num_classes']
    
        self.model = CNNModel(self.n_rgs, self.n_cls, self.cls_nums)

        # eval
        self.f1 = []
        for i in range(self.n_cls):
            self.f1.append(MulticlassF1Score(num_classes=self.cls_nums[i]))

    def get_embedding_for_one_shot(self, x: Tensor) -> Tensor:
        
        # compute mel spectrogram
        x = self.mel_spec(x)

        # add batch and channel dim if there is not
        for _ in range(4 - x.dim()):
            x = x.unsqueeze(0)

        # forward pass through the net
        embeddings = self.model.get_embedding(x)
        
        return embeddings

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        
        # compute mel spectrogram
        x = self.mel_spec(x)

        # forward pass through the net
        pred_rgs, pred_cls = self.model(x)
        
        return pred_rgs, pred_cls
    
    def training_step(self, batch, batch_idx):
        
        # get batch data
        x, y = batch
        
        # forward pass
        pred_rgs, pred_cls = self(x)
        
        # compute loss
        loss = torch.tensor(.0, device=self.device)
        for i in range(self.n_rgs):
            loss += F.l1_loss(pred_rgs[:, i], y[:, i])
        for i in range(self.n_cls):
            loss += F.cross_entropy(pred_cls[i], y[:, i + self.n_rgs].long())
        
        return loss

    def on_validation_epoch_start(self):
        self.maes = torch.zeros(self.n_rgs, device=self.device)
        self.step_num = torch.tensor(0., device=self.device)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_rgs, pred_cls = self(x)
        for i in range(self.n_rgs):
            self.maes[i] += F.l1_loss(pred_rgs[:, i], y[:, i])
        for i in range(self.n_cls):
            self.f1[i].update(pred_cls[i], y[:, i + self.n_rgs].long())
        self.step_num += 1.
    
    def on_validation_epoch_end(self):
        for i in range(self.n_rgs + self.n_cls):
            pname = self.config['model']['parameter_names'][i]
            if i < self.n_rgs:
                self.log(f'val-l1/{pname}', self.maes[i] / self.step_num)
            else:
                self.log(f'val-f1/{pname}', self.f1[i - self.n_rgs].compute())
        for i in range(self.n_cls):
            self.f1[i].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == '__main__':
    
    from data import VitalSoundDataModule
    from lightning.pytorch.loggers import TensorBoardLogger
    from utils import load_yaml_config
    from paths import CONFIGS_DIR
    import os

    # Load configuration from YAML file
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'minimal_training.yaml'))

    # Set seed for reproducibility
    L.seed_everything(config['seed'])

    # Usage
    data_module = VitalSoundDataModule(config)
    data_module.setup()

    model = VitalSoundMatching(config)

    trainer = L.Trainer(max_epochs=2, logger=TensorBoardLogger('lightning_logs', name='vitalsound_matching'))

    trainer.fit(model, data_module)

