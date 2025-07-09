import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from sklearn.model_selection import KFold

from models import Audio2LogMelSpec, CNNModel
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score
from timbremetrics import TimbreMetric, list_datasets

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
        self.maes = nn.ModuleList()
        for i in range(self.n_rgs):
            self.maes.append(MeanAbsoluteError())
        self.f1 = nn.ModuleList()
        for i in range(self.n_cls):
            self.f1.append(MulticlassF1Score(num_classes=self.cls_nums[i]))

        self.n_splits = config['validation']['n_splits']
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=config['seed'])
        timbre_space_datasets = list_datasets()
        self.timbre_metrics = []
        self.timbre_metrics_online_test = []
        self.test_sets = []
        for valid_idx, test_idx in kf.split(timbre_space_datasets):
            valid_set = [timbre_space_datasets[i] for i in valid_idx]
            test_set = [timbre_space_datasets[i] for i in test_idx]
            self.timbre_metrics.append(
                TimbreMetric(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    sample_rate=config['model']['sr'], 
                    fixed_duration=config['data']['duration'],
                    datasets=valid_set,
                )
            )
            self.timbre_metrics_online_test.append(
                TimbreMetric(
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    sample_rate=config['model']['sr'], 
                    fixed_duration=config['data']['duration'],
                    datasets=test_set,
                )
            )
            self.test_sets.append(test_set)

    def get_task_embedding(self, x: Tensor) -> Tensor:
        x = self.mel_spec(x)
        x = self.model.unsqueeze_input(x)
        return self.model.get_task_embedding(x)

    def get_style_embedding_Gatys(self, x: Tensor) -> Tensor:
        x = self.mel_spec(x)
        x = self.model.unsqueeze_input(x)
        return self.model.get_style_embedding_Gatys(x)

    def get_style_embedding_Huang(self, x: Tensor) -> Tensor:
        x = self.mel_spec(x)
        x = self.model.unsqueeze_input(x)
        return self.model.get_style_embedding_Huang(x)

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_rgs, pred_cls = self(x)
        for i in range(self.n_rgs):
            self.maes[i].update(pred_rgs[:, i], y[:, i])
        for i in range(self.n_cls):
            self.f1[i].update(pred_cls[i], y[:, i + self.n_rgs].long())
    
    def on_validation_epoch_end(self):
        for i in range(self.n_rgs + self.n_cls):
            pname = self.config['model']['parameter_names'][i]
            if i < self.n_rgs:
                self.log(f'val-l1/{pname}', self.maes[i].compute())
            else:
                self.log(f'val-f1/{pname}', self.f1[i - self.n_rgs].compute())
        for i in range(self.n_rgs):
            self.maes[i].reset()
        for i in range(self.n_cls):
            self.f1[i].reset()

        for i in range(self.n_splits):
        
            results = self.timbre_metrics[i](self.get_task_embedding)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_task-split{i+1}/{dist}_{metric}', score)
            
            results = self.timbre_metrics[i](self.get_style_embedding_Gatys)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_style_Gatys-split{i+1}/{dist}_{metric}', score)

            results = self.timbre_metrics[i](self.get_style_embedding_Huang)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_style_Huang-split{i+1}/{dist}_{metric}', score)

        for i in range(self.n_splits):
            
            results = self.timbre_metrics_online_test[i](self.get_task_embedding)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_task-split{i+1}/{dist}_{metric}_test', score)
            
            results = self.timbre_metrics_online_test[i](self.get_style_embedding_Gatys)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_style_Gatys-split{i+1}/{dist}_{metric}_test', score)

            results = self.timbre_metrics_online_test[i](self.get_style_embedding_Huang)
            for dist, metrics in results.items():
                for metric, score in metrics.items():
                    self.log(f'val-tm_style_Huang-split{i+1}/{dist}_{metric}_test', score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['optimizer']['lr'])
        return optimizer

# ---------- example code ---------- #

if __name__ == '__main__':
    
    from data import VitalSoundDataModule
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
    from utils import load_yaml_config
    from paths import CONFIGS_DIR
    import os

    # Load configuration from YAML file
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'train_one_epoch.yaml'))

    # Set seed for reproducibility
    L.seed_everything(config['seed'])

    # Usage
    data_module = VitalSoundDataModule(config)
    data_module.setup()

    model = VitalSoundMatching(config)

    trainer = L.Trainer(max_epochs=1, 
                        logger=[
                            TensorBoardLogger('logs/', name='tensorboard'), 
                            CSVLogger('logs/', name='csv'),
                        ],
                        val_check_interval=0.05,
                        deterministic=True)

    trainer.validate(model, data_module.val_dataloader()) # evaluate before training
    trainer.fit(model, data_module)
