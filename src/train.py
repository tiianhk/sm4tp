import os
import argparse
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from timbremetrics import TimbreMetric

from data import VitalSoundDataModule
from lightning_module import VitalSoundMatching
from utils import load_yaml_config
from paths import LOG_DIR, CHECKPOINTS_DIR


def get_ckpt_callbacks(config, metric, mode):
    test_score = f'{metric}_test'
    sanitized_metric = metric.replace('/', '-')
    return ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINTS_DIR, config['trainer']['version']),
        filename=f'{sanitized_metric}={{{metric}: .3f}}_{{{test_score}: .3f}}',
        monitor=metric,
        mode=mode,
        save_top_k=1,
        auto_insert_metric_name=False,
    )


def get_metrics_and_modes(config):
    metrics_and_modes = []
    prefixes = ['val-tm_task', 'val-tm_style_Gatys', 'val-tm_style_Huang']
    tm = TimbreMetric()
    distances = [d.__name__ for d in tm.distances]
    metrics = [m.__name__ for m in tm.metrics]
    for prefix in prefixes:
        for dist in distances:
            for metric in metrics:
                for split_idx in range(config['validation']['n_splits']):
                    mode = 'min' if metric == 'mae' else 'max'
                    metrics_and_modes.append((f'{prefix}-split{split_idx+1}/{dist}_{metric}', mode))
    return metrics_and_modes


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)

    L.seed_everything(config['seed'])

    data_module = VitalSoundDataModule(config)
    data_module.setup()

    model = VitalSoundMatching(config)

    ckpt_callbacks = [
        get_ckpt_callbacks(config, metric, mode)
        for metric, mode in get_metrics_and_modes(config)
    ]

    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'], 
        callbacks=ckpt_callbacks,
        logger=[
            TensorBoardLogger(LOG_DIR, name='tensorboard', version=config['trainer']['version']), 
            CSVLogger(LOG_DIR, name='csv', version=config['trainer']['version']),
        ],
        val_check_interval=config['trainer']['val_check_interval'],
        deterministic=True
    )

    trainer.validate(model, data_module.val_dataloader())
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
