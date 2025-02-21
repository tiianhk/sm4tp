import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import lightning as L

from paths import DATA_DIR, CONFIGS_DIR
from utils import load_config


class VitalSoundDataset(Dataset):
    def __init__(self, idx_list, max_duration):
        self.idx_list = idx_list
        self.max_duration = max_duration
        
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        idx = self.idx_list[idx]
        wav_path = os.path.join(DATA_DIR, f"{idx}.wav")
        pt_path = os.path.join(DATA_DIR, f"{idx}.pt")
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # If audio duration is less than the max, pad it with zeros
        num_samples = waveform.shape[1]
        max_samples = int(self.max_duration * sample_rate)
        if num_samples < max_samples:
            padding = max_samples - num_samples
            # Pad the audio on the right side (last dimension)
            waveform = F.pad(waveform, (0, padding))
        
        # Load the target tensor
        target = torch.load(pt_path)
        
        return waveform, target


class VitalSoundDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.sample_rate = config['data']['sr']
        self.num_samples = config['data']['num_samples']
        self.batch_size = config['data']['batch_size']
        self.val_split = config['data']['val_split']
        self.max_duration = config['data']['max_duration']
        self.num_workers = config['data']['num_workers']
        
    def setup(self, stage=None):
        
        train_size = int(self.num_samples * (1 - self.val_split))
        val_size = self.num_samples - train_size
        all_indices = list(range(self.num_samples))
        
        train_indices, val_indices = random_split(all_indices, [train_size, val_size])
        
        # Create the datasets for train and validation
        self.train_dataset = VitalSoundDataset(train_indices, self.max_duration)
        self.val_dataset = VitalSoundDataset(val_indices, self.max_duration)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    
    # Load configuration from YAML file
    config = load_config(os.path.join(CONFIGS_DIR, 'minimal_training.yaml'))

    # Set seed for reproducibility
    L.seed_everything(config['seed'])

    # Usage
    data_module = VitalSoundDataModule(config)
    data_module.setup()

    # Accessing the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # print first batch shape
    for batch in train_loader:
        print(batch[0].shape)
        break