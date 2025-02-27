import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import lightning as L

from paths import CONFIGS_DIR
from utils import load_yaml_config


class VitalSoundDataset(Dataset):
    def __init__(self, data_dir, idx_list, duration):
        self.data_dir = data_dir
        self.idx_list = idx_list
        self.duration = duration
        
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        idx = self.idx_list[idx]
        audio_path = os.path.join(self.data_dir, f"{idx}.wav")
        synth_params_path = os.path.join(self.data_dir, f"{idx}.pt")
        
        # Load the audio file
        audio, sr = torchaudio.load(audio_path)
        
        # If audio duration is less than the max, pad it with zeros
        num_samples = audio.shape[1]
        target_sample_num = int(self.duration * sr)
        if num_samples < target_sample_num:
            padding = target_sample_num - num_samples
            # Pad the audio on the right side (last dimension)
            audio = F.pad(audio, (0, padding))
        
        # Load the synth parameters
        synth_params = torch.load(synth_params_path)
        
        return audio, synth_params


class VitalSoundDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config['data_dir']
        self.num_samples = config['data']['num_samples']
        self.batch_size = config['data']['batch_size']
        self.val_split = config['data']['val_split']
        self.duration = config['data']['duration']
        self.num_workers = config['data']['num_workers']
        
    def setup(self, stage=None):
        
        val_size = int(self.num_samples * self.val_split)
        train_size = self.num_samples - val_size
        all_indices = list(range(self.num_samples))
        
        train_indices, val_indices = random_split(all_indices, [train_size, val_size])
        assert len(train_indices) == train_size
        assert len(val_indices) == val_size
        
        # Create the datasets for train and validation
        self.train_dataset = VitalSoundDataset(self.data_dir, train_indices, self.duration)
        self.val_dataset = VitalSoundDataset(self.data_dir, val_indices, self.duration)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    
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

    # print first batch shape
    for batch in train_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break