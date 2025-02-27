from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity

from utils import list_datasets, get_audio, get_raw_true_dissim


def infer_audio_embeddings_with(model: nn.Module, num_samples: int):
    '''
    for this function to work, model should take input shape (n_samples)
    generally, model should take input shapes (..., n_samples)
    ref: torchaudio.transforms.MelSpectrogram takes (..., n_samples)
    '''
    if model.training:
        model.eval()
    audio = get_audio()
    
    embeddings = {}
    for d in audio.keys():
        embeddings[d] = _extract_dataset_embeddings(model, audio[d], num_samples)
    return embeddings


@torch.no_grad()
def _extract_dataset_embeddings(model: nn.Module, dataset: list, num_samples: int):
    
    first_param = next(model.parameters())
    dtype = first_param.dtype
    device = first_param.device
    embeddings = []
    for x in dataset:
        audio_tensor = torch.tensor(x["audio"], dtype=dtype, device=device)
        if audio_tensor.shape[-1] < num_samples:
            audio_tensor = F.pad(audio_tensor, (0, num_samples - audio_tensor.shape[0]))
        assert hasattr(model, 'get_embedding_for_one_shot')
        embedding = model.get_embedding_for_one_shot(audio_tensor)
        embedding = embedding.flatten()
        embeddings.append(embedding)
    
    if len(set([embedding.shape for embedding in embeddings])) > 1:
        raise ValueError(
            "The model is outputting embeddings of different shapes. "
            + "All embeddings must have the same shape."
        )
    
    return torch.stack(embeddings)


def get_pred_dissim(model: nn.Module, num_samples: int):
    emb = infer_audio_embeddings_with(model, num_samples)
    pred_dissim = {}
    for d in emb.keys():
        x = 1 - pairwise_cosine_similarity(emb[d])
        pred_dissim[d] = min_max_normalization(mask(x))
    return pred_dissim


def min_max_normalization(a):
    return (a - a.min()) / (a.max() - a.min()) # order-preserving


def mask(x):
    mask = torch.ones_like(x).triu(1)
    return mask * x # keep the upper triangle


def get_true_dissim(device):
    true_dissim = get_raw_true_dissim()
    for d in true_dissim.keys():
        x = torch.tensor(true_dissim[d], device=device)
        true_dissim[d] = min_max_normalization(mask(x))
    return true_dissim


def mae(pred, true):
    N = pred.shape[0]
    count = N * (N - 1) / 2
    absolute_error = torch.sum(torch.abs(pred - true))
    return absolute_error / count


def mse(pred, true):
    N = pred.shape[0]
    count = N * (N - 1) / 2
    squared_error = torch.sum((pred - true) ** 2)
    return squared_error / count


def compute_timbre_metrics(model: nn.Module, metrics: list[Callable], num_samples: int):
    pred_dissim = get_pred_dissim(model, num_samples)
    device = next(iter(pred_dissim.values())).device
    true_dissim = get_true_dissim(device)
    results = {}
    for metric in metrics:
        value = torch.tensor(0., device=device)
        for d in pred_dissim.keys():
            value += metric(pred_dissim[d], true_dissim[d])
        results[metric.__name__] = value / len(list(pred_dissim.keys()))
    return results

# the sample rate od sounds/ is also 44100 but is not check now


if __name__ == '__main__':
    
    import os
    import lightning as L
    from lightning_module import VitalSoundMatching
    from utils import load_yaml_config
    from paths import CONFIGS_DIR

    # load config
    config = load_yaml_config(os.path.join(CONFIGS_DIR, 'minimal_training.yaml'))
    
    # set random seed
    L.seed_everything(config['seed'])
    
    model = VitalSoundMatching(config)

    res = compute_timbre_metrics(
        model = model,
        metrics = [mae,mse],
        num_samples = int(config['data']['duration'] * config['model']['sr'])
    )
    print(res)
