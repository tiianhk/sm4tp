from typing import Callable
import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity

from utils import list_datasets, get_audio, get_raw_true_dissim


def infer_audio_embeddings_with(model: nn.Module):
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
        embeddings[d] = _extract_dataset_embeddings(model, audio[d])
    return embeddings


@torch.no_grad()
def _extract_dataset_embeddings(model: nn.Module, dataset):
    
    first_param = next(model.parameters())
    dtype = first_param.dtype
    device = first_param.device
    embeddings = []
    for x in dataset:
        audio_tensor = torch.tensor(x["audio"], dtype=dtype, device=device)
        embedding = model(audio_tensor)
        embedding = embedding.flatten()
        embeddings.append(embedding)
    
    if len(set([embedding.shape for embedding in embeddings])) > 1:
        raise ValueError(
            "The model is outputting embeddings of different shapes. "
            + "All embeddings must have the same shape."
        )
    
    return torch.stack(embeddings)


def get_pred_dissim(model: nn.Module):
    emb = infer_audio_embeddings_with(model)
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


def compute_timbre_metrics(model: nn.Module, metrics: list[Callable]):
    pred_dissim = get_pred_dissim(model)
    device = next(iter(pred_dissim.values())).device
    true_dissim = get_true_dissim(device)
    results = {}
    for metric in metrics:
        value = torch.tensor(0., device=device)
        for d in pred_dissim.keys():
            value += metric(pred_dissim[d], true_dissim[d])
        results[metric.__name__] = value / len(list(pred_dissim.keys()))
    return results
