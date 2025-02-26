import os
import yaml
import aifc
import numpy as np

from paths import STIMULI_DIR, TRUE_DISSIM_DIR

def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def write_preset(synth, preset_path):
    json_text = synth.to_json()
    # https://github.com/DBraun/Vita/issues/2
    if 'version":"99999.9.9"' in json_text:
        json_text = json_text.replace('version":"99999.9.9"', 'version":"1.5.5"')
    with open(preset_path, "w") as f:
        f.write(json_text)

def list_datasets():
    dataset_files = [
        f.replace("_dissimilarity_matrix.txt", "")
        for f in os.listdir(TRUE_DISSIM_DIR)
    ]
    return sorted(dataset_files)

def load_audio(dataset, audio_file):
    f = os.path.join(STIMULI_DIR, dataset, audio_file)
    aif = aifc.open(f)

    type_string, dtype = {
        1: (">i1", np.int8),
        2: (">i2", np.int16),
        4: (">i4", np.int32),
        8: (">i4", np.int64),
    }[aif.getsampwidth()]

    sr = aif.getframerate()
    n_frames = aif.getnframes()
    audio_bytes = aif.readframes(n_frames)
    audio = np.fromstring(audio_bytes, type_string) / np.iinfo(dtype).max

    return audio, sr

def load_dataset_audio(dataset):
    audio_files = os.listdir(os.path.join(STIMULI_DIR, dataset))
    audio_files = sorted(audio_files)

    audio_data = []
    for audio_file in audio_files:
        if os.path.splitext(audio_file)[1] != ".aiff":
            continue

        audio, sr = load_audio(dataset, audio_file)
        audio_data.append({"file": audio_file, "audio": audio, "sample_rate": sr})

    return audio_data

def get_audio():
    datasets = list_datasets()
    dataset_audio = {}
    for d in datasets:
        dataset_audio[d] = load_dataset_audio(d)
    return dataset_audio

def load_dissimilarity_matrix(dataset):
    f = os.path.join(TRUE_DISSIM_DIR, f"{dataset}_dissimilarity_matrix.txt")
    return np.loadtxt(f)

def get_raw_true_dissim():
    datasets = list_datasets()
    raw_true_dissim = {}
    for d in datasets:
        raw_true_dissim[d] = load_dissimilarity_matrix(d)
    return raw_true_dissim
