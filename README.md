# Sound Matching for Timbre Perception
This repo contains code that trains and evaluates a sound matching model that predicts selected synth parameters of the Vital synthesizer. We are interested in whether, via this task, the model can learn meaningful representations that align with timbre similarity perception. We use [timbremetrics](https://github.com/tiianhk/timbremetrics) to evaluate this alignment.

## Setup
1. Install [uv](https://github.com/astral-sh/uv).
2. Clone this repository and open its directory:
```
git clone https://github.com/tiianhk/sm4tp.git
cd sm4tp
```
3. Start a new environment and install python dependencies:
```
uv venv --python 3.12.6 # the version used to develop this project
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data Generation
The data generation process is configured by `configs/generate_dataset.yaml`. \
Update the `data_dir` field to point to your own data directory. The current path `/data/scratch/acw751/vital_synth` is specific to my setup and will not work for you. \
Run the following command to generate 500k (audio, synth parameters) pairs under the data directory:
```
uv run src/generate_dataset.py --config configs/generate_dataset.yaml
```
This process will take **~8 hours** on a single CPU core. \
To test things first, set a smaller value for `num_samples` in the config before generating the full dataset.

## Training
