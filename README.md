# Sound Matching for Timbre Perception

## Setup
1. Install [uv](https://github.com/astral-sh/uv)
2. Clone this repository and open its directory
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
The data generation process is configured by `configs/generate_dataset.yaml`.
Update the `data_dir` field to point to your own data directory — the current path is specific to my setup and will not work for you.
Run the following command to generate 500k (audio, synth parameters) pairs:
```
uv run src/generate_dataset.py --config configs/generate_dataset.yaml
```
This process will take approximately **8 hours** on a single CPU core.
To test things first, set a smaller value for `num_samples` in the config before generating the full dataset.
