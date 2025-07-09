# Sound Matching for Timbre Perception

## Setup
1. Install uv
2. Clone this repository and open its directory
```
git clone https://github.com/tiianhk/sm4tp.git
cd sm4tp
```
3. Start a new environment and install python dependencies:
```
uv venv --python 3.12.6
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data Generation
```
uv run src/generate_dataset.py --config configs/generate_dataset.yaml
```
