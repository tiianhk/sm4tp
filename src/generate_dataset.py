import argparse
import vita
from scipy.io import wavfile
import random
from tqdm import tqdm
import os
import torch

from paths import SRC_DIR, DATA_DIR
from utils import load_config, write_preset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # init synth
    synth = vita.Synth()
    synth.load_preset(os.path.join(SRC_DIR, 'basic_shapes.vital'))
    controls = synth.get_controls()
    controls['filter_1_on'].set(1.0)
    controls['stereo_routing'].set(0.0)

    # sample parameters
    random.seed(config['seed'])
    for i in tqdm(range(config['num_samples'])):

        y = torch.zeros(11)

        for idx, (param, details) in enumerate(config['parameter_ranges'].items()):
            min_, max_ = details['min'], details['max']
            if details['scale'] == 'indexed':
                value = random.randint(min_, max_)
                if 'value_mapping' in details:
                    value = details['value_mapping'][value]
                y_true = value - min_
            else:
                value = random.uniform(min_, max_)
                y_true = (value - min_) / (max_ - min_)
            controls[param].set(value)
            y[idx] = y_true
        
        pitch = random.randint(config['pitch']['min'], config['pitch']['max'])
        y[-1] = pitch - config['pitch']['min']

        audio = synth.render(pitch, config['velocity'], 1, 2)

        # write dataset
        wavfile.write(f'{DATA_DIR}/{i}.wav', config['sr'], audio[0])
        torch.save(y, f'{DATA_DIR}/{i}.pt')
        write_preset(synth, f'{DATA_DIR}/{i}.vital')

if __name__ == '__main__':
    main()