import os
import random
import torch
from scipy.io import wavfile

from utils import start_vital_synth, strip_unit_and_get_value
from constants import SAMPLE_RATE

VITAL_PLUGIN_PATH = '/Library/Audio/Plug-Ins/VST3/Vital.vst3'
INIT_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'init.state')
PARAMETER_RANGES = {
    # continuous
    48:     (0., 0.4),      # env 1 attack time
    50:     (0.25, 0.4),    # env 1 decay time
    54:     (0., 1.),       # env 1 sustain
    102:    (0., 0.6),      # filter 1 blend
    104:    (0.2, 0.8),     # filter 1 cutoff
    377:    (0., 0.9),      # osc 1 distortion
    395:    (0., 0.3),      # osc 1 unison detune
    # discrete
    380:    (0., 0.54),     # osc 1 distortion type: 7 types
    396:    (0., 1.),       # osc 1 unison voices: 1-16 voices
    397:    (0., 1.),       # osc 1 wave frame: 7 basic shapes
}

if __name__ == '__main__':
    
    engine, vital = start_vital_synth(
        plugin_path=VITAL_PLUGIN_PATH, 
        state_path=INIT_STATE_PATH)
    
    random.seed(42)
    for i in range(500):

        print(f'synthesizing {i+1}th sample...')

        attack_and_decay = 0.
        y = torch.zeros(10)

        for j, (idx, interval) in enumerate(PARAMETER_RANGES.items()):
            
            # use random values for each parameter
            value = random.uniform(*interval)
            vital.set_parameter(idx, value)
            
            if j < 7: # continuous values (regression)
                y_true = (value - interval[0]) / (interval[1] - interval[0])
            else: # discrete values (classification)
                if j < 9:
                    value_ranges = vital.get_parameter_range(idx)
                else:
                    # not use get_parameter_range because it returns finer ranges
                    value_ranges = {
                        (0., 0.124): 0,
                        (0.124, 0.248): 1,
                        (0.248, 0.372): 2,
                        (0.372, 0.496): 3,
                        (0.496, 0.748): 4,
                        (0.748, 0.872): 5,
                        (0.872, 1.): 6
                    }
                for k, (start, end) in enumerate(value_ranges.keys()):
                    if value >= start and value < end:
                        y_true = k
                        break
            y[j] = y_true
            
            # accumulate attack and decay times
            if idx == 48 or idx == 50:
                attack_and_decay += strip_unit_and_get_value(vital, idx)

        torch.save(y, f'data/{i+1}.pt')
        vital.save_state(f'data/{i+1}.state')

        # one-shot note generation
        pitch = random.randint(48, 72)
        vel = 100
        start = 0.
        dur = random.uniform(attack_and_decay, 2.)
        vital.add_midi_note(pitch, vel, start, dur)
        engine.load_graph([
            (vital, [])
        ])
        release = strip_unit_and_get_value(vital, 52)
        engine.render(start + dur + release)
        audio = engine.get_audio()
        wavfile.write(f'data/{i+1}.wav', SAMPLE_RATE, audio[0])
        vital.clear_midi()
