import dawdreamer as daw
from constants import SAMPLE_RATE, BUFFER_SIZE

def start_vital_synth(plugin_path, state_path=None):
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    vital = engine.make_plugin_processor('vital_synth', plugin_path)
    if state_path:
        vital.load_state(state_path)
    return engine, vital

def check_changes(synth, old_param, new_param):
    for i in range(len(old_param)):
        if old_param[i] != new_param[i]:
            print(f'Parameter {i}: {synth.get_parameter_name(i)} '
                  f'changed from {old_param[i]} to {new_param[i]}')

def strip_unit_and_get_value(synth, idx):
    return float(synth.get_parameter_text(idx).split()[0])

def change_parameters_manually(synth):
    num = synth.get_plugin_parameter_size()
    old_param = [synth.get_parameter(i) for i in range(num)]
    synth.open_editor()
    new_param = [synth.get_parameter(i) for i in range(num)]
    check_changes(synth, old_param, new_param)