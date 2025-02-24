import yaml

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