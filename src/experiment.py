import json
from typing import Dict
import torch.nn as nn
from pathlib import Path


class MyExperiment(nn.Module):
    def __init__(self, config_path: str):
        assert Path(config_path).suffix in ['.json'], f"Please pass a JSON config file"
        if not Path(config_path).is_file():
            raise FileNotFoundError(f"No json found at {config_path}")

        with open(config_path, 'r') as fr:
            self.experiment_config = json.load(fr)
        self.model_config = self.experiment_config.get('model_config', None)
        self.trainer_config = self.experiment_config.get('trainer_config', None)
        self.tokenizer_config = self.experiment_config.get('tokenizer_config', None)
        self.evaluation_config = self.experiment_config.get('eval_config', None)
        super(MyExperiment, self).__init__()

    def get_experiment_config(self) -> Dict:
        return self.experiment_config

    def get_model_config(self) -> Dict:
        return self.model_config

    def get_trainer_config(self) -> Dict:
        return self.trainer_config

    def get_tokenizer_config(self) -> Dict:
        return self.tokenizer_config

    def get_eval_config(self) -> Dict:
        return self.evaluation_config
