# @OldAPIStack
from gymnasium.spaces import Dict

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class TorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.input_layer = nn.Linear(obs_space.shape[0], 32)
        self.output_layer = nn.Linear(32, num_outputs)
        self.activation = nn.ReLU()
        

    def forward(self, input_dict, state, seq_lens):
        x = self.input_layer(input_dict["obs"])
        x = self.activation(x)
        x = self.output_layer(x)
        return x, []

    def value_function(self):
        assert self.output_layer is not None, "must call forward first!"
        return torch.reshape(torch.mean(self.output_layer , -1), [-1])
    

def init_model():
    ModelCatalog.register_custom_model(
        "torch_model", TorchModel
    )