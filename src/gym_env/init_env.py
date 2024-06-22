from ray.tune.registry import register_env
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig

from .env import Env



def init_env(cfg: DictConfig):
    config = cfg.env_config
    register_env('Env', lambda config: Env(**config))