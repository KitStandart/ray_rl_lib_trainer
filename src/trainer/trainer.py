# @OldAPIStack
"""Example of using a custom ModelV2 Keras-style model."""

import argparse
import os
import hydra
from omegaconf import DictConfig
import ray
from ray import air, tune
from ray.tune.registry import get_trainable_cls

import sys
sys.path.append('/home/ivan/RL/ray_trainer/')
from src.gym_env.init_env import init_env
from src.model.model import init_model
from callbacks import init_callbacks
from metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='config')
def main(cfg: DictConfig):
    ray.init(num_cpus=cfg.train.num_cpus)

    init_env(cfg.env)
    init_model()
    callbacks = init_callbacks(cfg.train)
    
    config = (
        get_trainable_cls(cfg.model.algo_type)
        .get_default_config()
        .environment('Env')
        .framework(cfg.train.framework)
        .callbacks(callbacks)
        .training(
            model={
                "custom_model": "torch_model"
            }
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    stop = {
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": cfg.train.stop,
    }

    tuner = tune.Tuner(
        cfg.model.algo_type,
        param_space=config,
        run_config=air.RunConfig(stop=stop),
    )
    tuner.fit()

if __name__ == "__main__":
    main()