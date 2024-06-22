from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from omegaconf import DictConfig


# Tests https://github.com/ray-project/ray/issues/7293
class Callbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        r = result["info"][LEARNER_INFO]
        if DEFAULT_POLICY_ID in r:
            r = r[DEFAULT_POLICY_ID].get(LEARNER_STATS_KEY, r[DEFAULT_POLICY_ID])
        # assert r["model"]["foo"] == 42, result


def init_callbacks(cfg: DictConfig):
    return Callbacks