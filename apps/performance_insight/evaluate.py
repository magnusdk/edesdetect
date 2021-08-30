import coax
import edesdetectrl.model as model
import jax.numpy as jnp
import pandas as pd
from edesdetectrl.config import config
from edesdetectrl.environments.binary_classification import EDESClassificationBase_v0
import edesdetectrl.dataloaders.echonet as echonet


class Evaluator:
    def __init__(self) -> None:
        self.videos_dir = config["data"]["videos_path"]
        self.volumetracings_df = pd.read_csv(
            config["data"]["volumetracings_path"], index_col="FileName"
        )
        self.env = EDESClassificationBase_v0()
        self.q = coax.Q(model.get_func_approx(self.env), self.env)
        self.q.params = coax.utils.load(config["data"]["trained_params_path"])

    def evaluate(self, filename):
        traces = self.volumetracings_df.loc[filename]
        seq, labels = echonet.get_item(filename, traces, self.videos_dir)
        self.seq = seq
        self.env.seq_and_labels = seq, labels

        s = self.env.reset()
        done = False

        states = [s]
        actions = []
        rewards = []
        q_values = []
        while not done:
            qs = self.q(s)
            a = jnp.argmax(qs)
            s, r, done, info = self.env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            q_values.append(qs)

        def calc_advantage(t):
            d, s = t
            v = (d + s) / 2
            return d - v, s - v

        advantage = list(map(calc_advantage, q_values))
        return advantage, rewards
