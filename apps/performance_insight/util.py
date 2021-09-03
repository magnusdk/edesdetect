from datetime import datetime, timedelta
from functools import wraps

import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.environments.mixins as mixins
import pandas as pd
from edesdetectrl.config import config


def video_getter():
    videos_dir = config["data"]["videos_path"]
    volumetracings_df = pd.read_csv(
        config["data"]["volumetracings_path"], index_col="FileName"
    )

    def get_video(filename):
        traces = volumetracings_df.loc[filename]
        return echonet.get_item(filename, traces, videos_dir)

    return get_video


def calc_advantage(trajectory_item: mixins.TrajectoryItem):
    """Calculate the advantage of taking actions.

    Advantage here is defined as the q-value minus the average q-value over all q-values.
    It can be viewed as a kind of normalization step."""
    d, s = trajectory_item.q_values
    v = (d + s) / 2
    return d - v, s - v


# From: https://gist.github.com/ChrisTM/5834503
class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """

    def __init__(self, seconds=0, minutes=0, hours=0):
        self.throttle_period = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper
