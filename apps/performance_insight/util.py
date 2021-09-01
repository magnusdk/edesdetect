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
