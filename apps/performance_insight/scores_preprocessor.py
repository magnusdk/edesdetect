import pickle
import queue
from concurrent.futures.thread import ThreadPoolExecutor

import apps.performance_insight.util as util
import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.model as model
import haiku as hk
import jax
from edesdetectrl.config import config
from edesdetectrl.environments.binary_classification import EDESClassificationBase_v0
from edesdetectrl.util import functional


def pre_process_files(filenames):
    print_progress_mod = int(len(filenames) / 10)

    get_video = util.video_getter()
    env = EDESClassificationBase_v0()
    network = functional.chainf(
        model.get_func_approx(env.action_space.n),
        hk.transform,
        hk.without_apply_rng,
    )
    with open(config["data"]["trained_params_path"], "rb") as f:
        trained_params = pickle.load(f)
    q = jax.jit(lambda s: network.apply(trained_params, s)[0])

    del env

    scores = {}

    def evaluate_average_reward(filename_enumeration):
        try:
            i, filename = filename_enumeration
            env = EDESClassificationBase_v0()
            env.seq_and_labels = get_video(filename)
            trajectory = env.generate_trajectory_using_q(q)
            scores[filename] = {
                "accuracy": trajectory.accuracy(),
                "balanced_accuracy": trajectory.balanced_accuracy(),
                "recall": trajectory.recall(),
                "precision": trajectory.precision(),
                "f1": trajectory.f1(),
            }
        except Exception as e:
            print(i, filename, e)
        finally:
            if i % print_progress_mod == 0 and i != 0:
                print(f"{i}/{len(filenames)}")

    video_queue = queue.Queue()
    with ThreadPoolExecutor() as e:
        for i, filename in enumerate(filenames):
            video_queue.put_nowait((i, filename))
            e.submit(lambda: evaluate_average_reward(video_queue.get()))

    return scores


def get_pre_processed_scores():
    try:
        scores_path = config["apps"]["performance_insight"]["pre_processed_scores_path"]
        with open(scores_path, "r") as f:
            return eval(f.read())
    except IOError:
        return {}  # Just an empty dict


if __name__ == "__main__":
    # We only bother to pre-process the TEST part of the dataset.
    filenames = echonet.get_filenames(config["data"]["filelist_path"], split="TEST")
    scores = pre_process_files(filenames)
    path = config["apps"]["performance_insight"]["pre_processed_scores_path"]
    with open(path, "w") as f:
        f.write(str(scores))
