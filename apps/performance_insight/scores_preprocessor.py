import queue
from concurrent.futures.thread import ThreadPoolExecutor

import apps.performance_insight.util as util
import coax
import edesdetectrl.dataloaders.echonet as echonet
import edesdetectrl.model as model
from edesdetectrl.config import config
from edesdetectrl.environments.binary_classification import EDESClassificationBase_v0


def pre_process_files(filenames):
    print_progress_mod = int(len(filenames) / 10)

    get_video = util.video_getter()
    env = EDESClassificationBase_v0()
    q = coax.Q(model.get_func_approx(env), env)
    q.params = coax.utils.load(config["data"]["trained_params_path"])
    del env

    scores = {}

    def evaluate_average_reward(filename_enumeration):
        try:
            i, filename = filename_enumeration
            env = EDESClassificationBase_v0()
            env.seq_and_labels = get_video(filename)
            trajectory = env.generate_trajectory_using_q(q)

            rewards = list(map(lambda item: item.r, trajectory))
            average_reward = sum(rewards) / len(rewards)
            scores[filename] = average_reward
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
