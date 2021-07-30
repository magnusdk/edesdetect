import os
import lz4.frame
import cloudpickle as pickle


class CheckpointManager:
    """
    Utility class for creating and restoring checkpoints for training.

    Saves and restores model parameters and training monitor counters.
    """

    def __init__(self, model, train_monitor, filepath):
        self.model = model
        self.train_monitor = train_monitor
        self.filepath = filepath

    def save_checkpoint(self):
        checkpointed_state = {
            "model_params": self.model.params,
            "train_monitor_counters": self.train_monitor.get_counters(),
        }

        dirpath = os.path.dirname(self.filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with lz4.frame.open(self.filepath, "wb") as f:
            f.write(pickle.dumps(checkpointed_state))

    def restore_latest(self):
        if os.path.isfile(self.filepath):
            with lz4.frame.open(self.filepath, "rb") as f:
                checkpointed_state = pickle.loads(f.read())
                self.model.params = checkpointed_state["model_params"]
                self.train_monitor.set_counters(
                    checkpointed_state["train_monitor_counters"]
                )
