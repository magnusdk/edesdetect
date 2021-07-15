import os
import lz4.frame
import cloudpickle as pickle


class CheckpointManager:
    """
    Utility class for creating and restoring checkpoints for training.

    Currently only stores checkpoints for the model parameters and the training monitor, i.e.: not the sample buffer.
    """

    def __init__(self, model, train_monitor, filepath):
        self.model = model
        self.train_monitor = train_monitor
        self.filepath = filepath

    def add_checkpoint(self):
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
        with lz4.frame.open(self.filepath, "rb") as f:
            checkpointed_state = pickle.loads(f.read())
            self.model.params = checkpointed_state["model_params"]
            self.train_monitor.set_counters(
                checkpointed_state["train_monitor_counters"]
            )
