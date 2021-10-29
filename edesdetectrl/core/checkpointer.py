import os
import pickle
import shutil

from acme import core
from acme.agents import replay
from edesdetectrl.core import stepper, timer, tracking


class CheckPointer(stepper.Stepper):
    def __init__(
        self,
        agent: core.Saveable,
        reverb_replay: replay.ReverbReplay,
        mlflow_initializer: tracking.MLflowInitializer,
        checkpoints_dir: str,
        time_delta_minutes: float,
    ) -> None:
        self._agent = agent
        self._reverb_replay = reverb_replay

        os.makedirs(checkpoints_dir, exist_ok=True)
        self._checkpoints_dir = checkpoints_dir
        self._checkpoints_learner_path = checkpoints_dir + "/learner"
        self._checkpoints_misc_path = checkpoints_dir + "/misc"

        # Restore learner state from checkpoint
        try:
            with open(self._checkpoints_learner_path, "rb") as f:
                learner_state = pickle.load(f)
                self._agent.restore(learner_state)
        except FileNotFoundError:
            # #EAFP (https://devblogs.microsoft.com/python/idiomatic-python-eafp-versus-lbyl/)
            pass

        # Restore mlflow state
        try:
            with open(self._checkpoints_misc_path, "rb") as f:
                misc = pickle.load(f)
                mlflow_initializer.set_run_id(misc["mlflow_run_id"])
        except FileNotFoundError:
            pass

        self._checkpoint_timer = timer.Timer(time_delta_minutes * 60)

    def set_run_id(self, run_id):
        """Set the MLflow run_id so that the run can be restored."""
        misc = {}
        # Try to update from disk
        try:
            with open(self._checkpoints_misc_path, "rb") as f:
                misc_from_disk = pickle.load(f)
                misc.update(misc_from_disk)
        except FileNotFoundError:
            pass

        # Set the run_id and write to disk
        misc["mlflow_run_id"] = run_id
        with open(self._checkpoints_misc_path, "wb") as f:
            pickle.dump(misc, f)

    def step(self, *_):
        if self._checkpoint_timer.check():
            # Checkpoint reverb replay
            self._reverb_replay.server.localhost_client().checkpoint()

            # Save learner state
            learner_state = self._agent.save()
            with open(self._checkpoints_learner_path, "wb") as f:
                pickle.dump(learner_state, f)

            self._checkpoint_timer.reset()

    def _last_checkpointed_episode(self):
        # TODO: Rethink how to do this. Maybe the checkpointer should hold a reference to a training loop object and restore it?
        client = self._reverb_replay.server.localhost_client()
        server_info = client.server_info()
        num_episodes = server_info["priority_table"].num_episodes
        return num_episodes

    def shutdown(self):
        "Remove all checkpoint data when shutting down."
        shutil.rmtree(self._checkpoints_dir)
