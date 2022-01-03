import logging
from typing import Any, Callable, Mapping, Optional

import mlflow
from acme.utils import loggers
from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base, csv, filters, terminal


class MlflowLogger(loggers.Logger):
    def __init__(
        self,
        tracking_uri,
        experiment,
        run_id,
    ):
        # A bit dirty to set these global state variables... Oh well.
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_id=run_id)

    def write(self, values: loggers.LoggingData):
        values = {k: float(v) for k, v in values.items()}
        mlflow.log_metrics(values)

    def close(self):
        pass


def make_default_logger(
    label: str,
    tracking_uri,
    experiment,
    run_id,
    save_data: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
):
    if not print_fn:
        print_fn = logging.info
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
    mlflow_logger = MlflowLogger(tracking_uri, experiment, run_id)

    loggers = [terminal_logger, mlflow_logger]

    if save_data:
        loggers.append(csv.CSVLogger(label=label))

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)
    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, time_delta)

    return logger
