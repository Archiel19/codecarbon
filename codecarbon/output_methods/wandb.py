import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

import wandb
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData


class WandbOutput(BaseOutput):
    """
    Send emissions data to Weights & Biases

    Implementation based on lightning.pytorch.loggers.wandb.WandbLogger
    """

    _REMOVE_KEYS = (
        "wue",
        "pue",
        "on_cloud",
        "tracking_mode",
        "ram_total_size",
        "latitude",
        "longitude",
        "gpu_model",
        "gpu_count",
        "cpu_model",
        "cpu_count",
        "codecarbon_version",
        "python_version",
        "cloud_provider",
        "os",
        "cloud_region",
        "region",
        "country_iso_code",
        "country_name",
        "experiment_id",
        "run_id",
        "project_name",
        "timestamp",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Path = ".",
        id: Optional[str] = None,
        offline: bool = False,
        project: Optional[str] = None,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        **kwargs: Any,
    ):

        super().__init__()
        self._offline = offline
        self._prefix = prefix
        self._experiment = experiment

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)

        # set wandb init arguments
        self._wandb_init: dict[str, Any] = {
            "name": name,
            "project": project,
            "dir": save_dir,
            "id": id,
            "resume": "allow",
        }
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")

    def __getstate__(self) -> dict[str, Any]:

        # Hack: If the 'spawn' launch method is used, the logger will get pickled and this `__getstate__` gets called.
        # We create an experiment here in the main process, and attach to it in the worker process.
        # Using wandb-service, we persist the same experiment even if multiple `Trainer.fit/test/validate` calls
        # are made.
        wandb.require("service")
        _ = self.experiment

        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    def experiment(self) -> Union["Run", "RunDisabled"]:

        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                warnings.warn(
                    "There is a wandb run already in progress and newly created instances of `WandbOutput` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbOutput`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

        return self._experiment

    def out(self, total: EmissionsData, delta: EmissionsData):
        emissions_dict = asdict(total)
        for k in self._REMOVE_KEYS:
            emissions_dict.pop(k)
        log_dict = {}
        for k, v in emissions_dict.items():
            if isinstance(v, list):
                for i, entry in enumerate(v):
                    log_dict[f"{self._prefix}/{k}_{i}"] = entry
            else:
                log_dict[f"{self._prefix}/{k}"] = v
        self.experiment.log(log_dict)

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        self.out(total, None)
