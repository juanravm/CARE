import re
import sys
import wandb
import yaml
import json
from typing import Any


def setup_hyperparameters(config_file: str) -> tuple[dict, Any]:
    """
    Load hyperparameters from a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.
        run_id (str | None): Optional W&B run ID for resuming.
        enable_wandb (bool): If False, skip initializing W&B (useful for debugging).

    Returns:
        tuple: (hyperparameters dict, run-like logger)
    """

    def init_wandb_run(hyperparameters: dict):
        """
        Initialize a Weights & Biases run with sane defaults for this project.
        """
        project = "CARE"
        entity = "juanravm-vall-d-hebron-institute-of-oncology"
        prefix = "CARE-"

        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        pat = re.compile(rf"^{re.escape(prefix)}(?P<n>\d+)$")
        max_n = 0
        for r in runs:
            if r.name is None:
                continue
            m = pat.match(r.name)
            if m:
                max_n = max(max_n, int(m.group("n")))

        run_name = f"{prefix}{max_n + 1}"

        init_kwargs = {
            "project": project,
            "entity": entity,
            "config": hyperparameters,
            "id": run_name,
            "settings": wandb.Settings(_disable_stats=True),
        }

        run = wandb.init(**init_kwargs)
        return run

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error while reading config file: {e}")
        sys.exit(1)

    hyperparameters = {
        # TODO: Define the hyperparameters to use
    }

    run = init_wandb_run(
        hyperparameters,
    )

    def stringify_unsupported(obj):
        if isinstance(obj, (dict, list, tuple)):
            return json.loads(json.dumps(obj))  # convierte a JSON-safe types
        elif obj is None:
            return "None"
        else:
            return str(obj)

    wandb_hparams = {
        k: stringify_unsupported(v)
        for k, v in {
            # TODO: Define the hyperparameters to log
        }.items()
    }

    run.config.update(wandb_hparams, allow_val_change=True)

    return hyperparameters, run
