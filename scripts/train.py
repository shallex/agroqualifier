import argparse
import datetime
import git

import lightning as L
from lightning.pytorch.callbacks import ModelSummary
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from scripts.utils import load_config
from callbacks.callbacks import ValidationMetricCallback
from data.utils import build_dataset, collate_fn
from data.constants import Split
from models.model_module import MandarinSegmentationModel
from models.utils import build_model


def _parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argument_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file.",
        required=True,
    )
    argument_parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to the dataset.",
    )
    argument_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
        default=False,
    )
    return argument_parser.parse_args()


def train(config, confif_dict, config_name, debug=False):
    L.seed_everything(seed=config.training.seed, workers=True)
    L.pytorch.seed_everything(config.training.seed, workers=True)

    if debug:
        config.training.num_epochs = 1
        config.training.batch_size = min(2, config.training.batch_size)
        config.training.num_workers = 0
        limit_train_batches = 2
        limit_val_batches = 2
        log_every_n_steps = 1
    else:
        limit_train_batches = None
        limit_val_batches = None
        log_every_n_steps = 1

    train_dataset = build_dataset(config, Split.train.value)
    valid_dataset = build_dataset(config, Split.valid.value)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    base_model = build_model(config)
    model = MandarinSegmentationModel(
        base_model,
        config,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelSummary(max_depth=3),
        ValidationMetricCallback(),
    ]
    
    if config.logger.name == "wandb":
        logger = WandbLogger(
            project="AQ-VkusVill",
            name=f'Name: {config.logger.run_name}. Config: {config_name}. {datetime.datetime.now().strftime("%d %B %Y, %H:%M")}'
        )
        logger.experiment.config.update(confif_dict)
    else:
        logger = None
    
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        # deterministic=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )


def dict_to_namespace(config_dict):
    """
    Recursively converts a dictionary (including nested dictionaries) into a Namespace object.
    
    Args:
        config_dict (dict): Configuration dictionary, possibly containing nested dictionaries.
    
    Returns:
        Namespace: A Namespace object with attributes corresponding to the keys of the dictionary.
    """
    # Recursively convert nested dictionaries
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = dict_to_namespace(value)  # Recursively apply to nested dicts
    
    # Convert the updated dictionary into a Namespace object
    return argparse.Namespace(**config_dict)


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary by concatenating keys with a separator.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to use for concatenation (used during recursion).
        sep (str): The separator to use between concatenated keys.

    Returns:
        dict: A flattened dictionary with concatenated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_git_commit_hash():
    try:
        repo = git.Repo(path="./agroqualifier")
        sha = repo.head.object.hexsha
    except:
        sha = "unknown"
    return sha[:8]


if __name__ == "__main__":
    args = _parse_args()
    config_dict = load_config(args.config)
    if args.dataset_path is not None:
        config_dict["dataset"]["dataset_path"] = args.dataset_path
    git_commit_hash = get_git_commit_hash()
    config_dict["logger"]["run_name"] += f"_commit-{git_commit_hash}"
    flatten_config = flatten_dict(config_dict)
    config = dict_to_namespace(config_dict)

    train(config, flatten_config, args.config.split("/")[-1], args.debug)
