from torch.utils.data import DataLoader, random_split
import torch
from torch.optim.lr_scheduler import StepLR

from agroqualifier.models import SimpleCNN
from agroqualifier_src.dataset import MandarinDataset

models_dict = {"SimpleCNN": SimpleCNN}

optimizer_dict = {"Adam": torch.optim.Adam}

scheduler_dict = {"StepLR": StepLR}


def get_model(params):
    model = models_dict[params.model_params.architecture](params)
    return model


def get_loaders(params):
    """Get dataloader for MandarinDataset."""
    dataset = MandarinDataset(
        params.dataset_params.data_dir,
        params.dataset_params.original_light,
        params.dataset_params.IR_lamp_light,
        train=True,
        size=params.dataset_params.size,
        transform=params.dataset_params.train_transforms,
    )

    val_size = int(params.dataset_params.val_ratio * len(dataset))
    test_size = int(params.dataset_params.test_ratio * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    val_dataset.train = False
    test_dataset.train = False

    train_loader = DataLoader(train_dataset, batch_size=params.training_params.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.training_params.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.training_params.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_optimizer(params, model):
    return optimizer_dict[params.training_params.optimizer](
        model.parameters(), lr=params.training_params.l_rate, **params.training_params.optimimzer_kwargs
    )


def get_scheduler(params, optimizer):
    scheduler = scheduler_dict[params.training_params.scheduler](optimizer, **params.training_params.scheduler_kwargs)
    return scheduler
