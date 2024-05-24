from pathlib import Path
from dataclasses import dataclass
from torch import nn

from torchvision import transforms


@dataclass
class DatasetParams:
    data_dir = Path("/content/drive/MyDrive/Датасеты/Мандарины/cropped_mandarins")
    original_light = True
    IR_lamp_light = False
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    size = (300, 300)


@dataclass
class TrainingParams:
    num_epochs = 10
    l_rate = 1e-3
    train_batch_size = 16
    val_batch_size = 16


@dataclass
class ModelParams:
    architecture = "SimpleCNN"


@dataclass
class LoggingParams:
    wandb = True


@dataclass
class Params:
    dataset_params = DatasetParams()
    training_params = TrainingParams()
    model_params = ModelParams()


##################### Experiments #####################


class Exp_1(Params):
    def __init__(self):
        super().__init__()
        self.dataset_params.train_transforms = None


class Vadim_Exp_1_time_pred(Params):
  def __init__(self):
    super().__init__()
    self.problem_type = "time_prediction"
    self.dataset_params.train_transforms = None
    self.training_params.criterion = nn.MSELoss()
    self.model_params.output_channels = 1
    self.training_params.num_epochs = 5
  
class Vadim_Exp_2_time_pred(Params):
  def __init__(self):
    super().__init__()
    self.problem_type = "time_prediction"
    self.dataset_params.size = (232, 232)
    self.training_params.criterion = nn.MSELoss()
    self.model_params.architecture = "ResNet101_2_L"
    self.training_params.num_epochs = 20
    self.training_params.train_batch_size = 64
    self.model_params.output_channels = 1

class Vadim_Exp_3_time_pred(Params):
  def __init__(self):
    super().__init__()
    self.problem_type = "time_prediction"
    self.dataset_params.size = (232, 232)
    self.training_params.criterion = nn.MSELoss()
    self.model_params.architecture = "ResNet101_3_L"
    self.training_params.num_epochs = 30
    self.training_params.train_batch_size = 64
    self.model_params.output_channels = 1