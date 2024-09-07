import pytorch_lightning as pl
import lightning as L


class ValidationMetricCallback(L.Callback):
    def __init__(self) -> None:
        super().__init__()
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module) -> None:
        result = pl_module.map_metric.compute()
        result.pop("map_per_class")
        result.pop("mar_100_per_class")
        result = {f"val/{k}": v for k, v in result.items()}
        pl_module.log_dict(result)
