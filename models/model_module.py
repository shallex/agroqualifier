import lightning as L
import torch
from torch import optim, nn, utils, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torch.optim.lr_scheduler import ExponentialLR

from data.constants import MEAN, STD

class MandarinSegmentationModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.threshold = config.model.threshold
        self.map_metric = MeanAveragePrecision(iou_type=["segm"])


    def training_step(self, batch, batch_idx):
        images, targets = batch
        if batch_idx == 0 and self.global_step == 0:
            self.log_images(images=images, preds=None, targets=targets, split="training")

        output = self.model(images, targets)
        output = {f"train/{k}": v for k, v in output.items()}

        loss = sum(loss for loss in output.values())

        self.log("train/sum_loss", loss, prog_bar=True)
        self.log_dict(output)
        return loss

    def draw_mask(self, image, pred):
        masks = (pred["masks"] > self.threshold)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")
        return output_image
    
    def draw_pred_image(self, image, pred):
        pred_boxes = pred["boxes"].long()
        pred_labels = [f"Score: {score:.3f}" for score in pred["scores"]]
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
        output_image = self.draw_mask(output_image, pred)

        return output_image

    def log_images(self, images, preds, targets, split):
        images_to_show = []
        caption = []
        for i in range(min(5, len(images))):
            image = images[i].cpu()
            for t, m, s in zip(image, MEAN, STD):
                t.mul_(s).add_(m)
            image = torch.clip(image * 255, 0, 255).to(torch.uint8)

            if preds is not None:
                # Predicted image
                output_image = self.draw_pred_image(image, preds[i])
                images_to_show.append(output_image)
                num_preds = preds[i]["labels"].shape[0]
                caption.append(f"Predicted, num predicted objects {num_preds}")

            # Ground Truth
            output_image = self.draw_mask(image, targets[i])
            images_to_show.append(output_image)
            caption.append("Ground Truth")

        self.logger.log_image(key=f"{split} batch, thr={self.threshold}", images=images_to_show, caption=caption)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        
        images, targets = batch
        preds = self.model(images, targets)

        if batch_idx == 0:
            self.log_images(images, preds, targets, split="validation")

        preds = [{k: (v.squeeze(1).to(torch.bool) if k == "masks" else v) for k, v in p.items()} for p in preds]
        targets = [{k: (v.to(torch.bool) if k == "masks" else v) for k, v in p.items()} for p in targets]
        self.map_metric.update(preds, targets)

        return 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, self.config.training.gamma),
                "interval": "epoch",
                "frequency": 1,
            },
        }