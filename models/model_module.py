import lightning as L
import torch
from torch import optim, nn, utils, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

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
        output = self.model(images, targets)
        output = {f"train/{k}": v for k, v in output.items()}

        loss = sum(loss for loss in output.values())

        self.log("train/sum_loss", loss, prog_bar=True)
        self.log_dict(output)
        return loss

    def draw_mask(self, image, pred):
        masks = (pred["masks"] > self.threshold)
        output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")
        return output_image

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        
        images, targets = batch
        preds = self.model(images, targets)
        preds = [{k: (v.squeeze(1).to(torch.uint8) if k == "masks" else v) for k, v in p.items()} for p in preds]

        if batch_idx == 0:
            images_to_show = []
            caption = []
            for i in range(min(5, len(images))):
                image = images[i].cpu()
                for t, m, s in zip(image, MEAN, STD):
                    t.mul_(s).add_(m)
                image = torch.clip(image * 255, 0, 255)

                output_image = self.draw_mask(image, preds[i])
                images_to_show.append(output_image)
                caption.append("Predicted")

                output_image = self.draw_mask(image, targets[i])
                images_to_show.append(output_image)
                caption.append("Ground Truth")

            self.logger.log_image(key=f"validation batch, thr={self.threshold}", images=images_to_show, caption=caption)

        self.map_metric.update(preds, targets)
    
        return 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.training.learning_rate)
        return optimizer