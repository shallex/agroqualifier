import lightning as L
import torch
from torch import optim, nn, utils, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


class MandarinSegmentationModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.map_metric = MeanAveragePrecision(iou_type=["segm"])

        

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets)

        loss = sum(loss for loss in output.values())
        
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def draw_mask(self, image, pred, threshold=0.7):
        masks = (pred["masks"] > threshold) #.squeeze(1)
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
            for i in range(len(images)):
                image = torch.clip(images[i].cpu() * 255, 0, 255)
                
                output_image = self.draw_mask(image, preds[i])
                images_to_show.append(output_image)
                caption.append("Predicted")

                output_image = self.draw_mask(image, targets[i])
                images_to_show.append(output_image)
                caption.append("Ground Truth")

            self.logger.log_image(key="validation batch", images=images_to_show, caption=caption)

        self.map_metric.update(preds, targets)
    
        return 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.training.learning_rate)
        return optimizer