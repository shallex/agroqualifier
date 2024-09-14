from typing import Tuple, List, Dict
import random

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T


class MandarinSegmentationDataset(Dataset):
    def __init__(self, config, dataframe, split, transforms) -> None:
        super().__init__()
        self._config = config
        self._dataframe = dataframe
        self._split = split
        self._transforms = transforms
        if self._config.dataset.augmentation.Pad:
            self.pad_sizes = list(range(config.dataset.size // 8, config.dataset.size // 6, 10))
            self.pad = ()


    def create_polygon_mask(self, image_size, vertices):
        """
        Create a grayscale image with a white polygonal area on a black background.

        Parameters:
        - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
        - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                            of the polygon. Vertices should be in clockwise or counter-clockwise order.

        Returns:
        - PIL.Image.Image: A PIL Image object containing the polygonal mask.
        """

        # Create a new black image with the given dimensions
        mask_img = Image.new('L', image_size, 0)
        
        # Draw the polygon on the image. The area inside the polygon will be white (255).
        ImageDraw.Draw(mask_img, 'L').polygon(vertices, fill=(255))

        # Return the image with the drawn polygon
        return mask_img

    def _get_image_and_mask(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img = read_image(self._dataframe.loc[index, "image_path"])
        polygon_points = self._dataframe.loc[index, 'segmentation']
        mask_imgs = [
            self.create_polygon_mask(
                (img.shape[2], img.shape[1]),
                polygon[0]
            )
            for polygon in polygon_points
        ]
        masks = torch.concat([tv_tensors.Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool).to(torch.uint8) for mask_img in mask_imgs])
        return img, masks

    def __getitem__(self, index):
        num_objs = len(self._dataframe.loc[index, "area"])
        
        img, masks = self._get_image_and_mask(index)
        img = tv_tensors.Image(img)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(self._dataframe.loc[index, "bboxes"], format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = index
        target["area"] = torch.tensor(self._dataframe.loc[index, "area"], dtype=torch.float32)
        target["iscrowd"] = iscrowd

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        if self.split == "train" and self._config.dataset.augmentation.Pad and num_objs == 1 and random.random() < 0.5:
            pad_size = random.choice(self.pad_sizes)
            img, target = T.Pad(pad_size)(img, target)
        return img, target

    def __len__(self):
        return len(self._dataframe)
    