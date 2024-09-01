from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MandarinDataset(Dataset):
    def __init__(
        self, data_dir: Path, original_light: bool, IR_lamp_light: bool, train: bool, size=(300, 300), transform=None
    ):
        """Mandarin Dataset.

        Arguments:
          data_dir: Path - path to dataset directory, contained 4 folders with images
          original_light: bool - whether to use images made without Infra-Red lamp, only environmental light
          IR_lamp_light: bool - whether to use images made with Infra-Red lamp
          train: bool - is it train dataset or not, at train dataset use augmentations
          size: Tuple[int] - size for resizing images
          transform: training augmentation transforms
        """
        assert original_light or IR_lamp_light
        self.original_light = original_light
        self.IR_lamp_light = IR_lamp_light
        self.data_dir = data_dir
        self.resize = transforms.Resize(size)
        self.train = train
        self.transform = transform

        self.subdirs = {subdir.name: subdir for subdir in data_dir.iterdir() if subdir.is_dir()}
        self.subdirs_names = list(self.subdirs.keys())

        assert len(self.subdirs) == 4
        self.subdir_images_paths = {}

        for subdir_name, subdir in self.subdirs.items():
            self.subdir_images_paths[subdir_name] = list(subdir.glob("*.png"))

        self.images_paths = {"original_light": [], "IR_lamp_light": []}

        for subdir_name in self.subdirs_names:
            if subdir_name.startswith("Infra"):
                self.images_paths["IR_lamp_light"].extend(self.subdir_images_paths[subdir_name])

        for img_path in self.images_paths["IR_lamp_light"]:
            subdir = img_path.parent
            original_lamp_path = subdir.parent / f"NO_IR_{subdir.name.split('_')[-1]}" / img_path.name
            assert original_lamp_path.exists()
            self.images_paths["original_light"].append(original_lamp_path)

        for i in range(len(self.images_paths["original_light"])):
            assert self.images_paths["original_light"][i].name == self.images_paths["IR_lamp_light"][i].name

    def __len__(self):
        return len(self.images_paths["original_light"])

    def __getitem__(self, idx):
        if self.original_light:
            image = Image.open(self.images_paths["original_light"][idx]).convert("RGB")
            is_damaged = self.images_paths["original_light"][idx].parent.name.endswith("damaged")
            image = transforms.functional.pil_to_tensor(image)
            image = self.resize(image)

        if self.IR_lamp_light:
            ir_image = Image.open(self.images_paths["IR_lamp_light"][idx]).convert("RGB")
            is_damaged = self.images_paths["original_light"][idx].parent.name.endswith("damaged")
            ir_image = transforms.functional.pil_to_tensor(ir_image)
            ir_image = self.resize(ir_image)

        if self.original_light and self.IR_lamp_light:
            image = torch.cat((image, ir_image))
        elif not self.original_light and self.IR_lamp_light:
            image = ir_image

        # TODO: idk can we apply transforms to 6 channels image or not?
        if self.transform and self.train:
            image = self.transform(image)

        return image, torch.as_tensor(is_damaged).long()
