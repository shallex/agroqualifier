import json
from pathlib import Path

import pandas as pd
import torch
from torchvision.transforms import v2 as T

from data.dataset import MandarinSegmentationDataset
from data.constants import Split, ANNOTATION_FILENAME, MEAN, STD


def convert_xywh_to_xyxy(bbox):
    """
    Convert bounding box from (x, y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        bbox (list): Bounding box in (x, y, width, height) format.

    Returns:
        list: Bounding box in (x1, y1, x2, y2) format.
    """
    x, y, width, height = bbox
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return [x1, y1, x2, y2]


def get_dataframe(dataset_path: Path, split: str):
    annotation_file = dataset_path / split / ANNOTATION_FILENAME

    with open(annotation_file) as f:
        d = json.load(f)

    img_df = pd.DataFrame(d["images"]).drop(["license", "date_captured"], axis=1).rename(columns={"id": "image_id"})
    img_df["image_path"] = dataset_path / split / img_df["file_name"]

    anno_df = pd.DataFrame(d["annotations"]).drop(["id", "iscrowd"], axis=1)

    df = pd.merge(img_df, anno_df, on="image_id")
    df.set_index('image_id', inplace=True)
    df = df.groupby('image_id').agg({
        'segmentation': list,
        'bbox': list, 
        'category_id': 'first',
        'area': list,
        'file_name': 'first', 
        'height': 'first', 
        'width': 'first',
        'image_path': 'first'
    })

    df.rename(columns={'bbox': 'bboxes'}, inplace=True)
    df["bboxes"] = df["bboxes"].apply(lambda x: [convert_xywh_to_xyxy(bb) for bb in x])
    return df


def get_transform(train, config):
    transforms = []

    transforms.append(T.Resize((config.dataset.size, config.dataset.size)))
    if train:
        transforms.append(T.RandomHorizontalFlip(config.dataset.horizontal_flip))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    transforms.append(T.Normalize(mean=MEAN, std=STD))

    return T.Compose(transforms)


def build_dataset(config, split: str = Split.train.value):
    """
    Build a dataset from a given path.
    :param path_to_dataset: Path to the dataset.
    :param split: Split of the dataset to build.
    :return: A dataset.
    """
    dataset_path = config.dataset.dataset_path
    assert split in [member for member in Split.__members__]
    dataframe = get_dataframe(Path(dataset_path), split)

    transform = get_transform(split == Split.train.value, config)

    dataset = MandarinSegmentationDataset(dataframe, split, transform)
    return dataset


def collate_fn(data):
    images = [image for image, _ in data]
    targets = [target for _, target in data]
    return images, targets