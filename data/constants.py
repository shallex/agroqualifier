import enum
from torchvision.transforms import v2 as T

ANNOTATION_FILENAME = "_annotations.coco.json"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Split(enum.Enum):
    train = "train"
    valid = "valid"
    test = "test"
