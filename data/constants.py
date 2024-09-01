import enum

ANNOTATION_FILENAME = "_annotations.coco.json"


class Split(enum.Enum):
    train = "train"
    valid = "valid"
    test = "test"
