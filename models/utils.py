from models.model import MaskRCNN

def batch_dict_to_list_of_dicts(batch_dict):
    """
    Converts a dictionary with batch-style lists of values into a list of dictionaries.

    Args:
        batch_dict (dict): A dictionary where each key corresponds to a list of values.

    Returns:
        list: A list of dictionaries, where each dictionary contains a single set of values from the original lists.
    """
    # Unzipping the dictionary to list of tuples using zip(*...)
    # Then converting each tuple to a dictionary using dict(zip(...))
    keys = batch_dict.keys()
    values = zip(*batch_dict.values())
    
    list_of_dicts = [dict(zip(keys, value_tuple)) for value_tuple in values]
    
    return list_of_dicts


def build_model(config):
    if config.model.name == "MaskRCNN":
        model = MaskRCNN(config.model.backbone, config.model.num_classes)
    else:
        raise ValueError(f"Model {config.model.name} is not supported.")
    return model
