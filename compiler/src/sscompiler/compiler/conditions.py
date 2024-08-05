import torch.nn as nn


def linear_precondition(layer):
    """
    Checks whether a given layer can be merged into an nn.Linear layer
    """

    equivalent_linear = layer
    if hasattr(layer, "get_equivalent_layer") and callable(layer.get_equivalent_layer):
        equivalent_linear = layer.get_equivalent_layer()

    if (
        isinstance(equivalent_linear, nn.Linear)
        and hasattr(equivalent_linear, "in_features")
        and hasattr(equivalent_linear, "out_features")
    ):
        return equivalent_linear

    raise TypeError(
        "linear_precondition failed: layer could not be merged into nn.Linear"
    )
