from dataclasses import asdict, dataclass

from simple_parsing import ArgumentParser


class SlimscaleParser(ArgumentParser):
    def parse_args(self, *args, **kwargs):
        parsed_args = super().parse_args(*args, **kwargs)
        additional_args = {}
        deletable = []
        for key, value in vars(parsed_args).items():
            if isinstance(
                value,
                (
                    SearchOptions,
                    ExperimentOptions,
                    ValidationOptions,
                ),
            ):
                for option_key, option_value in asdict(value).items():
                    additional_args[option_key] = option_value
                deletable.append(key)

        for key, value in additional_args.items():
            setattr(parsed_args, key, value)

        return parsed_args


@dataclass
class BaselineOptions:
    """Class for defining CLI arguments for baseline experiments"""

    baseline: str  # Baseline optimization to search over


@dataclass
class SearchOptions:
    """Class for defining CLI arguments for search experiments"""

    num_opts: int = 3  # Number of optimizations to apply in the search algorithm
    trials: int = 100  # LLM task to optimize over
    debug_mode: bool = False  # Useful for debugging. Does not create a new database


@dataclass
class ExperimentOptions:
    """Class for defining CLI arguments for all experiments"""

    task: str  # LLM task to optimize over
    model: str  # Huggingface model to use
    epochs: int = 5  # Number of epochs to train model for
    batch_size: int = 32  # The batch size to use when training the model. 32 by default
    max_length: int = 512  # The max length for tokenizing the dataset
    should_pad: bool = False  # whether we want to pad inputs to max length
    train_head: bool = (
        False  # Whether we want to train a classification head on the model or not
    )
    quantize: bool = False  # Whether we should quantize the model or not
    prune: bool = False  # Whether we should prune the model
    sparsity_ratio: float = (
        0.5  # If prune is set to True, this indicates how much of the model we want to prune
    )
    structured: bool = (
        False  # If prune is set to True, whether we want to do structured or unstructured pruning
    )


@dataclass
class ValidationOptions:
    """Class for defining CLI arguments for validation scripts"""

    database: str  # Database to query for optimal baseline hyperparams
