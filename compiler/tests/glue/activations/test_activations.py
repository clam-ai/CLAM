import h5py
import numpy as np
import pandas as pd


def load_activations_from_hdf5(file_path):
    activation_dict = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            activation_dict[key] = f[key][()]
    return activation_dict


activation_dict = load_activations_from_hdf5(
    "/srv/shared_home/common-data/slimscale/clam_search/activations/gemma_no_ft/gemma_cb_activations.h5"
)


def activations_to_dataframe(activation_dict):
    # Determine the total number of samples and the flattened size of activations
    total_samples = sum(
        activations.shape[0] for activations in activation_dict.values()
    )
    flattened_size = np.prod(next(iter(activation_dict.values())).shape[1:])

    # Initialize an empty NumPy array to hold all flattened activations
    flattened_activations = np.empty((total_samples, flattened_size), dtype=np.float16)
    layer_names = []

    current_index = 0
    for layer_name, activations in activation_dict.items():
        num_samples = activations.shape[0]
        # Flatten the activations
        flattened_activations[current_index : current_index + num_samples, :] = (
            activations.reshape(num_samples, -1)
        )
        # Store layer names for the DataFrame index
        layer_names.extend([f"{layer_name}_sample_{i}" for i in range(num_samples)])
        current_index += num_samples

    # Convert to DataFrame
    df = pd.DataFrame(flattened_activations, index=layer_names)
    return df


df_activations = activations_to_dataframe(activation_dict)
print(df_activations)
